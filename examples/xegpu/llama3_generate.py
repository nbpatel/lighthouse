"""Run the Llama-3.2 forward on the Intel GPU (XeGPU) with REAL HuggingFace weights.

Unlike llama3.py (random weights, toy dims, self-check against a numpy ref), this
driver loads a real Llama-3.2 checkpoint, tokenizes a real prompt, does the token
embedding on the host, runs the GPU forward, and reports the predicted next token.

It reuses the same payload (llama3_payload.build_llama_payload) and schedule
(llama3_schedule.build_combined_schedule) as llama3.py -- only the dims, weights,
and input come from the checkpoint instead of np.random.

  python llama3_generate.py --model ../../models/llama-3.2-1b \
      --prompt "The capital of France is" [--n-layers N] [--seq-len T]

--n-layers truncates the model to the first N transformer blocks (default: all 16),
which is how we de-risk compilation at real width before running the full stack.
--seq-len pads/truncates the token sequence to T (must match the compiled T).
"""

import argparse
import numpy as np
from mlir import ir

from lighthouse import dialects as lh_dialects
from lighthouse.pipeline.driver import TransformDriver
from lighthouse.execution.runner import Runner
from lighthouse.execution import GPUMemoryManager
from lighthouse.schedule.xegpu import xegpu_to_binary, XeGPUParameterSelector
from llama3_payload import build_llama_payload
from llama3_schedule import build_combined_schedule
from llama3_weights import load_llama_weights, rope_tables_from_config


def main():
    parser = argparse.ArgumentParser(
        description="Llama-3.2 forward with real HF weights on the Intel GPU (XeGPU).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="../../models/llama-3.2-1b",
                        help="Path to the HF checkpoint directory.")
    parser.add_argument("--prompt", default="The capital of France is",
                        help="Text prompt to run the forward pass on.")
    parser.add_argument("--n-layers", type=int, default=None,
                        help="Truncate to the first N transformer blocks (default: all).")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Compiled sequence length T (default: padded token count).")
    parser.add_argument("--max-new-tokens", type=int, default=0,
                        help="If >0, greedily generate this many tokens (re-run the "
                             "forward per step, append argmax). 0 = just report the "
                             "single next token (default).")
    parser.add_argument("--vocab-cap", type=int, default=None,
                        help="DIAGNOSTIC: shrink the output (lm_head) width to this many "
                             "columns. Transformer block + embeddings stay full-width; "
                             "only the final logits are truncated. Isolates whether the "
                             "128256-wide output matmul is what crashes binary codegen.")
    parser.add_argument("--no-causal", action="store_true",
                        help="Disable causal masking.")
    parser.add_argument("--dump", type=str, default=None,
                        choices=["initial", "schedule", "tiled", "vectorized",
                                 "bufferized", "inner-tiled", "gpu-outlining",
                                 "xegpu-initial", "xegpu-wg", "final"],
                        help="Print the IR at the given stage and exit.")
    args = parser.parse_args()
    causal = not args.no_causal

    # ---- load real weights + config, and tokenize the prompt ----
    W = load_llama_weights(args.model, n_layers=args.n_layers)
    cfg = W["cfg"]
    C = cfg["hidden_size"]
    H = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    hidden = cfg["intermediate_size"]
    vocab = cfg["vocab_size"]
    if args.vocab_cap:  # diagnostic: truncate only the output width
        vocab = args.vocab_cap
        W["lmw"] = np.ascontiguousarray(W["lmw"][:, :vocab])
    hs = cfg["head_dim"]
    n_layers = len(W["layers"])
    eps = cfg["rms_norm_eps"]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(args.prompt, return_tensors="np")["input_ids"][0].astype(np.int64)
    n_tok = len(ids)
    T = args.seq_len if args.seq_len else n_tok
    if n_tok > T:
        ids = ids[:T]
        n_tok = T
    print(f"prompt {args.prompt!r} -> {n_tok} tokens; T={T}, C={C}, H={H}, n_kv={n_kv}, "
          f"hidden={hidden}, vocab={vocab}, n_layers={n_layers}")

    # host embedding lookup: x[t] = embed_tokens[ids[t]]. Rows past the prompt (pad)
    # are zeros; with causal masking they don't affect the prompt positions' logits.
    x = np.zeros((T, C), np.float32)
    x[:n_tok] = W["embeddings"][ids]

    # RoPE tables from the real config (theta=500000 + llama3 NTK scaling).
    cos, sin = rope_tables_from_config(cfg, T)

    # ---- XeGPU tiling params (same structure as llama3.py) ----
    param_selector = XeGPUParameterSelector()
    mm_params = param_selector.get_parameters_dict((T, C, C))
    mm_params["gpu_specs"] = param_selector.gpu_specs
    ln_params = {"wg_rows": 64, "sg_rows": 8, "subgroup_size": 16,
                 "reduction_step_size": 16, "T": T}
    fa_params = {"batch_size": 1, "num_heads": H, "n_ctx": T, "n_head": hs,
                 "wg_rows": 128, "sg_rows": 16, "subgroup_size": 16,
                 "inner_loop_tile_size": 64, "causal": causal}

    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        mod, kinds, mm_shapes = build_llama_payload(
            "payload", T, C, hidden, vocab, n_layers, H, n_kv, eps=eps)
        if args.dump == "initial":
            print(mod); print("KINDS:", kinds); return

        shape_params = {}
        for shp in mm_shapes:
            if shp not in shape_params:
                p = param_selector.get_parameters_dict(shp)
                p["gpu_specs"] = param_selector.gpu_specs
                shape_params[shp] = p
        mm_params_list = [dict(shape_params[shp]) for shp in mm_shapes]

        sched = build_combined_schedule(
            dict(mm_params), dict(ln_params), kinds,
            stop_at_stage=(args.dump or ""), fa_params=dict(fa_params),
            mm_params_list=mm_params_list)
        if args.dump == "schedule":
            print(sched); return
        schedules = [sched]
        if not args.dump or args.dump == "final":
            schedules.append(xegpu_to_binary())
        payload = TransformDriver(schedules).apply(mod)
        if args.dump:
            print(payload); return
        print(f"LOWERED OK: 'llama' to {len(kinds)} kernels in one module")

        # ---- run on the GPU ----
        runner = Runner(payload, mem_manager_cls=GPUMemoryManager,
                        shared_libs=["libmlir_levelzero_runtime.so"])
        out = np.zeros((T, vocab), np.float32)
        cb = Runner.get_gpu_argument_access_callback(out, arg_index=0)
        # arg order matches build_llama_payload: out, x, cos, sin, then per layer
        # [an,wq,wk,wv,wo,fn,w1,w2,w3], then final [fn_w, lmw]. Weights/cos/sin are
        # fixed across generation steps; only `x` (the embedded sequence) changes,
        # so we build `host` once and just overwrite host[1] each step.
        host = [x, cos, sin]
        for lw in W["layers"]:
            host += [lw["an"], lw["wq"], lw["wk"], lw["wv"], lw["wo"],
                     lw["fn"], lw["w1"], lw["w2"], lw["w3"]]
        host += [W["fn_w"], W["lmw"]]

        def forward(seq_ids, n):
            """Embed the first `n` tokens of seq_ids into x, run the GPU forward,
            return the logits row at the last real position (n-1)."""
            xb = np.zeros((T, C), np.float32)
            xb[:n] = W["embeddings"][seq_ids[:n]]
            runner.execute(host_input_buffers=[out, xb] + host[1:],
                           payload_function_name="payload",
                           argument_access_callback=cb)
            return out[n - 1].copy()

        # ---- single next token: report top-5 ----
        last = forward(ids, n_tok)
        top = np.argsort(last)[::-1][:5]
        print(f"\nprompt: {args.prompt!r}")
        print("top-5 next tokens (GPU):")
        for i in top:
            print(f"  {int(i):7d}  {tok.decode([int(i)])!r:20s} logit={last[i]:.3f}")
        print(f"argmax next token id: {int(last.argmax())}  "
              f"-> {tok.decode([int(last.argmax())])!r}")

        # ---- optional greedy generation loop (re-run forward per token, no KV cache) ----
        if args.max_new_tokens > 0:
            seq = list(ids)
            for _ in range(args.max_new_tokens):
                if len(seq) >= T:
                    print(f"[stop: sequence reached compiled T={T}]")
                    break
                logits = forward(np.array(seq, np.int64), len(seq))
                seq.append(int(logits.argmax()))
            print(f"\ngenerated ({len(seq) - n_tok} new tokens):")
            print(tok.decode(seq))


if __name__ == "__main__":
    main()
