"""GPU driver: a MULTI-LAYER Llama-3 (n blocks + final RMSNorm + LM head) from torch-mlir.

This is A10: stacking. `torch_mlir_block_gpu.py` proved one decoder block; this one proves the
whole model, which is what the hand-written-payload path already does
(`examples/llama/test_llama3_gpu.py`, and `llama3_actual_data` at real Llama-3.2-1B weights).

Nothing about the schedule is layer-aware -- `classify_payload` just walks the IR, so N layers
is N times the plan. What stacking actually tests is whether anything in the flow is
accidentally global: handle bookkeeping that assumes one attention region or one reduction, the
`match_and_split(..., nhandles=len(plan))` counts, and compile time (every kernel is its own
`gpu.module`, each compiled by ocloc).

Run:  python examples/xegpu/torch_mlir_llama_gpu.py [--layers N] [--inspect] [--no-causal]
                                    [--transpose-views]
                                    [--seq N --width N --hidden N --heads N --kv N --vocab N]
"""

import argparse
import types

import numpy as np
import torch
from mlir import ir

from lighthouse import dialects as lh_dialects
from lighthouse.execution import GPUMemoryManager
from lighthouse.execution.runner import Runner
from lighthouse.ingress.torch import import_from_model
from lighthouse.pipeline.driver import TransformDriver
from lighthouse.schedule.func import convert_function_results
from lighthouse.schedule.xegpu import xegpu_to_binary

from llama3_torch_model import (
    Llama3,
    expand_rope_tables,
    make_weights,
)
from llama3_torch_schedule import (
    EW_PARAMS,
    classify_payload,
    eliminate_empty_tensors_schedule,
    fold_gqa_broadcasts,
    fuse_rope_halves,
    generic_schedule,
    generic_schedule_tail,
    params_for_plan,
    redirect_staged_destination_copies,
    replace_transpose_kernels_with_views,
)


def masked_attention(self, q, k, v):
    """`LlamaBlock._attention` with the causal mask spliced in, for the f32 reference.

    The payload is non-causal on its own; the schedule masks inside the flash loop. Shared by
    every layer of the reference model.
    """
    t = q.shape[0]
    hs, H, n_kv = self.hs, self.H, self.n_kv
    qh = q.view(t, H, hs).transpose(0, 1)
    kh = k.view(t, n_kv, hs).transpose(0, 1).repeat_interleave(H // n_kv, dim=0)
    vh = v.view(t, n_kv, hs).transpose(0, 1).repeat_interleave(H // n_kv, dim=0)
    scores = torch.matmul(qh, kh.transpose(1, 2)) * (1.0 / hs**0.5)
    keep = torch.tril(torch.ones(t, t, dtype=torch.bool))
    w = torch.softmax(scores.masked_fill(~keep, float("-inf")), dim=-1)
    return torch.matmul(w, vh).transpose(0, 1).reshape(t, H * hs)


def reference(
    dims, n_layers, eps, args_tuple, causal, zero_attention=False, dtype=torch.float32
):
    """PyTorch reference. `zero_attention` gives the no-attention control the guard needs.

    `dtype` picks WHICH question is being asked, and with real weights the two answers differ
    by two orders of magnitude:
      float32 -- "is the model right?", i.e. the true value the payload approximates.
      float16 -- "did the LOWERING reproduce the payload?", since every kernel in this flow
                 keeps f16 intermediates. This is the right oracle for the schedule, and the
                 gap between the two is the cost of the payload's dtype choice, not a bug.
    """
    C, hidden, vocab, H, n_kv = dims
    model = Llama3(C, hidden, vocab, H, n_kv, n_layers, eps).eval()
    for blk in model.blocks:
        if zero_attention:
            blk._attention = types.MethodType(
                lambda self, q, k, v: torch.zeros_like(q), blk
            )
        elif causal:
            blk._attention = types.MethodType(masked_attention, blk)
    with torch.no_grad():
        return model(*(a.to(dtype) for a in args_tuple)).float().numpy()


def real_inputs(args):
    """Load a real HuggingFace Llama-3.2 checkpoint into this payload's argument order.

    Reuses `llama3_weights.py` -- the loader the hand-payload driver (`llama3.py --model`)
    uses -- so both paths consume the SAME checkpoint tensors in the same `[in, out]` layout.
    The embedding lookup and the RoPE tables are host-side in both, for the same reason: a
    gather is not something this schedule lowers, and the tables are data, not compute.

    Returns (dims, n_layers, eps, fwd, tokenizer, ids, n_tok).
    """
    from llama3_weights import load_llama_weights, rope_tables_from_config

    W = load_llama_weights(args.model, n_layers=args.layers)
    cfg = W["cfg"]
    C, H = cfg["hidden_size"], cfg["num_attention_heads"]
    n_kv, hidden = cfg["num_key_value_heads"], cfg["intermediate_size"]
    vocab, eps = cfg["vocab_size"], cfg["rms_norm_eps"]
    n_layers = len(W["layers"])

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(args.prompt, return_tensors="np")["input_ids"][0].astype(np.int64)
    n_tok = len(ids)
    T = args.seq or n_tok
    ids, n_tok = ids[:T], min(n_tok, T)

    # Host embedding lookup: x[t] = embed_tokens[ids[t]]. Rows past the prompt stay zero;
    # with causal masking they cannot affect the prompt positions' logits.
    x = np.zeros((T, C), np.float32)
    x[:n_tok] = W["embeddings"][ids]

    cos, sin = rope_tables_from_config(cfg, T)
    cos_t, sin_t = torch.from_numpy(cos).half(), torch.from_numpy(sin).half()

    # Norm gains come back f32 (the hand payload norms in f32); this payload is f16
    # throughout, so cast. They are gains near 1, so f16 is ample.
    def t16(a):
        return torch.from_numpy(np.ascontiguousarray(a)).half()

    weights = []
    for layer in W["layers"]:
        weights += [
            t16(layer["an"]),
            t16(layer["wq"]),
            t16(layer["wk"]),
            t16(layer["wv"]),
            t16(layer["wo"]),
            t16(layer["fn"]),
            t16(layer["w1"]),
            t16(layer["w2"]),
            t16(layer["w3"]),
        ]
    weights += [t16(W["fn_w"]), t16(W["lmw"])]

    fwd = (
        torch.from_numpy(x).half(),
        *expand_rope_tables(cos_t, sin_t, H),
        *expand_rope_tables(cos_t, sin_t, n_kv),
        *weights,
    )
    print(
        f"  checkpoint {args.model}: C={C} hidden={hidden} H={H} n_kv={n_kv} "
        f"vocab={vocab} layers={n_layers} eps={eps}"
    )
    print(f"  prompt {args.prompt!r} -> {n_tok} tokens, T={T}")
    return (C, hidden, vocab, H, n_kv), n_layers, eps, fwd, tok, ids, n_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument(
        "--model",
        default="",
        help="HuggingFace Llama-3.2 checkpoint dir: real weights and a real prompt "
        "(same loader as the hand-payload driver llama3.py --model)",
    )
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--kv", type=int, default=2, help="KV heads (GQA)")
    ap.add_argument("--vocab", type=int, default=128)
    ap.add_argument(
        "--no-causal",
        dest="causal",
        action="store_false",
        help="compare against the unmasked payload, to separate lowering from masking",
    )
    ap.add_argument(
        "--transpose-views",
        action="store_true",
        help="delete the head-major transpose KERNELS and present their results as strided "
        "memref views, the way the hand payload and Inductor both do it (no data movement). "
        "3 kernels per layer.",
    )
    ap.add_argument(
        "--fuse-halves",
        action="store_true",
        help="merge each RoPE's two half-chains into ONE 2-result linalg.generic, the form the "
        "hand payload uses (2 kernels per RoPE -> 1). 2 kernels per layer.",
    )
    ap.add_argument(
        "--fold-gqa",
        action="store_true",
        help="delete the GQA K/V broadcast kernels: attention indexes the head-major K/V "
        "views directly with the repeat dim dropped (the hand payload's zero-copy GQA). "
        "2 kernels per layer.",
    )
    ap.add_argument("--inspect", action="store_true", help="print plan, no GPU run")
    args = ap.parse_args()
    tok = ids = n_tok = None
    eps = 1e-5
    if args.model:
        dims, L, eps, fwd, tok, ids, n_tok = real_inputs(args)
        C, hidden, vocab, H, n_kv = dims
        T = fwd[0].shape[0]
    else:
        T, C, hidden = args.seq, args.width, args.hidden
        H, n_kv, vocab, L = args.heads, args.kv, args.vocab, args.layers
        half = C // H // 2
        dims = (C, hidden, vocab, H, n_kv)
        x = torch.randn(T, C).half()
        cos, sin = torch.randn(T, half).half(), torch.randn(T, half).half()
        weights = tuple(
            w.half() for w in make_weights(C, hidden, vocab, H, n_kv, n_layers=L)
        )
        fwd = (
            x,
            *expand_rope_tables(cos, sin, H),
            *expand_rope_tables(cos, sin, n_kv),
            *weights,
        )

    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        model = Llama3(C, hidden, vocab, H, n_kv, L, eps).eval()

        mod = import_from_model(model, fwd, ir_context=ir.Context.current)
        mod = TransformDriver(schedules=[convert_function_results("main")]).apply(mod)

        if args.fuse_halves:
            # Elimination first, so each RoPE's destination is already rooted at its real
            # buffer when the merged op takes its two destination slices.
            mod = TransformDriver(
                schedules=[eliminate_empty_tensors_schedule("main")]
            ).apply(mod)
            print(f"  RoPE half-pairs fused: {fuse_rope_halves(mod, 'main')}")

        plan = classify_payload(mod, "main")
        kernel_params = params_for_plan(plan, "B70", EW_PARAMS)
        for p in kernel_params:
            if "causal" in p:
                p["causal"] = args.causal
        kinds: dict[str, int] = {}
        for entry in plan:
            kinds[entry["kind"]] = kinds.get(entry["kind"], 0) + 1
        print(f"  {L} layer(s) -> {len(plan)} kernel(s): {kinds}")
        print(f"  per layer: {(len(plan) - 2) / L:.1f}   causal={args.causal}")
        if args.inspect:
            for i, entry in enumerate(plan):
                print(
                    f"  {i:3} {entry['kind']:<12} "
                    f"{str(tuple(entry['shape'] or ())):<20} n={len(entry['members'])}"
                )

        out = np.zeros((T, vocab), np.float16)
        host_ins = [out, *(a.numpy() for a in fwd)]
        GPUMemoryManager.emit_memory_management_funcs(mod, host_inputs=host_ins)
        Runner.make_function_callable(mod, "main")

        sched = generic_schedule(
            "main",
            "B70",
            plan,
            kernel_params,
            inspect=args.inspect,
            stop_after_bufferize=args.transpose_views or args.fuse_halves or args.fold_gqa,
        )
        if args.inspect:
            TransformDriver(schedules=[sched]).apply(mod)
            print("tiled OK")
            return 0

        if args.transpose_views or args.fuse_halves or args.fold_gqa:
            # Three stages, as in torch_mlir_block_gpu.py: schedule up to bufferization, apply
            # the rewrites the transform dialect has no ops for (copy -> view; staged
            # destination -> the destination itself; broadcast -> indexed read), then the
            # tail over what is left. The GQA fold reads K/V through the views, so it is last.
            mod = TransformDriver(schedules=[sched]).apply(mod)
            if args.transpose_views:
                plan, kernel_params, n = replace_transpose_kernels_with_views(
                    mod, "main", plan, kernel_params
                )
                print(
                    f"  transpose kernels replaced by views: {n} "
                    f"-> {len(plan)} kernel(s), {(len(plan) - 2) / L:.1f} per layer"
                )
            if args.fuse_halves:
                n = redirect_staged_destination_copies(mod, "main")
                print(f"  staged destination buffers removed: {n}")
            if args.fold_gqa:
                plan, kernel_params, n = fold_gqa_broadcasts(
                    mod, "main", plan, kernel_params
                )
                print(
                    f"  GQA broadcast kernels folded: {n} "
                    f"-> {len(plan)} kernel(s), {(len(plan) - 2) / L:.1f} per layer"
                )
            tail = generic_schedule_tail("main", "B70", plan, kernel_params)
            lowered = TransformDriver(schedules=[tail, xegpu_to_binary()]).apply(mod)
        else:
            lowered = TransformDriver(schedules=[sched, xegpu_to_binary()]).apply(mod)
        print("lowered to GPU binary:", "gpu.binary" in str(lowered))

        runner = Runner(
            lowered,
            mem_manager_cls=GPUMemoryManager,
            shared_libs=["libmlir_levelzero_runtime.so"],
        )
        runner.execute(
            payload_function_name="main",
            host_input_buffers=host_ins,
            argument_access_callback=Runner.get_gpu_argument_access_callback(
                out, arg_index=0
            ),
        )

        got = out.astype(np.float32)
        if tok is not None:
            # Greedy next token from the LAST PROMPT ROW -- the same check the hand-payload
            # driver reports, so the two paths can be compared token-for-token.
            last = got[n_tok - 1]
            print(f"\nprompt: {args.prompt!r}")
            print(
                f"=== next token: id {int(last.argmax())} -> "
                f"{tok.decode([int(last.argmax())])!r} ==="
            )
            top = np.argsort(last)[::-1][:5]
            print(
                "    top-5: "
                + ", ".join(f"{tok.decode([int(i)])!r}({last[i]:.2f})" for i in top)
            )

        # With REAL weights the payload's f16 intermediates cost real accuracy (measured:
        # 0.136 at 2 layers), so score the LOWERING against an f16 reference and report the
        # f32 gap separately as the dtype cost. With the synthetic weights the two coincide.
        ref = reference(dims, L, eps, fwd, args.causal, dtype=torch.float16)
        ref32 = reference(dims, L, eps, fwd, args.causal)
        # In real mode only the PROMPT rows are meaningful. `T` must stay a multiple of the
        # attention work-group tile, so a short prompt is padded with zero rows -- and a zero
        # row's RMSNorm is degenerate (`ssq = 0`, so the result is `0 * eps**-0.5`, with eps
        # subnormal in f16), which makes those rows disagree wildly and dominate a max-norm
        # taken over the whole tensor. They cannot affect the prompt's logits: they sit after
        # the prompt, and the mask is causal.
        rows = slice(0, n_tok) if tok is not None else slice(None)
        scale = np.abs(ref[rows]).max() + 1e-6
        rel = np.abs(got[rows] - ref[rows]).max() / scale
        rel32 = np.abs(got[rows] - ref32[rows]).max() / (
            np.abs(ref32[rows]).max() + 1e-6
        )
        print(f"=== LLAMA-3 {L}-LAYER GPU result: rel={rel:.6f} vs the f16 payload ===")
        print(f"    vs an f32 reference: {rel32:.6f}  (= the payload's f16 dtype cost)")
        if tok is not None:
            ref_id = int(ref32[n_tok - 1].argmax())
            print(
                f"    f32 PyTorch reference next token: id {ref_id} -> "
                f"{tok.decode([ref_id])!r}"
                f"   {'MATCH' if ref_id == int(got[n_tok - 1].argmax()) else 'MISMATCH'}"
            )

        # Same guard as torch_mlir_block_gpu.py, and it matters more here: with N layers the
        # residual stream grows while each layer's attention contribution does not, so a
        # dropped attention term is an even smaller share of the output.
        zero_ref = reference(
            dims, L, eps, fwd, args.causal, zero_attention=True, dtype=torch.float16
        )
        share = np.abs(ref[rows] - zero_ref[rows]).max() / scale
        print(f"    attention's share of the output: {share:.6f}")
        if rel > share / 4:
            print(
                f"GPU FAILED: the error ({rel:.6f}) is not small compared with attention's "
                f"own contribution ({share:.6f}) -- attention is not reaching the output"
            )
        else:
            print("GPU PASSED" if rel < 5e-2 else "GPU FAILED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
