# RUN: %PYTHON %s --dump xegpu-wg --n-layers 1 | FileCheck %s
# CHECK: module attributes {gpu.container_module} {

"""Llama-3 forward pass on the Intel GPU (XeGPU) -- the DRIVER ("run it").

Runs a Llama-3 transformer forward as one MLIR module lowered to many separate
un-fused XeGPU kernels with on-device handoff. Llama building blocks:

  * RMSNorm
  * SwiGLU FFN     w2( silu(z @ w1) * (z @ w3) )
  * grouped-query attention via the fused flash kernel (causal by default;
    --no-causal to disable), with RoPE applied to q/k
  * no biases (Llama uses bias=False on every Linear)

Two modes, same payload + schedule underneath:

  * default (no --model): TOY self-check. Random weights, small dims
    (T=256, C=256, H=4, n_kv=2, hidden=1024, vocab=256); `--check` runs on the GPU
    and compares against a plain-numpy reference. Needs only numpy -- this is the
    CI-friendly path (see the RUN line above).
  * --model PATH: REAL inference. Loads a HuggingFace Llama-3.2 checkpoint, tokenizes
    --prompt, runs the forward, and reports the predicted next token(s). Needs
    transformers + safetensors + the checkpoint on disk (see --model).

Both reuse the same payload (llama3_payload.build_llama_payload) and schedule
(llama3_schedule.build_combined_schedule); only dims/weights/input differ.

Three-stage organization (compiling a model to the GPU here):
  1. Payload  ("what to compute") -> examples/xegpu/llama3_payload.py
  2. Schedule ("how to lower it")  -> examples/xegpu/llama3_schedule.py
  3. Driver   ("run it")           -> this file.

Run:
  .venv/bin/python examples/xegpu/llama3.py [--n-layers N] [--check]   # toy self-check
  .venv/bin/python examples/xegpu/llama3.py --dump STAGE               # dump IR and exit
  .venv/bin/python examples/xegpu/llama3.py --model <ckpt> --seq-len 256 \
      --prompt "The capital of France is"                             # real inference
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


# =============================================================================
# NUMPY REFERENCE -- the same math in plain numpy, to CHECK the GPU result.
# `_f16` rounds through float16 to model the GPU's f16 matmul precision.
# =============================================================================
def _numpy_rms(x, weight, eps=1e-5):
    ms = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
    return x / np.sqrt(ms + eps) * weight


def _numpy_f16(a):
    # round f32 -> f16 -> f32: models the precision loss of the GPU's f16 matmul.
    return a.astype(np.float16).astype(np.float32)


def _numpy_silu(x):
    return x / (1.0 + np.exp(-x))


def _numpy_rope_tables(T, hs, theta=10000.0):
    """Precompute (cos, sin) tables of shape (T, hs/2) for half-split RoPE.
    freq[i] = theta**(-2i/hs); angle[t,i] = t * freq[i]. Matches the payload's
    cos/sin memref args (see Builder.rope)."""
    half = hs // 2
    freqs = theta ** (-np.arange(0, half, dtype=np.float32) * 2.0 / hs)
    ang = np.outer(np.arange(T, dtype=np.float32), freqs)  # (T, half)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def _numpy_rope(x, cos, sin, nh):
    """Half-split (GPT-NeoX / HF-Llama) rotary embedding on (T, nh*hs) f32, per head.
    Mirrors Builder.rope: within each head, coord d (first half) pairs with d+half
    (second half): out[d]=a*cos-b*sin, out[d+half]=b*cos+a*sin. cos/sin are (T,half)."""
    T, D = x.shape
    hs = D // nh
    half = hs // 2
    v = x.reshape(T, nh, hs)
    a = v[:, :, :half]
    b = v[:, :, half:]
    c = cos[:, None, :]  # (T,1,half) broadcast over heads
    s = sin[:, None, :]
    out = np.empty_like(v)
    out[:, :, :half] = a * c - b * s
    out[:, :, half:] = b * c + a * s
    return out.reshape(T, D)


def _numpy_mha(q, k, v, H, n_kv, causal=False):
    """Grouped-query attention over q (T,C) and narrow k/v (T, n_kv*hs), per query
    head, with an optional causal mask. Returns (T,C). Query head h reads KV head
    h // n_rep (n_rep = H // n_kv), matching the fused kernel's floordiv K/V index.
    Mirrors the fused kernel's math; pass `causal` to match `fa_params["causal"]`."""
    T, C = q.shape
    hs = C // H
    n_rep = H // n_kv
    scale = 1.0 / (hs**0.5)
    mask = np.triu(np.full((T, T), -np.inf, np.float32), k=1) if causal else 0.0
    attn = np.zeros((T, C), np.float32)
    for h in range(H):
        q_sl = slice(h * hs, (h + 1) * hs)
        kv = h // n_rep  # GQA: this query head's KV head
        kv_sl = slice(kv * hs, (kv + 1) * hs)
        scores = (q[:, q_sl] @ k[:, kv_sl].T) * scale + mask
        scores = scores - scores.max(-1, keepdims=True)
        e = np.exp(scores)
        w = e / e.sum(-1, keepdims=True)
        attn[:, q_sl] = _numpy_f16(w) @ v[:, kv_sl]
    return attn


def numpy_ref_block_llama(x, w, cos, sin, H, n_kv, eps=1e-5, causal=False):
    """Grouped-query + RoPE Llama block reference (matches _emit_block_llama). wk/wv
    are narrow (C, n_kv*hs), so k/v are (T, n_kv*hs) and _mha does the head grouping.
    RoPE is applied to q and k on the f32 projection, before the f16 cast (v skips
    RoPE) -- matching Builder.fused_attention."""
    rms1 = _numpy_f16(_numpy_rms(x, w["attn_norm"], eps))
    q = _numpy_f16(_numpy_rope(rms1 @ w["wq"].astype(np.float32), cos, sin, H))
    k = _numpy_f16(_numpy_rope(rms1 @ w["wk"].astype(np.float32), cos, sin, n_kv))
    v = _numpy_f16(rms1 @ w["wv"].astype(np.float32))
    attn = _numpy_mha(q, k, v, H, n_kv, causal)
    proj = _numpy_f16(attn) @ w["wo"].astype(np.float32)
    h = x + proj
    rms2 = _numpy_f16(_numpy_rms(h, w["ffn_norm"], eps))
    gate = _numpy_silu(rms2 @ w["w1"].astype(np.float32))
    up = rms2 @ w["w3"].astype(np.float32)
    o = _numpy_f16(gate * up) @ w["w2"].astype(np.float32)
    return h + o


def numpy_ref_llama(x, layer_w, fn_w, lmw, cos, sin, H, n_kv, eps=1e-5, causal=False):
    """Grouped-query + RoPE full-Llama reference (matches build_llama_payload);
    `causal` toggles the autoregressive attention mask to match the fused kernel."""
    h = x
    for w in layer_w:
        h = numpy_ref_block_llama(h, w, cos, sin, H, n_kv, eps, causal)
    hf = _numpy_rms(h, fn_w, eps)
    return _numpy_f16(hf) @ lmw.astype(np.float32)


# =============================================================================
# SHARED LOWERING -- build the payload + schedule and lower to a GPU binary.
# =============================================================================
def _lower_payload(T, C, hidden, vocab, n_layers, H, n_kv, hs, eps, causal, dump):
    """Build the payload + combined schedule and lower it to a GPU-binary module.

    Must be called inside an ``ir.Context``. Returns ``(payload, kinds)``, or ``None``
    when a ``--dump`` stage was printed (the caller should just return).
    """
    param_selector = XeGPUParameterSelector()
    mm_params = param_selector.get_parameters_dict((T, C, C))
    mm_params["gpu_specs"] = param_selector.gpu_specs
    ln_params = {
        "wg_rows": 64,
        "sg_rows": 8,
        "subgroup_size": 16,
        "reduction_step_size": 16,
        "T": T,
    }
    fa_params = {
        "batch_size": 1,
        "num_heads": H,
        "n_ctx": T,
        "n_head": hs,
        "wg_rows": 128,
        "sg_rows": 16,
        "subgroup_size": 16,
        "inner_loop_tile_size": 64,
        "causal": causal,
    }

    mod, kinds, mm_shapes = build_llama_payload(
        "payload", T, C, hidden, vocab, n_layers, H, n_kv, eps=eps
    )
    if dump == "initial":
        print(mod)
        print("KINDS:", kinds)
        return None

    # Per-matmul DPAS params: the K/V projections are narrow (N = n_kv*hs < C), so
    # they need different wg_n/sg_n tiling than the wide matmuls. Select once per
    # distinct (M,N,K) shape (the selector reads the tuple as (M,N,K)).
    shape_params = {}
    for shp in mm_shapes:
        if shp not in shape_params:
            p = param_selector.get_parameters_dict(shp)
            p["gpu_specs"] = param_selector.gpu_specs
            shape_params[shp] = p
    mm_params_list = [dict(shape_params[shp]) for shp in mm_shapes]

    sched = build_combined_schedule(
        dict(mm_params),
        dict(ln_params),
        kinds,
        stop_at_stage=(dump or ""),
        fa_params=dict(fa_params),
        mm_params_list=mm_params_list,
    )
    if dump == "schedule":
        print(sched)
        return None
    schedules = [sched]
    if not dump or dump == "final":
        schedules.append(xegpu_to_binary())
    payload = TransformDriver(schedules).apply(mod)
    if dump:
        print(payload)
        return None
    print(f"LOWERED OK: 'llama' to {len(kinds)} kernels in one module")
    return payload, kinds


# =============================================================================
# TOY MODE -- random weights, self-check against the numpy reference (CI path).
# =============================================================================
def run_toy(args):
    """Random-weights forward, optionally checked against the numpy reference."""
    causal = not args.no_causal  # Llama-3 is autoregressive/causal
    n_layers = args.n_layers if args.n_layers is not None else 1
    eps = 1e-5

    # Kernel-friendly shapes: T=dim=256 (q/k/v/proj matmuls), hidden=1024,
    # vocab=256. Multi-head: H heads of head_size=dim/H=64. GQA: n_kv KV heads
    # (n_kv <= H, H % n_kv == 0); each query head reads KV head h // (H//n_kv).
    T, C, hidden = 256, 256, 1024
    vocab = 256
    H = 4  # attention (query) heads (hs = C/H = 64)
    n_kv = 2  # 2 query heads share each KV head (n_rep = H // n_kv = 2)
    hs = C // H
    kv_dim = n_kv * hs  # narrow K/V feature width

    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        lowered = _lower_payload(
            T, C, hidden, vocab, n_layers, H, n_kv, hs, eps, causal, args.dump
        )
        if lowered is None:
            return
        payload, kinds = lowered

        if not args.check:
            return
        runner = Runner(
            payload,
            mem_manager_cls=GPUMemoryManager,
            shared_libs=["libmlir_levelzero_runtime.so"],
        )
        np.random.seed(0)
        out = np.zeros((T, vocab), np.float32)
        cb = Runner.get_gpu_argument_access_callback(out, arg_index=0)
        sc = 0.05  # small weight scale -> O(1) activations so f16 stays accurate

        # host "embeddings": simulate tok_embeddings(tokens) as the input x.
        x = (np.random.randn(T, C) * 0.5).astype(np.float32)
        # RoPE (T, hs/2) tables, shared across layers.
        cos, sin = _numpy_rope_tables(T, hs)
        layers = []
        host = [out, x, cos, sin]  # matches payload arg order: out, x, cos, sin, ...
        for _ in range(n_layers):
            lw = dict(
                attn_norm=np.ones(C, np.float32),
                wq=(np.random.randn(C, C) * sc).astype(np.float16),
                wk=(np.random.randn(C, kv_dim) * sc).astype(np.float16),
                wv=(np.random.randn(C, kv_dim) * sc).astype(np.float16),
                wo=(np.random.randn(C, C) * sc).astype(np.float16),
                ffn_norm=np.ones(C, np.float32),
                w1=(np.random.randn(C, hidden) * sc).astype(np.float16),
                w2=(np.random.randn(hidden, C) * sc).astype(np.float16),
                w3=(np.random.randn(C, hidden) * sc).astype(np.float16),
            )
            layers.append(lw)
            host += [
                lw["attn_norm"],
                lw["wq"],
                lw["wk"],
                lw["wv"],
                lw["wo"],
                lw["ffn_norm"],
                lw["w1"],
                lw["w2"],
                lw["w3"],
            ]
        fn_w = np.ones(C, np.float32)
        lmw = (np.random.randn(C, vocab) * sc).astype(np.float16)
        host += [fn_w, lmw]
        runner.execute(
            host_input_buffers=host,
            payload_function_name="payload",
            argument_access_callback=cb,
        )
        ref = numpy_ref_llama(x, layers, fn_w, lmw, cos, sin, H, n_kv, causal=causal)

        rel = np.abs(out - ref).max() / (np.abs(ref).max() + 1e-6)
        print(f"max abs diff={np.abs(out - ref).max():.4f}  rel={rel:.6f}")
        print("PASSED" if rel < 5e-2 else "FAILED")


# =============================================================================
# REAL MODE -- load a HuggingFace checkpoint, tokenize a prompt, predict tokens.
# =============================================================================
def run_real(args):
    """Real HuggingFace weights: tokenize --prompt, run the forward, report tokens."""
    # Imported here so the toy/CI path never needs safetensors / the checkpoint.
    from llama3_weights import load_llama_weights, rope_tables_from_config

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
    causal = not args.no_causal

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(args.prompt, return_tensors="np")["input_ids"][0].astype(np.int64)
    n_tok = len(ids)
    T = args.seq_len if args.seq_len else n_tok
    if n_tok > T:
        ids = ids[:T]
        n_tok = T
    print(
        f"prompt {args.prompt!r} -> {n_tok} tokens; T={T}, C={C}, H={H}, n_kv={n_kv}, "
        f"hidden={hidden}, vocab={vocab}, n_layers={n_layers}"
    )

    # host embedding lookup: x[t] = embed_tokens[ids[t]]. Rows past the prompt (pad)
    # are zeros; with causal masking they don't affect the prompt positions' logits.
    x = np.zeros((T, C), np.float32)
    x[:n_tok] = W["embeddings"][ids]

    # RoPE tables from the real config (theta=500000 + llama3 NTK scaling).
    cos, sin = rope_tables_from_config(cfg, T)

    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        lowered = _lower_payload(
            T, C, hidden, vocab, n_layers, H, n_kv, hs, eps, causal, args.dump
        )
        if lowered is None:
            return
        payload, kinds = lowered

        runner = Runner(
            payload,
            mem_manager_cls=GPUMemoryManager,
            shared_libs=["libmlir_levelzero_runtime.so"],
        )
        out = np.zeros((T, vocab), np.float32)
        cb = Runner.get_gpu_argument_access_callback(out, arg_index=0)
        # arg order matches build_llama_payload: out, x, cos, sin, then per layer
        # [an,wq,wk,wv,wo,fn,w1,w2,w3], then final [fn_w, lmw]. Weights/cos/sin are
        # fixed across generation steps; only `x` (the embedded sequence) changes,
        # so we build `host` once and just overwrite host[1] each step.
        host = [x, cos, sin]
        for lw in W["layers"]:
            host += [
                lw["an"],
                lw["wq"],
                lw["wk"],
                lw["wv"],
                lw["wo"],
                lw["fn"],
                lw["w1"],
                lw["w2"],
                lw["w3"],
            ]
        host += [W["fn_w"], W["lmw"]]

        def forward(seq_ids, n):
            """Embed the first `n` tokens of seq_ids into x, run the GPU forward,
            return the logits row at the last real position (n-1)."""
            xb = np.zeros((T, C), np.float32)
            xb[:n] = W["embeddings"][seq_ids[:n]]
            runner.execute(
                host_input_buffers=[out, xb] + host[1:],
                payload_function_name="payload",
                argument_access_callback=cb,
            )
            return out[n - 1].copy()

        # ---- single next token: report top-5 ----
        last = forward(ids, n_tok)
        top = np.argsort(last)[::-1][:5]
        print(f"\nprompt: {args.prompt!r}")
        print("top-5 next tokens (GPU):")
        for i in top:
            print(f"  {int(i):7d}  {tok.decode([int(i)])!r:20s} logit={last[i]:.3f}")
        print(
            f"argmax next token id: {int(last.argmax())}  "
            f"-> {tok.decode([int(last.argmax())])!r}"
        )

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


def main():
    parser = argparse.ArgumentParser(
        description="Llama-3 forward pass on the Intel GPU (XeGPU).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --model selects REAL inference; omit it for the toy random-weights self-check.
    # The checkpoint is NOT shipped with the repo (gated model, ~2.4 GB); download it
    # first (e.g. `huggingface-cli download meta-llama/Llama-3.2-1B --local-dir <dir>`)
    # and point --model at that directory.
    parser.add_argument(
        "--model",
        default=None,
        help="HF checkpoint dir -> real inference. Omit for the toy self-check.",
    )
    parser.add_argument(
        "--prompt",
        default="The capital of France is",
        help="(real mode) text prompt to run the forward pass on.",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=None,
        help="Truncate to the first N transformer blocks (toy default: 1; real: all).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="(real mode) compiled sequence length T (default: padded token count).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="(toy mode) run on the GPU and compare against the numpy reference.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=0,
        help="(real mode) if >0, greedily generate this many tokens (re-run the "
        "forward per step, no KV cache). 0 = just report the next token.",
    )
    parser.add_argument(
        "--vocab-cap",
        type=int,
        default=None,
        help="(real mode) DIAGNOSTIC: shrink the output (lm_head) width to this many "
        "columns; the transformer block + embeddings stay full-width.",
    )
    parser.add_argument(
        "--no-causal",
        action="store_true",
        help="Disable causal masking (run non-causal attention).",
    )
    parser.add_argument(
        "--dump",
        type=str,
        default=None,
        choices=[
            "initial",
            "schedule",
            "tiled",
            "vectorized",
            "bufferized",
            "inner-tiled",
            "gpu-outlining",
            "xegpu-initial",
            "xegpu-wg",
            "final",
        ],
        help="Print the IR at the given stage and exit.",
    )
    args = parser.parse_args()

    if args.model is not None:
        run_real(args)
    else:
        run_toy(args)


if __name__ == "__main__":
    main()
