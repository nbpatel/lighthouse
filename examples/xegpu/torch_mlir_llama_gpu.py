# RUN: %PYTHON %s --layers 1 --inspect | FileCheck %s
# REQUIRES: torch_mlir
# CHECK: tiled OK

"""GPU driver: a multi-layer Llama-3 (n blocks + final RMSNorm + LM head) from torch-mlir.

The torch-mlir counterpart of `llama3.py`: the payload comes from `llama3_torch_model.py`
through torch-mlir instead of the hand-written `Builder`, and `llama3_torch_schedule.py`
derives the kernel plan from the IR.

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
import torch._dynamo as dynamo
import torch.nn as nn
from mlir import ir

from lighthouse import dialects as lh_dialects
from lighthouse.execution import GPUMemoryManager
from lighthouse.execution.runner import Runner
from lighthouse.ingress.torch import import_from_model
from lighthouse.pipeline.driver import TransformDriver
from lighthouse.schedule.func import convert_function_results
from lighthouse.schedule.xegpu import xegpu_to_binary

from llama3_torch_model import (
    BLOCK_WEIGHTS,
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


_PROJ_SHAPES = {  # HF `[out, in]`, as (out, in) in units of (C, hidden, kv_dim)
    "self_attn.q_proj": ("C", "C"),
    "self_attn.k_proj": ("kv", "C"),
    "self_attn.v_proj": ("kv", "C"),
    "self_attn.o_proj": ("C", "C"),
    "mlp.gate_proj": ("hidden", "C"),
    "mlp.up_proj": ("hidden", "C"),
    "mlp.down_proj": ("C", "hidden"),
}
# `BLOCK_WEIGHTS` order, by HF name.
_HF_BLOCK = {
    "attn_norm": "input_layernorm",
    "wq": "self_attn.q_proj",
    "wk": "self_attn.k_proj",
    "wv": "self_attn.v_proj",
    "wo": "self_attn.o_proj",
    "ffn_norm": "post_attention_layernorm",
    "w1": "mlp.gate_proj",
    "w2": "mlp.down_proj",
    "w3": "mlp.up_proj",
}


def _key(hf_name):
    """`nn.ParameterDict` key for an HF parameter name (keys may not contain dots)."""
    return hf_name.replace(".", "__")


class HFParamLlama(nn.Module):
    """`Llama3` over `nn.Parameter`s named like the HuggingFace checkpoint.

    Projections are stored `[in, out]` (the checkpoint transposed once in torch): HF's `[out, in]`
    would need `W.T` in the forward, i.e. a 2-D transpose kernel per projection, and those do not
    lower. The tied LM head is likewise the embedding table transposed. The embedding lookup
    itself stays on the host.
    """

    def __init__(self, cfg, n_layers):
        super().__init__()
        C, hidden = cfg.hidden_size, cfg.intermediate_size
        H, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
        size = {"C": C, "hidden": hidden, "kv": n_kv * (C // H)}
        self.n_layers = n_layers
        self.llama = Llama3(
            C, hidden, cfg.vocab_size, H, n_kv, n_layers, cfg.rms_norm_eps
        )
        self.params = nn.ParameterDict()
        for i in range(n_layers):
            for name in ("input_layernorm", "post_attention_layernorm"):
                self.params[_key(f"model.layers.{i}.{name}.weight")] = nn.Parameter(
                    torch.ones(C)
                )
            for name, (out_f, in_f) in _PROJ_SHAPES.items():
                self.params[_key(f"model.layers.{i}.{name}.weight")] = nn.Parameter(
                    torch.empty(size[in_f], size[out_f])
                )
        self.params[_key("model.norm.weight")] = nn.Parameter(torch.ones(C))
        self.params[_key("lm_head.weight")] = nn.Parameter(
            torch.empty(C, cfg.vocab_size)
        )

    def weight_pack(self):
        """The weights in `Llama3.forward` order: n_layers * BLOCK_WEIGHTS, final norm, LM head."""
        p = self.params
        pack = [
            p[_key(f"model.layers.{i}.{_HF_BLOCK[w]}.weight")]
            for i in range(self.n_layers)
            for w in BLOCK_WEIGHTS
        ]
        return (*pack, p[_key("model.norm.weight")], p[_key("lm_head.weight")])

    def forward(self, x, cos_q, sin_q, cos_k, sin_k):
        return self.llama(x, cos_q, sin_q, cos_k, sin_k, *self.weight_pack())


def capture_graph(model, inputs):
    """The (GraphModule, example_inputs) `torch.compile` hands a backend -- as in kernel_bench.

    Dynamo lifts the parameters into graph placeholders, so torch-mlir imports them as function
    ARGUMENTS; `import_from_model(nn.Module)` would instead bake them as `dense_resource`
    constants, which the matmul schedule cannot prefetch from.
    """
    got = {}

    def backend(gm, example_inputs):
        got["gm"], got["ex"] = gm, example_inputs
        raise RuntimeError("graph captured")

    dynamo.reset()
    try:
        torch.compile(model, backend=backend, dynamic=False, fullgraph=True)(*inputs)
    except Exception:
        if "gm" not in got:
            raise
    return got["gm"], list(got["ex"])


def real_model(args):
    """Real Llama-3.2 weights, loaded the kernel_bench way: into torch parameters.

    Returns (model, activations, dims, n_layers, eps, tokenizer, ids, n_tok). `activations` are
    the forward inputs (x, cos_q, sin_q, cos_k, sin_k); the weights travel as the model's
    parameters.
    """
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    cfg = LlamaConfig.from_pretrained(args.model)
    n_layers = min(args.layers, cfg.num_hidden_layers)
    C, H, n_kv = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads
    dims = (C, cfg.intermediate_size, cfg.vocab_size, H, n_kv)

    sd = load_file(f"{args.model}/model.safetensors")
    embed = sd.pop("model.embed_tokens.weight")
    sd = {
        k: v.T.contiguous() if k.endswith("_proj.weight") else v
        for k, v in sd.items()
        if not k.startswith("model.layers.") or int(k.split(".")[2]) < n_layers
    }
    sd["lm_head.weight"] = embed.T.contiguous()  # tie_word_embeddings
    model = HFParamLlama(cfg, n_layers)
    model.params.load_state_dict({_key(k): v for k, v in sd.items()}, strict=True)
    model = model.half().eval().requires_grad_(False)
    del sd

    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(args.prompt, return_tensors="pt")["input_ids"][0]
    T = args.seq or len(ids)
    ids = ids[:T]
    n_tok = len(ids)
    # Host embedding lookup. Rows past the prompt stay zero; the causal mask keeps them out of
    # the prompt positions' logits.
    x = torch.zeros(T, C, dtype=torch.float16)
    x[:n_tok] = embed[ids].half()
    # transformers' own RoPE (rope_theta + llama3 frequency scaling); `cos` repeats its halves.
    cos, sin = LlamaRotaryEmbedding(cfg)(x.float(), torch.arange(T)[None])
    half = C // H // 2
    cos, sin = cos[0, :, :half].half(), sin[0, :, :half].half()
    act = (x, *expand_rope_tables(cos, sin, H), *expand_rope_tables(cos, sin, n_kv))
    print(
        f"  checkpoint {args.model}: C={C} hidden={dims[1]} H={H} n_kv={n_kv} "
        f"vocab={dims[2]} layers={n_layers} eps={cfg.rms_norm_eps}"
    )
    print(f"  prompt {args.prompt!r} -> {n_tok} tokens, T={T}")
    return model, act, dims, n_layers, cfg.rms_norm_eps, tok, ids, n_tok


def hf_reference_logits(args, n_layers, ids):
    """transformers' own `LlamaForCausalLM` (f32, CPU) on the prompt: an independent check."""
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig.from_pretrained(args.model, num_hidden_layers=n_layers)
    hf = LlamaForCausalLM.from_pretrained(
        args.model, config=cfg, torch_dtype=torch.float32
    ).eval()
    with torch.no_grad():
        return hf(ids[None]).logits[0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument(
        "--model",
        default="",
        help="HuggingFace Llama-3.2 checkpoint dir: real weights and a real prompt, loaded "
        "into torch parameters that torch.compile lifts to MLIR arguments (as kernel_bench)",
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
    tok = ids = n_tok = hf_model = None
    eps = 1e-5
    if args.model:
        hf_model, act, dims, L, eps, tok, ids, n_tok = real_model(args)
        C, hidden, vocab, H, n_kv = dims
        T = act[0].shape[0]
        # The same tensors in `Llama3.forward` order, for the PyTorch references below.
        fwd = (*act, *hf_model.weight_pack())
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
        if hf_model is not None:
            gm, graph_args = capture_graph(hf_model, act)
            mod = import_from_model(gm, graph_args, ir_context=ir.Context.current)
        else:
            graph_args = list(fwd)
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
        host_ins = [out, *(a.detach().contiguous().numpy() for a in graph_args)]
        GPUMemoryManager.emit_memory_management_funcs(mod, host_inputs=host_ins)
        Runner.make_function_callable(mod, "main")

        sched = generic_schedule(
            "main",
            "B70",
            plan,
            kernel_params,
            inspect=args.inspect,
            stop_after_bufferize=args.transpose_views
            or args.fuse_halves
            or args.fold_gqa,
        )
        if args.inspect:
            TransformDriver(schedules=[sched]).apply(mod)
            print("tiled OK")
            return 0

        if args.transpose_views or args.fuse_halves or args.fold_gqa:
            # Three stages: schedule up to bufferization, apply
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
            hf_ref = hf_reference_logits(args, L, ids)
            hf_id = int(hf_ref[n_tok - 1].argmax())
            hf_rel = np.abs(got[rows] - hf_ref).max() / (np.abs(hf_ref).max() + 1e-6)
            print(
                f"    transformers LlamaForCausalLM (f32) next token: {tok.decode([hf_id])!r}"
                f"   {'MATCH' if hf_id == int(got[n_tok - 1].argmax()) else 'MISMATCH'}"
                f"; logits rel={hf_rel:.6f}"
            )

        # Zero-attention guard. With N layers the
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
