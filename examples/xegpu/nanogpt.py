# RUN: %PYTHON %s --stage=layernorm --dump-kernel=xegpu-wg | FileCheck %s
# CHECK: module attributes {gpu.container_module} {

"""
nano-GPT (Karpathy ng-video-lecture gpt.py) inference on XeGPU.

This file is the orchestrator for bringing the gpt.py model up on the Intel GPU
one op at a time, reusing the per-op XeGPU schedules that already exist in
lighthouse (layer_norm, softmax, matmul). Each stage is validated against a
dependency-free numpy reference before the next stage is composed on top.

Build order (each `--stage` is validated against numpy before the next):
    1. layernorm   <-- THIS STAGE: prove the LayerNorm reuse path end-to-end
    2. attention   (matmul -> scaled -> causal mask -> softmax -> matmul)
    3. ffn         (matmul+bias -> relu -> matmul+bias)
    4. block       (ln + attn + residual + ln + ffn + residual)
    5. gpt         (embed[host] + N blocks + ln_f + lm_head -> logits)

Run a stage end-to-end on the GPU (requires the local SPIRV-enabled MLIR
bindings -- see the gpu-end-to-end-setup memory):
    export PYTHONPATH=/home/jovyan/llvm-project/upstream/tools/mlir/python_packages/mlir_core
    .venv/bin/python examples/xegpu/nanogpt.py --stage=layernorm --check-result -v

Dump IR at a lowering stage (works with the pip wheel, no GPU needed):
    .venv/bin/python examples/xegpu/nanogpt.py --stage=layernorm --dump-kernel=xegpu-wg
"""

import argparse
import sys
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import numpy as np
from mlir import ir

from lighthouse import dialects as lh_dialects
from lighthouse.execution.runner import Runner
from lighthouse.pipeline.driver import TransformDriver
from lighthouse.execution import GPUMemoryManager
from lighthouse.utils.numpy import mlir_to_numpy_dtype
from lighthouse.ingress.mlir_gen import get_mlir_elem_type
from lighthouse.ingress.mlir_gen.gpu_layer_norm_payload import (
    generate_gpu_layer_norm_payload,
)
from lighthouse.ingress.mlir_gen.gpu_softmax_payload import (
    generate_gpu_softmax_payload,
)
from lighthouse.ingress.mlir_gen import generate_gpu_matmul_payload
from lighthouse.schedule.xegpu import (
    layer_norm_schedule,
    softmax_schedule,
    mlp_schedule,
    xegpu_to_binary,
    xegpu_parameter_selector,
)


# ---------------------------------------------------------------------------
# ocloc/unwinder deadlock fix: stop the loadDialectModule exception storm.
# ---------------------------------------------------------------------------
_STUB_FINDER_INSTALLED = False


def install_dialect_module_stubs():
    """Stop the `loadDialectModule` C++ exception storm that causes the in-process
    "block hang" (and, with partial stubs, a SIGABRT).

    Root cause: MLIR's C++ PyGlobals::loadDialectModule probes
    `mlir.dialects.<ns>` on every createOpView. A miss raises a C++
    ModuleNotFoundError that is caught internally but NEVER cached
    (loadedDialectModules records only successes). So every op-view over an op
    whose dialect has no Python module (xegpu, xevm, and lighthouse's custom
    transform_ext / smt / tune transform ops) re-throws. That benign-but-repeated
    throw races with libocloc's backtrace()-in-signal-handler and libgcc's
    non-reentrant unwind lock -> deadlock; with only some namespaces stubbed the
    leftover throw aborts instead. The set of missing namespaces is open-ended,
    so rather than enumerate it we install a sys.meta_path finder that satisfies
    ANY otherwise-missing `mlir.dialects.<ns>` with an empty module. The probe
    then always SUCCEEDS and gets cached, so the throw never fires. The stub is
    empty, so it never shadows real op-view classes (a real module, when present,
    is found first by the normal finders and wins). No LLVM rebuild required.
    """
    global _STUB_FINDER_INSTALLED
    if _STUB_FINDER_INSTALLED:
        return

    import sys
    import types
    import importlib.abc
    import importlib.machinery

    prefix = "mlir.dialects."

    class _DialectStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, fullname, path, target=None):
            if not fullname.startswith(prefix):
                return None
            if fullname.count(".") != 2:  # only mlir.dialects.<ns>, not deeper
                return None
            # This finder is appended at the END of sys.meta_path, so it is only
            # reached after every real finder has already missed. Hence a genuine
            # stub case -- a real module would have been found first and won.
            return importlib.machinery.ModuleSpec(fullname, self)

        def create_module(self, spec):
            mod = types.ModuleType(spec.name)
            mod.__doc__ = (
                "Empty stub so PyGlobals::loadDialectModule succeeds and caches, "
                "avoiding the repeated ModuleNotFoundError throw that races with "
                "libocloc's signal handler (in-process deadlock fix)."
            )
            return mod

        def exec_module(self, module):
            pass

    # Install at the END of meta_path so real modules are always preferred.
    sys.meta_path.append(_DialectStubFinder())
    _STUB_FINDER_INSTALLED = True


# ---------------------------------------------------------------------------
# Model configuration -- locked to gpt.py (ng-video-lecture).
# ---------------------------------------------------------------------------
@dataclass
class GPTConfig:
    """Hyperparameters mirroring gpt.py.

    vocab_size is dynamic in gpt.py (len(set(input.txt))); tinyshakespeare
    is 65. It only matters for the embedding/lm_head stages, not LayerNorm.
    """

    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    block_size: int = 256
    vocab_size: int = 65
    # nn.LayerNorm default eps; gpt.py uses nn.LayerNorm(n_embd) with defaults.
    ln_eps: float = 1e-5

    @property
    def head_size(self) -> int:
        assert self.n_embd % self.n_head == 0
        return self.n_embd // self.n_head


# ---------------------------------------------------------------------------
# Numpy reference -- the ground truth every GPU stage is checked against.
# Kept dependency-free (no torch in this venv).
# ---------------------------------------------------------------------------
def layernorm_ref(
    x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float
) -> np.ndarray:
    """nn.LayerNorm over the last dim. x: (M, N), gamma/beta: (N,)."""
    x = x.astype(np.float32)
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    return (x - mean) * inv_std * gamma.astype(np.float32) + beta.astype(np.float32)


def softmax_ref(x: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax over the last dim. x: (M, N)."""
    x = x.astype(np.float32)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def attention_ref(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, head_size: int
) -> np.ndarray:
    """Single-head causal self-attention reference (gpt.py Head.forward).

    q, k, v: (T, head_size). Returns (T, head_size).
        wei = q @ k^T * head_size^-0.5
        wei = masked_fill(future, -inf); wei = softmax(wei)
        out = wei @ v
    """
    q, k, v = (a.astype(np.float32) for a in (q, k, v))
    scores = (q @ k.T) * (head_size ** -0.5)
    weights = softmax_ref(causal_mask_scores(scores))
    return weights @ v


def ffn_ref(
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    """gpt.py FeedFoward reference: Linear(n,4n) -> ReLU -> Linear(4n,n).

    nn.Linear stores weight as (out, in) and computes x @ W^T + b.
    x: (T, n_embd); w1: (4n, n); b1: (4n,); w2: (n, 4n); b2: (n,).
    """
    x = x.astype(np.float32)
    h = x @ w1.astype(np.float32).T + b1.astype(np.float32)
    h = np.maximum(h, 0.0)
    return h @ w2.astype(np.float32).T + b2.astype(np.float32)


def block_ref(x, params, n_head, head_size, eps):
    """gpt.py Block reference (single- or multi-head):
        x = x + sa(ln1(x));  x = x + ffwd(ln2(x))
    `params` is the dict produced by XeGPUBlockStage._weights()."""
    x = x.astype(np.float32)

    # --- self-attention sub-block ---
    xn1 = layernorm_ref(x, params["g1"], params["b1"], eps)
    # QKV projections (nn.Linear, no bias): q = xn1 @ Wq^T
    q = xn1 @ params["wq"].astype(np.float32).T  # (T, n_head*head_size)
    k = xn1 @ params["wk"].astype(np.float32).T
    v = xn1 @ params["wv"].astype(np.float32).T
    head_outs = []
    for h in range(n_head):
        sl = slice(h * head_size, (h + 1) * head_size)
        head_outs.append(attention_ref(q[:, sl], k[:, sl], v[:, sl], head_size))
    concat = np.concatenate(head_outs, axis=-1)  # (T, n_head*head_size)
    attn = concat @ params["wproj"].astype(np.float32).T + params["bproj"].astype(
        np.float32
    )
    x = x + attn  # residual

    # --- feed-forward sub-block ---
    xn2 = layernorm_ref(x, params["g2"], params["b2"], eps)
    ff = ffn_ref(xn2, params["w1"], params["bf1"], params["w2"], params["bf2"])
    x = x + ff  # residual
    return x


def gpt_ref(tok_ids, wte, wpe, blocks, lnf_g, lnf_b, lm_w, lm_b, n_head, head_size, eps):
    """Full gpt.py forward reference (logits), numpy/f32.

    tok_ids: (T,) int token ids. wte:(vocab,n_embd) wpe:(block,n_embd).
    blocks: list of per-block weight dicts (block_ref format). lm_w:(vocab,n_embd).
    """
    T = len(tok_ids)
    x = wte[tok_ids] + wpe[np.arange(T)]            # (T, n_embd) token+pos embed
    for p in blocks:
        x = block_ref(x, p, n_head, head_size, eps)
    x = layernorm_ref(x, lnf_g, lnf_b, eps)         # final LN
    return x @ lm_w.astype(np.float32).T + lm_b.astype(np.float32)  # (T, vocab)


def causal_mask_scores(scores: np.ndarray) -> np.ndarray:
    """Apply gpt.py's causal mask to attention scores in-place style.

    gpt.py: wei = wei.masked_fill(tril[:T,:T] == 0, -inf) before softmax.
    scores is (..., T, T); position (i, j) is masked (set to -inf) when j > i
    (a query at time i may not attend to a future key at time j).
    """
    T = scores.shape[-1]
    assert scores.shape[-2] == T, "scores must be square (T, T) in last two dims"
    i = np.arange(T)[:, None]
    j = np.arange(T)[None, :]
    future = j > i  # upper triangle, excluding diagonal
    out = scores.astype(np.float32).copy()
    out[..., future] = -np.inf
    return out


# ---------------------------------------------------------------------------
# Stage 1: LayerNorm on XeGPU.
#
# Reuses the existing gpu_layer_norm_payload + layer_norm_schedule verbatim.
# The token/sequence dims (B, T) of gpt.py collapse to rows M = B*T, with the
# normalized dimension N = n_embd. That is exactly what nn.LayerNorm(n_embd)
# computes per (batch, time) position, so the existing 2D kernel applies
# directly.
# ---------------------------------------------------------------------------
class XeGPULayerNormStage:
    """A single LayerNorm(n_embd) over M = B*T rows."""

    def __init__(self, cfg: GPTConfig, M: int, dtype: str = "f32"):
        self.cfg = cfg
        self.M = M
        self.N = cfg.n_embd
        self.eps = cfg.ln_eps
        self.shape = (self.M, self.N)
        self.bias_shape = (self.N,)
        assert dtype == "f32", "Only f32 is supported"
        self.elem_type = get_mlir_elem_type(dtype)
        self.dtype = mlir_to_numpy_dtype(self.elem_type)
        self.memory_manager_class = GPUMemoryManager
        self.payload_function_name = "payload"

    @cached_property
    def _initial_host_arrays(self) -> tuple[np.ndarray, ...]:
        """Host inputs in the (output, input, gamma, beta) order the payload
        expects. Seeded for reproducibility."""
        rng = np.random.default_rng(1337)
        input_arr = rng.uniform(-1.0, 1.0, self.shape).astype(self.dtype)
        gamma_arr = rng.uniform(0.5, 1.5, self.bias_shape).astype(self.dtype)
        beta_arr = rng.uniform(-0.1, 0.1, self.bias_shape).astype(self.dtype)
        output_arr = np.zeros(self.shape, dtype=self.dtype)
        return (output_arr, input_arr, gamma_arr, beta_arr)

    def payload_module(self) -> ir.Module:
        return generate_gpu_layer_norm_payload(
            func_name=self.payload_function_name,
            M=self.M,
            N=self.N,
            dtype=self.elem_type,
            eps=self.eps,
        )

    def schedule_modules(
        self, stop_at_stage: Optional[str] = None, parameters: Optional[dict] = None
    ) -> list[ir.Module]:
        schedules = [Runner.get_bench_wrapper_schedule(self.payload_function_name)]
        schedules.append(
            layer_norm_schedule(stop_at_stage=stop_at_stage, parameters=parameters)
        )
        if stop_at_stage and stop_at_stage != "final":
            return schedules
        schedules.append(xegpu_to_binary())
        return schedules

    def shared_libs(self) -> list[str]:
        return ["libmlir_levelzero_runtime.so"]

    def check(self, result: np.ndarray, verbose: int = 0) -> bool:
        _, input_arr, gamma_arr, beta_arr = self._initial_host_arrays
        ref = layernorm_ref(input_arr, gamma_arr, beta_arr, self.eps)
        ok = np.allclose(result, ref, rtol=1e-3, atol=1e-4)
        if verbose:
            if ok:
                print("PASSED")
            else:
                print(f"FAILED! Max abs diff: {np.abs(result - ref).max():.6e}")
        return ok


# ---------------------------------------------------------------------------
# Stage 2a: causal-masked softmax on XeGPU.
#
# In gpt.py attention ([gpt.py:83-85]), the (T, T) score matrix per (batch, head)
# is masked (future positions -> -inf) and then softmax'd along the last dim.
#
# Design decision -- the causal mask is applied HOST-SIDE, not as a GPU op:
#   * gpt.py itself masks BEFORE softmax (masked_fill then F.softmax), so feeding
#     pre-masked scores to the existing softmax kernel is faithful.
#   * The softmax_schedule does match_and_split(..., nhandles=6), expecting
#     EXACTLY the 6 ops of a softmax decomposition. Inserting a mask op into the
#     payload would break that match. Host masking needs zero schedule surgery.
#   * It also exercises that the GPU softmax handles -inf correctly (exp(-inf)=0),
#     which is exactly what the masked rows require.
#
# Rows M = B * n_head * T (one row per query position, per head, per batch);
# columns N = T (keys). The existing 2D softmax kernel applies directly.
# ---------------------------------------------------------------------------
class XeGPUMaskedSoftmaxStage:
    """Causal-masked softmax over attention scores of shape (B*n_head*T, T)."""

    def __init__(self, cfg: GPTConfig, B: int, T: int, dtype: str = "f32"):
        self.cfg = cfg
        self.B = B
        self.T = T
        self.M = B * cfg.n_head * T
        self.N = T
        self.shape = (self.M, self.N)
        assert dtype == "f32", "Only f32 is supported"
        self.elem_type = get_mlir_elem_type(dtype)
        self.dtype = mlir_to_numpy_dtype(self.elem_type)
        self.memory_manager_class = GPUMemoryManager
        self.payload_function_name = "payload"

    @cached_property
    def _initial_host_arrays(self) -> tuple[np.ndarray, ...]:
        """Host inputs in (output, input) order. `input` is the score matrix
        with the causal mask already applied (future positions = -inf), exactly
        as gpt.py hands it to F.softmax."""
        rng = np.random.default_rng(1337)
        # Raw scores ~ q@k^T * head_size^-0.5; magnitude here is irrelevant to
        # correctness, so just use a reasonable spread.
        raw = rng.uniform(-2.0, 2.0, self.shape).astype(self.dtype)
        # Reshape to (B, n_head, T, T) to mask per (batch, head), then flatten.
        raw4 = raw.reshape(self.B, self.cfg.n_head, self.T, self.T)
        masked = causal_mask_scores(raw4).reshape(self.shape).astype(self.dtype)
        output_arr = np.zeros(self.shape, dtype=self.dtype)
        return (output_arr, masked)

    def payload_module(self) -> ir.Module:
        return generate_gpu_softmax_payload(
            func_name=self.payload_function_name,
            M=self.M,
            N=self.N,
            dtype=self.elem_type,
        )

    def schedule_modules(
        self, stop_at_stage: Optional[str] = None, parameters: Optional[dict] = None
    ) -> list[ir.Module]:
        schedules = [Runner.get_bench_wrapper_schedule(self.payload_function_name)]
        schedules.append(
            softmax_schedule(stop_at_stage=stop_at_stage, parameters=parameters)
        )
        if stop_at_stage and stop_at_stage != "final":
            return schedules
        schedules.append(xegpu_to_binary())
        return schedules

    def shared_libs(self) -> list[str]:
        return ["libmlir_levelzero_runtime.so"]

    def check(self, result: np.ndarray, verbose: int = 0) -> bool:
        _, masked = self._initial_host_arrays
        ref = softmax_ref(masked)
        ok = np.allclose(result, ref, rtol=1e-3, atol=1e-4)
        if verbose:
            if ok:
                print("PASSED")
            else:
                print(f"FAILED! Max abs diff: {np.abs(result - ref).max():.6e}")
        return ok


# ---------------------------------------------------------------------------
# Stage 2b: matmul on XeGPU (C = A @ B).
#
# Reuses the matmul kernel proven by examples/xegpu/matmul.py. Important details:
#   * The DPAS matmul is HARDWIRED to f16 inputs (A, B) + f32 accumulator (C).
#   * The parameter selector only auto-fills tiles for m>=128, n>=256, k>=64.
#     Per the "shapes are free" steer, we use generously large dims so we never
#     fight tile constraints. (Result-arg index 0 is C, same as other stages.)
#   * accumulate_c=False -> pure C = A@B (no read-add of C).
#
# This is the building block for all of nano-gpt's linears + attention matmuls.
# Output arg order from the payload is (C, A, B): C is the result (arg 0).
# ---------------------------------------------------------------------------
class XeGPUMatMulStage:
    """C = A @ B on XeGPU. A,B are f16; C is f32. M,N,K are free dims."""

    def __init__(
        self, M: int, N: int, K: int, A=None, B=None, bias=None, has_relu=False
    ):
        self.M, self.N, self.K = M, N, K
        # Optional injected operands (used when composing, e.g. attention/FFN).
        # Stored as-is; cast to f16 in _initial_host_arrays.
        self._A = A
        self._B = B
        # Fused epilogue: relu(A@B + bias). bias is (N,) f32 or None.
        self._bias = bias
        self.has_bias = bias is not None
        self.has_relu = has_relu
        self.ab_type = ir.F16Type.get()
        self.c_type = ir.F32Type.get()
        self.ab_dtype = mlir_to_numpy_dtype(self.ab_type)
        self.c_dtype = mlir_to_numpy_dtype(self.c_type)
        self.a_shape = (M, K)
        self.b_shape = (K, N)
        self.c_shape = (M, N)
        self.bias_shape = (N,)
        # Result is C -> matches arg_index 0 used by the driver. `.shape`/`.dtype`
        # are the result's, so the generic driver's result buffer is correct.
        self.shape = self.c_shape
        self.dtype = self.c_dtype
        self.memory_manager_class = GPUMemoryManager
        self.payload_function_name = "payload"

    @cached_property
    def _initial_host_arrays(self) -> tuple[np.ndarray, ...]:
        """Host arrays in (C, A, B[, bias]) order. If A/B were injected
        (composition), use them (cast to f16); otherwise generate integer-valued
        randoms to avoid f16/f32 rounding noise, exactly as matmul.py does."""
        if self._A is not None and self._B is not None:
            # ascontiguousarray is REQUIRED: the kernel's host->device copy
            # assumes row-major contiguous memory. A non-contiguous view (e.g.
            # k.T, which only swaps strides) is copied in the wrong layout and
            # silently yields garbage. Force a real contiguous copy here.
            A = np.ascontiguousarray(self._A, dtype=self.ab_dtype)
            B = np.ascontiguousarray(self._B, dtype=self.ab_dtype)
        else:
            rng = np.random.default_rng(2)

            def gen(shape, dtype):
                return rng.integers(-3, 4, shape).astype(dtype)

            A = gen(self.a_shape, self.ab_dtype)
            B = gen(self.b_shape, self.ab_dtype)
        assert A.shape == self.a_shape and B.shape == self.b_shape
        C = np.zeros(self.c_shape, dtype=self.c_dtype)
        if self.has_bias:
            bias = np.ascontiguousarray(self._bias, dtype=self.c_dtype)
            assert bias.shape == self.bias_shape
            return (C, A, B, bias)
        return (C, A, B)

    def payload_module(self) -> ir.Module:
        mod = generate_gpu_matmul_payload(
            func_name=self.payload_function_name,
            M=self.M,
            N=self.N,
            K=self.K,
            ab_type=self.ab_type,
            c_type=self.c_type,
            has_bias=self.has_bias,
            has_relu=self.has_relu,
            accumulate_c=False,
        )
        ranks_and_types = [(2, self.ab_type), (2, self.c_type)]
        if self.has_bias:
            ranks_and_types.append((1, self.c_type))
        self.memory_manager_class.emit_memory_management_funcs(
            mod, ranks_and_types=ranks_and_types
        )
        return mod

    def schedule_modules(
        self, stop_at_stage: Optional[str] = None, parameters: Optional[dict] = None
    ) -> list[ir.Module]:
        schedules = [Runner.get_bench_wrapper_schedule(self.payload_function_name)]
        schedules.append(
            mlp_schedule(stop_at_stage=stop_at_stage, params=[parameters])
        )
        if stop_at_stage and stop_at_stage != "final":
            return schedules
        schedules.append(xegpu_to_binary())
        return schedules

    def matmul_params(self) -> dict:
        """DPAS tile params from the selector (requires large enough M,N,K)."""
        return xegpu_parameter_selector.get_matmul_parameters(self.M, self.N, self.K)

    def shared_libs(self) -> list[str]:
        return ["libmlir_levelzero_runtime.so"]

    def check(self, result: np.ndarray, verbose: int = 0) -> bool:
        arrs = self._initial_host_arrays
        A, B = arrs[1], arrs[2]
        ref = A.astype(np.float32) @ B.astype(np.float32)
        if self.has_bias:
            ref = ref + arrs[3].astype(np.float32)
        if self.has_relu:
            ref = np.maximum(ref, 0.0)
        # f16 inputs -> looser tolerance than the f32-only stages.
        ok = np.allclose(result, ref, rtol=1e-2, atol=1e-2)
        if verbose:
            if ok:
                print("PASSED")
            else:
                print(f"FAILED! Max abs diff: {np.abs(result - ref).max():.6e}")
        return ok


# ---------------------------------------------------------------------------
# CLI / driver
# ---------------------------------------------------------------------------
def parse_cli():
    p = argparse.ArgumentParser(
        description="nano-GPT (gpt.py) inference on XeGPU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--stage",
        type=str,
        default="layernorm",
        choices=["layernorm", "softmax", "matmul", "attention", "ffn", "block", "gpt"],
        help="Which model stage to build and run.",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Batch size B (rows = B*T for layernorm).",
    )
    p.add_argument(
        "--seq-len",
        type=int,
        default=64,
        help="Sequence length T (<= block_size).",
    )
    p.add_argument(
        "--mm-sizes",
        type=int,
        nargs=3,
        default=[256, 256, 256],
        metavar=("M", "N", "K"),
        help="M N K for --stage=matmul (use large dims; selector needs "
        "m>=128,n>=256,k>=64).",
    )
    p.add_argument(
        "--attn-t",
        type=int,
        default=256,
        help="Sequence length T for --stage=attention (large so matmuls fit).",
    )
    p.add_argument(
        "--attn-hs",
        type=int,
        default=256,
        help="Head size for --stage=attention (large so weights@v fits).",
    )
    p.add_argument(
        "--ffn-t",
        type=int,
        default=256,
        help="Sequence length T (rows) for --stage=ffn.",
    )
    p.add_argument(
        "--ffn-n",
        type=int,
        default=256,
        help="n_embd for --stage=ffn; hidden = 4*n_embd (large so matmuls fit).",
    )
    p.add_argument("--blk-t", type=int, default=256, help="T for --stage=block.")
    p.add_argument(
        "--blk-n", type=int, default=256, help="n_embd for --stage=block."
    )
    p.add_argument(
        "--blk-heads",
        type=int,
        default=1,
        help="n_head for --stage=block (default 1: head_size=n_embd clears DPAS "
        "gate; multi-head needs head_size>=256).",
    )
    p.add_argument("--gpt-t", type=int, default=256, help="T (seq len) for --stage=gpt.")
    p.add_argument("--gpt-n", type=int, default=256, help="n_embd for --stage=gpt.")
    p.add_argument("--gpt-layers", type=int, default=2, help="n_layer for --stage=gpt.")
    p.add_argument(
        "--gpt-vocab", type=int, default=256,
        help="vocab_size for --stage=gpt (>=256 for the lm_head DPAS gate).",
    )
    # Schedule tuning knobs (forwarded to layer_norm_schedule).
    p.add_argument("--wg-rows", type=int, default=64)
    p.add_argument("--sg-rows", type=int, default=8)
    p.add_argument("--subgroup-size", type=int, default=16)
    p.add_argument("--reduction-step-size", type=int, default=16)
    p.add_argument(
        "--dump-kernel",
        type=str,
        choices=[
            "initial",
            "tiled",
            "vectorized",
            "bufferized",
            "gpu-outlining",
            "xegpu-initial",
            "xegpu-wg",
            "final",
        ],
        help="Dump kernel IR at a lowering stage and exit (no GPU needed).",
    )
    p.add_argument("--dump-schedule", action="store_true")
    p.add_argument("--check-result", action="store_true")
    p.add_argument(
        "--isolate",
        action="store_true",
        help="Opt in to subprocess-per-kernel isolation for composed stages "
        "(attention/ffn/block/gpt). This was the original workaround for a "
        "libocloc/unwinder deadlock; that deadlock is now fixed at the root by "
        "install_dialect_module_stubs(), so the default is fast single-process "
        "execution. Kept as a fallback only.",
    )
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def build_stage(args, cfg: GPTConfig):
    assert args.seq_len <= cfg.block_size, "T must be <= block_size"
    if args.stage == "layernorm":
        M = args.batch * args.seq_len
        return XeGPULayerNormStage(cfg, M=M, dtype="f32")
    if args.stage == "softmax":
        return XeGPUMaskedSoftmaxStage(cfg, B=args.batch, T=args.seq_len, dtype="f32")
    if args.stage == "matmul":
        M, N, K = args.mm_sizes
        return XeGPUMatMulStage(M=M, N=N, K=K)
    if args.stage == "attention":
        return XeGPUAttentionStage(cfg, T=args.attn_t, head_size=args.attn_hs)
    if args.stage == "ffn":
        # n_embd from CLI (default large enough for the DPAS gate, not cfg's 384).
        return XeGPUFeedForwardStage(T=args.ffn_t, n_embd=args.ffn_n)
    if args.stage == "block":
        return XeGPUBlockStage(
            cfg, T=args.blk_t, n_embd=args.blk_n, n_head=args.blk_heads
        )
    if args.stage == "gpt":
        return XeGPUGPTStage(
            cfg, T=args.gpt_t, n_embd=args.gpt_n, n_layer=args.gpt_layers,
            vocab_size=args.gpt_vocab, n_head=1,
        )
    raise ValueError(f"Unknown stage: {args.stage}")


def stage_params(args, cfg: GPTConfig, wload) -> dict:
    """Schedule parameters. matmul uses the DPAS tile selector; the reduction
    stages (layernorm/softmax) use the wg/sg row knobs."""
    if args.stage == "matmul":
        return wload.matmul_params()
    return {
        "sizes": list(wload.shape),
        "wg_rows": args.wg_rows,
        "sg_rows": args.sg_rows,
        "subgroup_size": args.subgroup_size,
        "reduction_step_size": args.reduction_step_size,
    }


# ---------------------------------------------------------------------------
# Subprocess-per-kernel isolation (works around a toolchain deadlock).
#
# Composing many GPU kernels in ONE process eventually deadlocks: MLIR throws a
# routine C++ exception during op verification; mid-unwind a SIGSEGV fires whose
# handler (installed by Intel's libocloc.so) calls backtrace(), which re-enters
# libgcc's non-reentrant unwinder mutex -> self-deadlock. (Confirmed via gdb;
# see memory.md "ROOT CAUSE FOUND".) In-process workarounds (skip verify, reset
# signal handlers, single-thread) do NOT help.
#
# THE FIX: run each kernel in a FRESH subprocess. Each child has pristine
# ocloc/unwinder state, so the deadlock cannot accumulate. Kernels exchange
# numpy arrays via pickle over stdin/stdout. Toggled by the ISOLATE flag (set
# by --isolate, default on for composed stages).
# ---------------------------------------------------------------------------
ISOLATE = False


def _run_kernel_in_process(spec: dict) -> np.ndarray:
    """Build the kernel workload from a picklable spec, run it on the GPU in the
    CURRENT process/context, and return the result. Used by the worker entry
    point and as the in-process fallback."""
    kind = spec["kind"]
    if kind == "matmul":
        wload = XeGPUMatMulStage(
            M=spec["M"], N=spec["N"], K=spec["K"], A=spec["A"], B=spec["B"],
            bias=spec.get("bias"), has_relu=spec.get("has_relu", False),
        )
        params = wload.matmul_params()
    elif kind == "layernorm":
        wload = _LayerNormRows(spec["x"], spec["gamma"], spec["beta"], spec["eps"])
        params = wload.params()
    elif kind == "softmax":
        wload = _SoftmaxRows(spec["x"])
        params = wload.params()
    else:
        raise ValueError(f"Unknown kernel kind: {kind}")
    return _execute_wload(wload, params)


def _execute_wload(wload, params: dict) -> np.ndarray:
    """Low-level: compile wload's payload + schedule, run on GPU, return result
    (arg 0). Must be called inside an ir.Context."""
    pipeline = TransformDriver(wload.schedule_modules(parameters=params))
    payload = pipeline.apply(wload.payload_module())
    runner = Runner(
        payload,
        mem_manager_cls=wload.memory_manager_class,
        shared_libs=wload.shared_libs(),
    )
    result = np.zeros(wload.shape, dtype=wload.dtype)
    cb = Runner.get_gpu_argument_access_callback(result, arg_index=0)
    runner.execute(
        host_input_buffers=wload._initial_host_arrays,
        payload_function_name=wload.payload_function_name,
        argument_access_callback=cb,
    )
    return result


def run_kernel(spec: dict, label: str = "") -> np.ndarray:
    """Run one GPU kernel described by `spec`. If ISOLATE, run it in a fresh
    subprocess (avoids the libocloc/unwinder deadlock); else run in-process.

    Logs per-kernel timing when PROGRESS is set. This is the orchestration
    primitive used by all composed stages (attention/ffn/block/gpt)."""
    import time

    t0 = time.monotonic()
    if ISOLATE:
        import subprocess, pickle, base64, os

        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "__kernel_worker__"],
            input=base64.b64encode(pickle.dumps(spec)),
            capture_output=True,
            env={**os.environ},
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"kernel worker ({label or spec.get('kind')}) failed:\n"
                + proc.stderr.decode()[-800:]
            )
        result = pickle.loads(base64.b64decode(proc.stdout))
    else:
        result = _run_kernel_in_process(spec)
    if PROGRESS:
        tag = label or spec.get("kind", "kernel")
        mode = "subproc" if ISOLATE else "inproc"
        print(
            f"  [{tag}] {result.shape} {mode}={time.monotonic()-t0:5.2f}s",
            flush=True,
        )
    return result


def run_stage_on_gpu(wload, params: dict, label: str = "") -> np.ndarray:
    """Compile `wload`'s payload with its schedule, execute on the GPU, and
    return a host copy of the result (output arg, index 0). Single-kernel path
    used by the primitive stages (always in-process). Must be called inside an
    `ir.Context`."""
    import time

    t0 = time.monotonic()
    result = _execute_wload(wload, params)
    if PROGRESS:
        tag = label or type(wload).__name__
        print(f"  [{tag}] {wload.shape} {time.monotonic()-t0:5.2f}s", flush=True)
    return result


# Module-level progress flag, toggled by -v on composed stages.
PROGRESS = False


# ---------------------------------------------------------------------------
# Stage 2b: full single-head causal self-attention, COMPOSED from GPU kernels.
#
# This is the first multi-kernel orchestration (Approach A). Per head:
#   1. scores = q @ k^T          -> matmul kernel (f16 in, f32 out)
#   2. host:  scale + causal mask (-inf on future positions)
#   3. weights = softmax(scores) -> softmax kernel (f32)
#   4. out = weights @ v         -> matmul kernel (f16 in, f32 out)
# Intermediates round-trip through host numpy arrays between kernels.
#
# Dims are free (per the "shapes are functional, not fixed" steer): we pick
# T and head_size large enough that BOTH matmuls clear the DPAS selector gate
# (m>=128, n>=256, k>=64). q@k^T is (T,T,hs); weights@v is (T,hs,T).
# ---------------------------------------------------------------------------
class XeGPUAttentionStage:
    """Single-head causal self-attention composed from matmul+softmax kernels."""

    def __init__(self, cfg: GPTConfig, T: int, head_size: int, qkv=None):
        self.cfg = cfg
        self.T = T
        self.head_size = head_size
        # Optional injected (q, k, v) from a composing caller (e.g. the Block).
        self._qkv = qkv
        # `.shape`/`.dtype` describe the final output, for the generic driver.
        self.shape = (T, head_size)
        self.dtype = np.float32
        self.memory_manager_class = GPUMemoryManager

    @cached_property
    def _initial_host_arrays(self) -> tuple[np.ndarray, ...]:
        """q, k, v in (T, head_size). If injected (composition), use those;
        else synthesize. The actual run is done by run_on_gpu."""
        if self._qkv is not None:
            q, k, v = (np.ascontiguousarray(a, dtype=np.float32) for a in self._qkv)
            return (q, k, v)
        rng = np.random.default_rng(7)
        q = rng.uniform(-1.0, 1.0, (self.T, self.head_size)).astype(np.float32)
        k = rng.uniform(-1.0, 1.0, (self.T, self.head_size)).astype(np.float32)
        v = rng.uniform(-1.0, 1.0, (self.T, self.head_size)).astype(np.float32)
        return (q, k, v)

    def run_on_gpu(self, params_unused=None) -> np.ndarray:
        """Orchestrate the attention chain across three GPU kernels (each run via
        run_kernel, i.e. subprocess-isolated when ISOLATE is set)."""
        q, k, v = self._initial_host_arrays
        T, hs = self.T, self.head_size

        # 1. scores = q @ k^T  (B operand is k^T, shape (hs, T))
        scores = run_kernel(
            dict(kind="matmul", M=T, N=T, K=hs, A=q, B=k.T), label="attn_qk"
        )  # (T, T)

        # 2. host: scale + causal mask
        scores = scores * (hs ** -0.5)
        scores = causal_mask_scores(scores).astype(np.float32)

        # 3. weights = softmax(scores)
        weights = run_kernel(dict(kind="softmax", x=scores), label="attn_softmax")

        # 4. out = weights @ v
        return run_kernel(
            dict(kind="matmul", M=T, N=hs, K=T, A=weights, B=v), label="attn_wv"
        )  # (T, hs)

    def check(self, result: np.ndarray, verbose: int = 0) -> bool:
        q, k, v = self._initial_host_arrays
        ref = attention_ref(q, k, v, self.head_size)
        # f16 matmuls -> loose tolerance.
        ok = np.allclose(result, ref, rtol=2e-2, atol=2e-2)
        if verbose:
            if ok:
                print("PASSED")
            else:
                print(f"FAILED! Max abs diff: {np.abs(result - ref).max():.6e}")
        return ok


class _SoftmaxRows:
    """Minimal softmax-over-rows workload around a concrete input matrix.

    Like XeGPUMaskedSoftmaxStage but takes an arbitrary pre-masked (M, N) input
    instead of synthesizing one -- used inside attention composition."""

    def __init__(self, masked_scores: np.ndarray):
        self.M, self.N = masked_scores.shape
        self.shape = (self.M, self.N)
        self.elem_type = get_mlir_elem_type("f32")
        self.dtype = mlir_to_numpy_dtype(self.elem_type)
        self._input = masked_scores.astype(self.dtype)
        self.memory_manager_class = GPUMemoryManager
        self.payload_function_name = "payload"

    @cached_property
    def _initial_host_arrays(self) -> tuple[np.ndarray, ...]:
        output_arr = np.zeros(self.shape, dtype=self.dtype)
        return (output_arr, self._input)

    def params(self) -> dict:
        return {
            "sizes": list(self.shape),
            "wg_rows": 64,
            "sg_rows": 8,
            "subgroup_size": 16,
            "reduction_step_size": 16,
        }

    def payload_module(self) -> ir.Module:
        return generate_gpu_softmax_payload(
            func_name=self.payload_function_name,
            M=self.M,
            N=self.N,
            dtype=self.elem_type,
        )

    def schedule_modules(
        self, stop_at_stage: Optional[str] = None, parameters: Optional[dict] = None
    ) -> list[ir.Module]:
        schedules = [Runner.get_bench_wrapper_schedule(self.payload_function_name)]
        schedules.append(
            softmax_schedule(stop_at_stage=stop_at_stage, parameters=parameters)
        )
        if stop_at_stage and stop_at_stage != "final":
            return schedules
        schedules.append(xegpu_to_binary())
        return schedules

    def shared_libs(self) -> list[str]:
        return ["libmlir_levelzero_runtime.so"]


# ---------------------------------------------------------------------------
# Stage 3: feed-forward network, COMPOSED from two fused matmul kernels.
#
# gpt.py FeedFoward: Linear(n, 4n) -> ReLU -> Linear(4n, n), both with bias.
# The DPAS matmul kernel ALREADY fuses bias + relu as an epilogue, so each
# Linear+activation is a SINGLE kernel call:
#   layer 1: h = relu(x @ W1^T + b1)   -> matmul(has_bias=True, has_relu=True)
#   layer 2: y =      h @ W2^T + b2    -> matmul(has_bias=True, has_relu=False)
#
# nn.Linear stores weight (out, in) and computes x @ W^T; we pass B = W^T
# (made contiguous inside XeGPUMatMulStage). Dims are free: n_embd / 4*n_embd
# are large enough to clear the DPAS selector gate.
# ---------------------------------------------------------------------------
class XeGPUFeedForwardStage:
    """gpt.py FeedFoward composed from two bias+relu-fused matmul kernels."""

    def __init__(self, T: int, n_embd: int, hidden: Optional[int] = None):
        self.T = T
        self.n_embd = n_embd
        self.hidden = hidden if hidden is not None else 4 * n_embd
        self.shape = (T, n_embd)
        self.dtype = np.float32
        self.memory_manager_class = GPUMemoryManager

    @cached_property
    def _initial_host_arrays(self) -> tuple[np.ndarray, ...]:
        """x and both Linear weights/biases. Weights in nn.Linear (out, in)
        layout. Small spread keeps f16 matmul error low."""
        rng = np.random.default_rng(11)
        n, h, T = self.n_embd, self.hidden, self.T
        x = rng.uniform(-1.0, 1.0, (T, n)).astype(np.float32)
        w1 = rng.uniform(-0.5, 0.5, (h, n)).astype(np.float32)  # (4n, n)
        b1 = rng.uniform(-0.1, 0.1, (h,)).astype(np.float32)
        w2 = rng.uniform(-0.5, 0.5, (n, h)).astype(np.float32)  # (n, 4n)
        b2 = rng.uniform(-0.1, 0.1, (n,)).astype(np.float32)
        return (x, w1, b1, w2, b2)

    def run_on_gpu(self, params_unused=None) -> np.ndarray:
        x, w1, b1, w2, b2 = self._initial_host_arrays
        T, n, h = self.T, self.n_embd, self.hidden

        # layer 1: h_act = relu(x @ w1^T + b1)   (x:(T,n), w1^T:(n,4n))
        h_act = run_kernel(
            dict(kind="matmul", M=T, N=h, K=n, A=x, B=w1.T, bias=b1, has_relu=True),
            label="ffn1",
        )  # (T, 4n)

        # layer 2: y = h_act @ w2^T + b2          (h_act:(T,4n), w2^T:(4n,n))
        return run_kernel(
            dict(kind="matmul", M=T, N=n, K=h, A=h_act, B=w2.T, bias=b2),
            label="ffn2",
        )  # (T, n)

    def check(self, result: np.ndarray, verbose: int = 0) -> bool:
        x, w1, b1, w2, b2 = self._initial_host_arrays
        ref = ffn_ref(x, w1, b1, w2, b2)
        ok = np.allclose(result, ref, rtol=2e-2, atol=2e-2)
        if verbose:
            if ok:
                print("PASSED")
            else:
                print(f"FAILED! Max abs diff: {np.abs(result - ref).max():.6e}")
        return ok


class _LayerNormRows:
    """LayerNorm over rows of a concrete (M, N) input with given gamma/beta.

    Like XeGPULayerNormStage but takes injected x/gamma/beta -- used to compose
    LayerNorm inside the Block. Payload arg order: (output, input, gamma, beta)."""

    def __init__(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float):
        self.M, self.N = x.shape
        self.shape = (self.M, self.N)
        self.eps = eps
        self.elem_type = get_mlir_elem_type("f32")
        self.dtype = mlir_to_numpy_dtype(self.elem_type)
        self._x = np.ascontiguousarray(x, dtype=self.dtype)
        self._g = np.ascontiguousarray(gamma, dtype=self.dtype)
        self._b = np.ascontiguousarray(beta, dtype=self.dtype)
        self.memory_manager_class = GPUMemoryManager
        self.payload_function_name = "payload"

    @cached_property
    def _initial_host_arrays(self) -> tuple[np.ndarray, ...]:
        output_arr = np.zeros(self.shape, dtype=self.dtype)
        return (output_arr, self._x, self._g, self._b)

    def params(self) -> dict:
        return {
            "sizes": list(self.shape),
            "wg_rows": 64,
            "sg_rows": 8,
            "subgroup_size": 16,
            "reduction_step_size": 16,
        }

    def payload_module(self) -> ir.Module:
        return generate_gpu_layer_norm_payload(
            func_name=self.payload_function_name,
            M=self.M,
            N=self.N,
            dtype=self.elem_type,
            eps=self.eps,
        )

    def schedule_modules(
        self, stop_at_stage: Optional[str] = None, parameters: Optional[dict] = None
    ) -> list[ir.Module]:
        schedules = [Runner.get_bench_wrapper_schedule(self.payload_function_name)]
        schedules.append(
            layer_norm_schedule(stop_at_stage=stop_at_stage, parameters=parameters)
        )
        if stop_at_stage and stop_at_stage != "final":
            return schedules
        schedules.append(xegpu_to_binary())
        return schedules

    def shared_libs(self) -> list[str]:
        return ["libmlir_levelzero_runtime.so"]


# ---------------------------------------------------------------------------
# Stage 4: full transformer Block, COMPOSED from all the GPU primitives.
#
# gpt.py Block.forward:
#     x = x + sa(ln1(x))     # multi-head self-attention sub-block
#     x = x + ffwd(ln2(x))   # feed-forward sub-block
#
# Pipeline (all on GPU except trivial host residual adds / concat):
#   ln1 (layernorm) -> Q,K,V projections (matmul, no bias) ->
#   per-head attention (XeGPUAttentionStage) -> concat (host) ->
#   output proj (matmul + bias) -> + residual (host) ->
#   ln2 (layernorm) -> FFN (XeGPUFeedForwardStage path) -> + residual (host)
#
# Heads: scores@v is (T, head_size, T), so head_size must clear the DPAS gate
# (>=256). With multi-head, head_size = n_embd/n_head shrinks below that. Per the
# "shapes are free" steer we default to n_head=1 (head_size=n_embd). Multi-head
# needs head_size padding or a small-matmul kernel (future work).
# ---------------------------------------------------------------------------
def _mm(A, B, bias=None, has_relu=False, label=""):
    """matmul kernel helper: C = [relu](A@B [+ bias]). Used by block/GPT."""
    M, K = A.shape
    N = B.shape[1]
    return run_kernel(
        dict(kind="matmul", M=M, N=N, K=K, A=A, B=B, bias=bias, has_relu=has_relu),
        label=label,
    )


def run_block_on_gpu(cfg, x, p, n_head, head_size, eps, tag=""):
    """Run one transformer Block on the GPU and return its output (T, n_embd).

    x: (T, n_embd) input activations. p: per-block weight dict (block_ref format).
    Shared by XeGPUBlockStage and the full-GPT stage so both drive identical
    GPU compute. All kernels go through run_kernel (subprocess-isolated)."""
    T = x.shape[0]
    hs, nh = head_size, n_head

    # --- attention sub-block ---
    xn1 = run_kernel(
        dict(kind="layernorm", x=x, gamma=p["g1"], beta=p["b1"], eps=eps),
        label=f"{tag}ln1",
    )
    q = _mm(xn1, p["wq"].T, label=f"{tag}proj_q")
    k = _mm(xn1, p["wk"].T, label=f"{tag}proj_k")
    v = _mm(xn1, p["wv"].T, label=f"{tag}proj_v")

    head_outs = []
    for hd in range(nh):
        sl = slice(hd * hs, (hd + 1) * hs)
        head = XeGPUAttentionStage(
            cfg, T=T, head_size=hs, qkv=(q[:, sl], k[:, sl], v[:, sl])
        )
        head_outs.append(head.run_on_gpu())
    concat = np.concatenate(head_outs, axis=-1)

    attn = _mm(concat, p["wproj"].T, bias=p["bproj"], label=f"{tag}out_proj")
    x = x + attn  # host residual

    # --- feed-forward sub-block ---
    xn2 = run_kernel(
        dict(kind="layernorm", x=x, gamma=p["g2"], beta=p["b2"], eps=eps),
        label=f"{tag}ln2",
    )
    h1 = _mm(xn2, p["w1"].T, bias=p["bf1"], has_relu=True, label=f"{tag}ffn1")
    ff = _mm(h1, p["w2"].T, bias=p["bf2"], label=f"{tag}ffn2")
    return x + ff  # host residual


class XeGPUBlockStage:
    """One gpt.py transformer Block composed from GPU kernels."""

    def __init__(self, cfg: GPTConfig, T: int, n_embd: int, n_head: int = 1):
        assert n_embd % n_head == 0
        self.cfg = cfg
        self.T = T
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.eps = cfg.ln_eps
        self.shape = (T, n_embd)
        self.dtype = np.float32
        self.memory_manager_class = GPUMemoryManager

    @cached_property
    def _weights(self) -> dict:
        rng = np.random.default_rng(23)
        n, h = self.n_embd, 4 * self.n_embd

        def u(shape, lo=-0.5, hi=0.5):
            return rng.uniform(lo, hi, shape).astype(np.float32)

        return {
            "x": u((self.T, n), -1.0, 1.0),
            "g1": u((n,), 0.5, 1.5),
            "b1": u((n,), -0.1, 0.1),
            "wq": u((n, n)),
            "wk": u((n, n)),
            "wv": u((n, n)),
            "wproj": u((n, n)),
            "bproj": u((n,), -0.1, 0.1),
            "g2": u((n,), 0.5, 1.5),
            "b2": u((n,), -0.1, 0.1),
            "w1": u((h, n)),
            "bf1": u((h,), -0.1, 0.1),
            "w2": u((n, h)),
            "bf2": u((n,), -0.1, 0.1),
        }

    @cached_property
    def _initial_host_arrays(self) -> tuple[np.ndarray, ...]:
        return (self._weights["x"],)

    def run_on_gpu(self, params_unused=None) -> np.ndarray:
        p = self._weights
        return run_block_on_gpu(
            self.cfg, p["x"], p, self.n_head, self.head_size, self.eps
        )

    def check(self, result: np.ndarray, verbose: int = 0) -> bool:
        ref = block_ref(
            self._weights["x"], self._weights, self.n_head, self.head_size, self.eps
        )
        # A full block chains ~10 f16 matmuls over a wide output range, so absolute
        # diff grows with the range. Validate via RELATIVE error instead: the f16
        # DPAS path inherently differs from f32 by ~0.3-0.4% here (verified against
        # an independent f16-rounded reference). See memory.md.
        rel = np.abs(result - ref).max() / max(np.abs(ref).max(), 1e-6)
        ok = rel < 0.02
        if verbose:
            print(f"  (relative error: {rel:.4f})")
        if verbose:
            if ok:
                print("PASSED")
            else:
                print(f"FAILED! Max abs diff: {np.abs(result - ref).max():.6e}")
        return ok


# ---------------------------------------------------------------------------
# Stage 5: full gpt.py forward (logits), COMPOSED end-to-end on the GPU.
#
# Pipeline (gpt.py GPTLanguageModel.forward, inference):
#   host: x = wte[tok_ids] + wpe[positions]      (token + position embeddings)
#   GPU:  for each of n_layer blocks: x = block(x)
#   GPU:  x = ln_f(x)                             (final LayerNorm)
#   GPU:  logits = x @ lm_head^T + lm_b           (vocab projection)
#   host: (caller) softmax + argmax/sample
#
# Embeddings are a host-side gather+add (no embedding kernel needed for v1).
# Dims free: n_head=1 (head_size=n_embd) and n_embd/vocab >= 256 so every matmul
# (incl. lm_head, shape (T, vocab, n_embd)) clears the DPAS selector gate.
# ---------------------------------------------------------------------------
class XeGPUGPTStage:
    """Full gpt.py forward producing logits, composed on the GPU."""

    def __init__(self, cfg: GPTConfig, T: int, n_embd: int, n_layer: int,
                 vocab_size: int, n_head: int = 1):
        assert n_embd % n_head == 0
        self.cfg = cfg
        self.T = T
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.eps = cfg.ln_eps
        self.shape = (T, vocab_size)
        self.dtype = np.float32
        self.memory_manager_class = GPUMemoryManager

    @cached_property
    def _model(self) -> dict:
        rng = np.random.default_rng(31)
        n, h, V = self.n_embd, 4 * self.n_embd, self.vocab_size

        def u(shape, lo=-0.5, hi=0.5):
            return rng.uniform(lo, hi, shape).astype(np.float32)

        def block_w():
            return {
                "g1": u((n,), 0.5, 1.5), "b1": u((n,), -0.1, 0.1),
                "wq": u((n, n)), "wk": u((n, n)), "wv": u((n, n)),
                "wproj": u((n, n)), "bproj": u((n,), -0.1, 0.1),
                "g2": u((n,), 0.5, 1.5), "b2": u((n,), -0.1, 0.1),
                "w1": u((h, n)), "bf1": u((h,), -0.1, 0.1),
                "w2": u((n, h)), "bf2": u((n,), -0.1, 0.1),
            }

        return {
            "tok_ids": rng.integers(0, V, (self.T,)),
            "wte": u((V, n), -0.1, 0.1),      # token embedding table
            "wpe": u((self.cfg.block_size, n), -0.1, 0.1),  # position embeddings
            "blocks": [block_w() for _ in range(self.n_layer)],
            "lnf_g": u((n,), 0.5, 1.5), "lnf_b": u((n,), -0.1, 0.1),
            "lm_w": u((V, n)), "lm_b": u((V,), -0.1, 0.1),
        }

    @cached_property
    def _initial_host_arrays(self) -> tuple[np.ndarray, ...]:
        return (self._model["tok_ids"],)

    def run_on_gpu(self, params_unused=None) -> np.ndarray:
        m = self._model
        tok = m["tok_ids"]
        T = self.T

        # host: token + position embeddings
        x = (m["wte"][tok] + m["wpe"][np.arange(T)]).astype(np.float32)  # (T, n)

        # GPU: N transformer blocks
        for i, p in enumerate(m["blocks"]):
            x = run_block_on_gpu(
                self.cfg, x, p, self.n_head, self.head_size, self.eps,
                tag=f"L{i}.",
            )

        # GPU: final LayerNorm
        x = run_kernel(
            dict(kind="layernorm", x=x, gamma=m["lnf_g"], beta=m["lnf_b"], eps=self.eps),
            label="ln_f",
        )

        # GPU: lm_head projection -> logits (T, vocab)
        return _mm(x, m["lm_w"].T, bias=m["lm_b"], label="lm_head")

    def check(self, result: np.ndarray, verbose: int = 0) -> bool:
        m = self._model
        ref = gpt_ref(
            m["tok_ids"], m["wte"], m["wpe"], m["blocks"], m["lnf_g"], m["lnf_b"],
            m["lm_w"], m["lm_b"], self.n_head, self.head_size, self.eps,
        )
        rel = np.abs(result - ref).max() / max(np.abs(ref).max(), 1e-6)
        # Argmax (the actual next-token prediction) is what matters for generation.
        argmax_match = (result.argmax(-1) == ref.argmax(-1)).mean()
        # Tolerance reflects deep f16 accumulation: a full GPT chains ~12*n_layer
        # f16 matmuls. Verified the GPU matches an independent f16-rounded reference
        # to ~1e-4 (rel 0.034 / argmax 0.992 for 2 layers IS the f16 limit, not our
        # error). So validate against the f16-achievable bar. See memory.md.
        ok = rel < 0.05 and argmax_match >= 0.98
        if verbose:
            print(f"  (relative error: {rel:.4f}, argmax match: {argmax_match:.3f})")
            print("PASSED" if ok else f"FAILED! rel={rel:.4f} argmax={argmax_match:.3f}")
        return ok


def _kernel_worker_main():
    """Hidden entry point: run ONE kernel (spec piped as b64 pickle on stdin) in
    this fresh process and write the result (b64 pickle) to stdout. Used by
    run_kernel under ISOLATE to dodge the libocloc/unwinder deadlock."""
    import pickle, base64

    spec = pickle.loads(base64.b64decode(sys.stdin.buffer.read()))
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        install_dialect_module_stubs()
        result = _run_kernel_in_process(spec)
    sys.stdout.buffer.write(base64.b64encode(pickle.dumps(result)))
    sys.stdout.buffer.flush()


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "__kernel_worker__":
    _kernel_worker_main()
    raise SystemExit(0)


if __name__ == "__main__":
    args = parse_cli()
    cfg = GPTConfig()
    PROGRESS = args.verbose > 0

    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        install_dialect_module_stubs()
        wload = build_stage(args, cfg)

        # Composed stages (multi-kernel) expose run_on_gpu instead of a single
        # payload; they can't be dump-kernel'd as one module.
        if hasattr(wload, "run_on_gpu"):
            if args.dump_kernel or args.dump_schedule:
                raise SystemExit(
                    f"--dump-kernel/--dump-schedule not supported for composed "
                    f"stage '{args.stage}' (it is multiple kernels)."
                )
            # The libocloc/unwinder deadlock is fixed at the root by
            # install_dialect_module_stubs(), so composed stages run in a single
            # process by default (~2x faster, and a prerequisite for on-device
            # buffer reuse + compile caching). --isolate opts back into the old
            # subprocess-per-kernel workaround as a fallback.
            ISOLATE = args.isolate
            result_host_copy = wload.run_on_gpu()
            if args.check_result:
                if not wload.check(result_host_copy, verbose=max(args.verbose, 1)):
                    raise ValueError("Result mismatch!")
                print("Result is correct.")
            else:
                print(f"stage={args.stage} executed (use --check-result to verify).")
            raise SystemExit(0)

        params = stage_params(args, cfg, wload)

        if args.dump_kernel or args.dump_schedule:
            pipeline = TransformDriver(
                wload.schedule_modules(stop_at_stage=args.dump_kernel, parameters=params)
            )
            payload = pipeline.apply(wload.payload_module())
            if args.dump_kernel:
                print(payload)
            if args.dump_schedule:
                for schedule_module in wload.schedule_modules(parameters=params):
                    print(schedule_module)
        else:
            result_host_copy = run_stage_on_gpu(wload, params)
            if args.check_result:
                if not wload.check(result_host_copy, verbose=max(args.verbose, 1)):
                    raise ValueError("Result mismatch!")
                print("Result is correct.")
            else:
                print(f"stage={args.stage} executed (use --check-result to verify).")
