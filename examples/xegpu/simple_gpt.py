"""simple_gpt.py -- the full nano-GPT (gpt.py) forward as ONE XeGPU module.

Runs Karpathy's gpt.py forward/inference on the Intel GPU,
lowered as a SINGLE MLIR module to many SEPARATE un-fused XeGPU kernels with
on-device handoff (gpu.alloc buffers between kernels). This is
"whole model in one module" structure, GPU-targeted. No cross-kernel fusion.

gpt.py forward:
    x = token_emb + pos_emb                 # embeddings (done host-side)
    for _ in range(n_layer): x = Block(x)    # Block: x+sa(ln1(x)); x+ffwd(ln2(x))
    x = ln_f(x); logits = lm_head(x)
where ffwd = Linear(C,4C) -> ReLU -> Linear(4C,C).

Every elementwise op (cast, bias-add, relu, residual-add, scale+mask, transpose)
is its OWN kernel: materialize its result into a device buffer + pre-tile into its
own forall.

Config: n_layer=6 (matches gpt.py). Other dims use proven kernel-friendly shapes
(C=256, hidden=1024, vocab=256, T=256) so every matmul clears the DPAS gate
(m>=128, n>=256, k>=64) -- NOT gpt.py's exact n_embd=384/n_head=6. SINGLE-HEAD
(hs=C); true n_head=6 (head_size=64) is a separate milestone (needs multi-head
attention + a fix for the head_size=64 scores@v matmul which fails the DPAS gate).

Variants (--variant): build-up stages, all lowered as one module each --
  ffn    : ln -> ffn1(mm)+bias+relu -> ffn2(mm)+bias -> residual          (~8 kernels)
  attn   : ln -> q/k/v proj -> scores -> scale+mask -> softmax -> @v -> proj -> res
  block  : full transformer block (attn sublayer + ffn sublayer)          (26 kernels)
  gpt    : full model -- n_layer Blocks -> ln_f -> lm_head -> logits       (default)

Run:
  export PYTHONPATH=/home/jovyan/llvm-project/upstream/tools/mlir/python_packages/mlir_core
  .venv/bin/python examples/xegpu/simple_gpt.py --variant gpt [--gpt-layers N] [--dump STAGE] [--check]
"""
import sys
import numpy as np
from mlir import ir
from mlir.dialects import linalg, bufferization, tensor, arith, math, gpu
from mlir.dialects import transform
from mlir.dialects.transform import structured, loop, xegpu
from mlir.dialects.transform import bufferization as transform_bufferization
from mlir.dialects.bufferization import LayoutMapOption

from lighthouse import dialects as lh_dialects
import lighthouse.transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.pipeline.helper import (
    apply_registered_pass, canonicalize, match, match_and_split, PipelineInterrupt,
)
from lighthouse.schedule import schedule_boilerplate
from lighthouse.ingress.mlir_gen.utils import emit_buf_to_tensor, affine_map, parallel, reduction
from lighthouse.ingress.mlir_gen.gpu_utils import emit_gpu_util_funcs
from lighthouse.ingress.mlir_gen.named import times_weights
from lighthouse.pipeline.driver import TransformDriver
from lighthouse.execution.runner import Runner
from lighthouse.execution import GPUMemoryManager
from lighthouse.schedule.xegpu import xegpu_to_binary, xegpu_parameter_selector
from lighthouse.schedule.xegpu.mlp_schedule import xegpu_wg_annotation_for_mlp_layer


# =============================================================================
# HOW THIS FILE IS ORGANIZED
# =============================================================================
# Compiling a model to the GPU here happens in THREE stages. The code is grouped
# to match them:
#
#   1. PAYLOAD  -- "WHAT to compute".  Build an MLIR module describing the math
#      as high-level `linalg` ops (matmul, layernorm, softmax, elementwise). This
#      is hardware-agnostic; it does not say how to run on the GPU yet.
#         -> class `Builder` (emits one op at a time) and the `build_*_payload`
#            functions (assemble ops into ffn / attn / block / full-gpt).
#
#   2. SCHEDULE -- "HOW to lower it to the GPU".  Build a SECOND MLIR module (a
#      "transform dialect" script) that rewrites the payload: tile each op into
#      GPU work-groups, vectorize, bufferize, outline each op into its own GPU
#      kernel, and attach XeGPU layout/target attributes.
#         -> `build_combined_schedule` / `_bundle` (the orchestrator) plus the
#            `_tile_one_matmul` / `_tile_softmax` / `_tile_one_layernorm` helpers.
#         *** THIS is the part that "schedules the passes". ***
#
#   3. DRIVER   -- "run it".  `main()` applies the schedule to the payload
#      (TransformDriver), JIT-compiles + runs it on the GPU (Runner), and checks
#      the result against a plain-numpy reference (`numpy_ref_*`).
#
# KEY IDEA -- one module, many separate kernels: the whole model is ONE MLIR
# module, but each op becomes its OWN GPU kernel (no cross-op fusion). Data passes
# between kernels through device buffers (`gpu.alloc`) that stay on the GPU -- no
# round-trip to the host between ops.
# =============================================================================


def install_dialect_module_stubs():
    """Work around a toolchain deadlock by silencing a benign-but-repeated error.

    MLIR's Python bindings probe `mlir.dialects.<name>` for every op when building
    op views. For dialects that have no Python module (xegpu, xevm, and some custom
    transform ops) this probe raises a C++ ModuleNotFoundError that is caught but
    NEVER cached -- so it re-throws on every op. That repeated exception races with
    the Intel GPU compiler's (libocloc) signal handler and DEADLOCKS the process.

    Fix: install a sys.meta_path finder that hands back an empty stub module for any
    otherwise-missing `mlir.dialects.<name>`, so the probe succeeds (and is cached)
    and the exception never fires. This is what lets the whole model run IN ONE
    PROCESS (no subprocess-per-kernel workaround). See memory.md part 4.
    """
    import types, importlib.abc, importlib.machinery
    prefix = "mlir.dialects."
    class F(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, fullname, path, target=None):
            # Only handle direct children `mlir.dialects.<name>`, and only as a
            # last resort (appended at the END of meta_path, so real modules win).
            if fullname.startswith(prefix) and fullname.count(".") == 2:
                return importlib.machinery.ModuleSpec(fullname, self)
            return None
        def create_module(self, spec):
            return types.ModuleType(spec.name)   # empty stub
        def exec_module(self, module):
            pass
    sys.meta_path.append(F())


F32 = lambda: ir.F32Type.get()   # 32-bit float (used for accumulation / norms)
F16 = lambda: ir.F16Type.get()   # 16-bit float (required by the GPU matmul units)


# =============================================================================
# STAGE 1 -- PAYLOAD: describe WHAT to compute (hardware-agnostic linalg ops)
# =============================================================================
# Each Builder method emits ONE high-level op that writes its result into a fresh
# on-device buffer (`gpu.alloc`), and returns a tensor "view" of that buffer for
# the next op to read. Because each op writes a distinct device buffer, each will
# become its OWN GPU kernel later; the buffers are the on-device handoff between
# kernels (kernel N writes buffer B, kernel N+1 reads B -- no host round-trip).
#
# dtype convention: the GPU matmul (DPAS) hardware needs f16 inputs and produces
# an f32 result. LayerNorm/softmax run in f32. So between a norm/softmax and a
# matmul we insert an explicit f32->f16 `cast` op (its own kernel).
# =============================================================================
class Builder:
    """Emits the model's ops and remembers the order/kind of each one.

    `kinds` is the crucial bookkeeping: an ordered list, one entry per op emitted,
    recording its "class" so the SCHEDULE (stage 2) can later tile and annotate
    each kernel correctly. Classes:
      'mm'  = matmul (linalg.matmul)          -> DPAS systolic-array kernel
      'ln'  = layernorm (3 generics + 2 fills) -> reduction kernel (uses shared mem)
      'sm'  = softmax (linalg.softmax)         -> reduction kernel (uses shared mem)
      'ew'  = elementwise (cast/bias/relu/residual/scale+mask/transpose) -> simple
              row-parallel kernel
    The op build order in the payload == the order of `kinds` == the order the
    kernels appear in the final module, which is how the schedule matches them up.
    """
    def __init__(self, T):
        self.T = T
        self.f32, self.f16 = F32(), F16()
        self.kinds = []          # ordered kernel classes (see docstring)
        self.to_dealloc = []     # device buffers to gpu.dealloc at the end

    def _buf(self, shape, dtype):
        # Allocate a DEVICE buffer (lives in GPU memory). Returns the memref.
        b = gpu.alloc(ir.MemRefType.get(shape, dtype), None, [], [], [])
        self.to_dealloc.append(b)
        return b

    def _par(self, rank=2):
        # Identity affine map (d0,d1,...) -> (d0,d1,...): a plain elementwise
        # access pattern where output[i,j] depends on input[i,j].
        return affine_map(rank, [ir.AffineDimExpr.get(i) for i in range(rank)])

    # ---- matmul: a(M,K) f16 @ b(K,N) f16 -> (M,N) f32 buffer ----
    def matmul(self, a, b, M, N, out_buf=None):
        # Standard C = A @ B. `times_weights` emits linalg.matmul; we first fill the
        # accumulator with 0. f16 inputs, f32 output -- matches the DPAS hardware.
        buf = out_buf if out_buf is not None else self._buf((M, N), self.f32)
        out_t = emit_buf_to_tensor(buf, restrict=True, writable=True)
        acc = linalg.fill(arith.constant(self.f32, 0.0), outs=[out_t])
        res = times_weights(a, b, acc)
        bufferization.materialize_in_destination(None, res, buf, restrict=True, writable=True)
        self.kinds.append("mm")
        if out_buf is not None:            # caller gave the final output buffer
            return None
        return emit_buf_to_tensor(buf, restrict=True)

    # ---- layernorm(x (M,N) f32, gamma,beta (N,)) -> (M,N) f32 buffer ----
    def layernorm(self, x, gamma, beta, M, N, eps=1e-5):
        # LayerNorm normalizes each ROW (length N) to mean 0 / variance 1, then
        # scales by gamma and shifts by beta. Built from 3 linalg.generic ops:
        #   (1) mean_sum[i] = sum_j x[i,j]                 (row reduction)
        #   (2) var_sum[i]  = sum_j (x[i,j]-mean_i)^2      (row reduction)
        #   (3) out[i,j]    = (x[i,j]-mean_i)*rsqrt(var_i+eps)*gamma[j] + beta[j]
        # Affine maps describe each operand's access pattern:
        #   par2  (d0,d1)->(d0,d1) : full 2-D elementwise
        #   red2  (d0,d1)->(d0)    : reduce over j -> one value per row
        #   bias2 (d0,d1)->(d1)    : gamma/beta indexed by column only
        f32 = self.f32
        par2, red2 = self._par(), affine_map(2, [ir.AffineDimExpr.get(0)])
        bias2 = affine_map(2, [ir.AffineDimExpr.get(1)])
        inv_n = arith.constant(f32, 1.0 / float(N))
        eps_c = arith.constant(f32, eps)
        zero = arith.constant(f32, 0.0)
        # (1) row sums -> mean_sum (linalg.fill zeroes the accumulator first)
        mean_acc = linalg.fill(zero, outs=[tensor.empty((M,), f32)])
        @linalg.generic([x], [mean_acc], [par2, red2], [parallel, reduction])
        def mean_sum(v, acc):
            return arith.AddFOp(v, acc)
        # (2) sum of squared deviations -> var_sum (mean_i = mean_sum_i / N)
        var_acc = linalg.fill(zero, outs=[tensor.empty((M,), f32)])
        @linalg.generic([x, mean_sum], [var_acc], [par2, red2, red2], [parallel, reduction])
        def var_sum(v, m_sum, acc):
            mean = arith.MulFOp(m_sum, inv_n).result
            c = arith.SubFOp(v, mean).result
            return arith.AddFOp(arith.MulFOp(c, c).result, acc)
        # (3) normalize + scale + shift -> output
        buf = self._buf((M, N), f32)
        out_t = emit_buf_to_tensor(buf, restrict=True, writable=True)
        @linalg.generic([x, mean_sum, var_sum, gamma, beta], [out_t],
                        [par2, red2, red2, bias2, bias2, par2], [parallel, parallel])
        def normed(v, m_sum, v_sum, g, b, _o):
            mean = arith.MulFOp(m_sum, inv_n).result
            var = arith.MulFOp(v_sum, inv_n).result
            inv_std = math.rsqrt(arith.AddFOp(var, eps_c).result)
            c = arith.SubFOp(v, mean).result
            return arith.AddFOp(arith.MulFOp(arith.MulFOp(c, inv_std).result, g).result, b)
        bufferization.materialize_in_destination(None, normed, buf, restrict=True, writable=True)
        self.kinds.append("ln")
        return emit_buf_to_tensor(buf, restrict=True)

    # ---- softmax(x (M,N) f32) over last dim -> (M,N) f32 buffer ----
    def softmax(self, x, M, N):
        # softmax along each row: out[i,j] = exp(x[i,j]-max_i) / sum_k exp(x[i,k]-max_i).
        # Emitted as one linalg.softmax op; the schedule decomposes it into the
        # max/exp/sum/divide reduction steps later (see `_tile_softmax`).
        buf = self._buf((M, N), self.f32)
        out_t = emit_buf_to_tensor(buf, restrict=True, writable=True)
        sm = linalg.softmax(result=[ir.RankedTensorType.get((M, N), self.f32)],
                            input=x, output=out_t, dimension=1)
        bufferization.materialize_in_destination(None, sm, buf, restrict=True, writable=True)
        self.kinds.append("sm")
        return emit_buf_to_tensor(buf, restrict=True)

    # ---- elementwise cast f32 -> f16 ----
    def cast_f16(self, x, M, N):
        par2 = self._par()
        buf = self._buf((M, N), self.f16)
        out_t = emit_buf_to_tensor(buf, restrict=True, writable=True)
        @linalg.generic([x], [out_t], [par2, par2], [parallel, parallel])
        def c(s, _o):
            return arith.TruncFOp(self.f16, s)
        bufferization.materialize_in_destination(None, c, buf, restrict=True, writable=True)
        self.kinds.append("ew")
        return emit_buf_to_tensor(buf, restrict=True)

    # ---- bias add (+ optional relu): out = max(x + bias, 0)?  x (M,N) f32, bias (N,) ----
    def bias(self, x, bias_vec, M, N, relu=False, out_buf=None):
        par2 = self._par()
        bias2 = affine_map(2, [ir.AffineDimExpr.get(1)])
        zero = arith.constant(self.f32, 0.0)
        buf = out_buf if out_buf is not None else self._buf((M, N), self.f32)
        out_t = emit_buf_to_tensor(buf, restrict=True, writable=True)
        @linalg.generic([x, bias_vec], [out_t], [par2, bias2, par2], [parallel, parallel])
        def b(v, bb, _o):
            s = arith.AddFOp(v, bb).result
            if relu:
                return arith.MaximumFOp(s, zero)
            return arith.AddFOp(s, zero)  # identity wrap so the op has a body
        bufferization.materialize_in_destination(None, b, buf, restrict=True, writable=True)
        self.kinds.append("ew")
        if out_buf is not None:
            return None
        return emit_buf_to_tensor(buf, restrict=True)

    # ---- residual add: out = a + b  (both (M,N) f32) ----
    def add(self, a, b, M, N, out_buf=None):
        par2 = self._par()
        buf = out_buf if out_buf is not None else self._buf((M, N), self.f32)
        out_t = emit_buf_to_tensor(buf, restrict=True, writable=True)
        @linalg.generic([a, b], [out_t], [par2, par2, par2], [parallel, parallel])
        def r(x, y, _o):
            return arith.AddFOp(x, y)
        bufferization.materialize_in_destination(None, r, buf, restrict=True, writable=True)
        self.kinds.append("ew")
        if out_buf is not None:
            return None
        return emit_buf_to_tensor(buf, restrict=True)

    # ---- transpose: out(N,M) = x(M,N)^T, f32. (Transpose must be done in f32:
    #      XeVM blockload2d-with-transpose only supports 32/64-bit elements, NOT
    #      f16. So transpose the f32 k, THEN cast to f16 for the DPAS matmul.) ----
    def transpose_f32(self, x, M, N):
        # Use the generic with output map (d0,d1)->(d1,d0), iterating over the
        # OUTPUT space (N,M) so the store is row-contiguous over the output. The
        # input read is then the transposed (column) access. Iterating the output
        # rows (tiled by wg_rows) keeps store_nd row-contiguous -> correct lowering.
        # iteration dims = (d0,d1) over output (N,M); read x[d1,d0], write out[d0,d1].
        in_map = affine_map(2, [ir.AffineDimExpr.get(1), ir.AffineDimExpr.get(0)])
        out_map = affine_map(2, [ir.AffineDimExpr.get(0), ir.AffineDimExpr.get(1)])
        buf = self._buf((N, M), self.f32)
        out_t = emit_buf_to_tensor(buf, restrict=True, writable=True)
        @linalg.generic([x], [out_t], [in_map, out_map], [parallel, parallel])
        def t(v, _o):
            return arith.AddFOp(v, arith.constant(self.f32, 0.0))  # identity (needs body)
        bufferization.materialize_in_destination(None, t, buf, restrict=True, writable=True)
        self.kinds.append("ew")
        return emit_buf_to_tensor(buf, restrict=True)

    # ---- scale + additive mask: out = x*scale + mask  (x,mask (M,N) f32) ----
    def scale_mask(self, x, mask, scale, M, N):
        par2 = self._par()
        scale_c = arith.constant(self.f32, scale)
        buf = self._buf((M, N), self.f32)
        out_t = emit_buf_to_tensor(buf, restrict=True, writable=True)
        @linalg.generic([x, mask], [out_t], [par2, par2, par2], [parallel, parallel])
        def sm(s, m, _o):
            return arith.AddFOp(arith.MulFOp(s, scale_c).result, m)
        bufferization.materialize_in_destination(None, sm, buf, restrict=True, writable=True)
        self.kinds.append("ew")
        return emit_buf_to_tensor(buf, restrict=True)

    # ---- single-head self-attention(x16 (T,C) f16) -> (T,C) f32 ----
    def attention(self, x16, wq, wk, wv, mask, scale, T, C, hs):
        # The transformer's core: every position attends to earlier positions.
        #   q,k,v = x@Wq, x@Wk, x@Wv          (projections, no bias)
        #   scores = (q @ k^T) * scale + mask  (causal mask = -inf above diagonal)
        #   weights = softmax(scores)
        #   out = weights @ v
        # The trailing comments show each op's class so you can match it to `kinds`.
        # NOTE single head: head_size hs == C here (see file header for why).
        # k must be TRANSPOSED for q@k^T; we transpose in f32 then cast (the GPU's
        # transposed f16 load is unsupported -- see transpose_f32 above).
        # q,k,v projections: (T,C)@(C,hs) -> (T,hs) f32, then cast to f16.
        q = self.cast_f16(self.matmul(x16, wq, T, hs), T, hs)   # mm, ew
        k = self.matmul(x16, wk, T, hs)                          # mm (f32)
        kt = self.cast_f16(self.transpose_f32(k, T, hs), hs, T)  # ew(transpose f32), ew(cast) -> (hs,T) f16
        v = self.cast_f16(self.matmul(x16, wv, T, hs), T, hs)   # mm, ew
        # scores = q @ k^T  (T,T) f32
        scores = self.matmul(q, kt, T, T)                        # mm
        masked = self.scale_mask(scores, mask, scale, T, T)      # ew
        weights = self.softmax(masked, T, T)                     # sm (f32)
        w16 = self.cast_f16(weights, T, T)                       # ew
        out = self.matmul(w16, v, T, hs)                         # mm (T,hs)=(T,C) f32
        return out


# ---------------------------------------------------------------------------
# PAYLOAD ASSEMBLY -- wire the Builder ops into a complete MLIR function.
# Each build_*_payload creates one `func.func` (the "payload") whose arguments
# are the input + all weights (as device memrefs) and whose body is the op graph.
# `emit_buf_to_tensor` turns a memref argument into a tensor the ops can read;
# `func_cif` makes the function callable from C/the Runner. Returns (module,
# kinds) where `kinds` drives the schedule. The variants share the same Builder
# ops, just composed into bigger and bigger pieces (ffn < attn < block < gpt).
# ---------------------------------------------------------------------------
def build_ffn_payload(func_name, T, C, hidden, eps=1e-5):
    """FFN sub-layer: x -> ln(x) -> ffn1(mm)+bias+relu -> ffn2(mm)+bias -> x + ."""
    f32, f16 = F32(), F16()
    mod = ir.Module.create()
    x_t = ir.MemRefType.get((T, C), f32)
    g_t = ir.MemRefType.get((C,), f32)
    w1_t = ir.MemRefType.get((C, hidden), f16)
    b1_t = ir.MemRefType.get((hidden,), f32)
    w2_t = ir.MemRefType.get((hidden, C), f16)
    b2_t = ir.MemRefType.get((C,), f32)
    out_t = ir.MemRefType.get((T, C), f32)
    from lighthouse.utils.mlir import func_cif
    bld = Builder(T)
    with ir.InsertionPoint(mod.body):
        @func_cif(out_t, x_t, g_t, g_t, w1_t, b1_t, w2_t, b2_t, name=func_name)
        def payload(output, x_arg, gamma_arg, beta_arg, w1_arg, b1_arg, w2_arg, b2_arg):
            emit_buf_to_tensor(output, restrict=True, writable=True)
            x = emit_buf_to_tensor(x_arg, restrict=True)
            gamma = emit_buf_to_tensor(gamma_arg, restrict=True)
            beta = emit_buf_to_tensor(beta_arg, restrict=True)
            w1 = emit_buf_to_tensor(w1_arg, restrict=True)
            b1 = emit_buf_to_tensor(b1_arg, restrict=True)
            w2 = emit_buf_to_tensor(w2_arg, restrict=True)
            b2 = emit_buf_to_tensor(b2_arg, restrict=True)

            ln = bld.layernorm(x, gamma, beta, T, C, eps)      # f32
            ln16 = bld.cast_f16(ln, T, C)                      # f16 for mm
            h = bld.matmul(ln16, w1, T, hidden)                # (T,hidden) f32
            h = bld.bias(h, b1, T, hidden, relu=True)          # +b1, relu
            h16 = bld.cast_f16(h, T, hidden)                   # f16 for mm
            o = bld.matmul(h16, w2, T, C)                      # (T,C) f32
            o = bld.bias(o, b2, T, C, relu=False)              # +b2
            bld.add(x, o, T, C, out_buf=output)                # residual -> output
            for b in bld.to_dealloc:
                gpu.dealloc(None, [], b)
        emit_gpu_util_funcs(f32, rank=2)
        emit_gpu_util_funcs(f32, rank=1)
        emit_gpu_util_funcs(f16, rank=2)
    return mod, bld.kinds


def build_attn_payload(func_name, T, C, eps=1e-5):
    """Attention sublayer only: a = x + proj(attn(ln1(x))). Isolates the attn path."""
    hs = C
    f32, f16 = F32(), F16()
    mod = ir.Module.create()
    x_t = ir.MemRefType.get((T, C), f32)
    g_t = ir.MemRefType.get((C,), f32)
    wqkv_t = ir.MemRefType.get((C, hs), f16)
    wproj_t = ir.MemRefType.get((hs, C), f16)
    bproj_t = ir.MemRefType.get((C,), f32)
    mask_t = ir.MemRefType.get((T, T), f32)
    out_t = ir.MemRefType.get((T, C), f32)
    scale = 1.0 / (hs ** 0.5)
    from lighthouse.utils.mlir import func_cif
    bld = Builder(T)
    with ir.InsertionPoint(mod.body):
        @func_cif(out_t, x_t, g_t, g_t, wqkv_t, wqkv_t, wqkv_t, wproj_t, bproj_t, mask_t, name=func_name)
        def payload(output, x_arg, g1_arg, b1n_arg, wq_arg, wk_arg, wv_arg, wp_arg, bp_arg, mask_arg):
            emit_buf_to_tensor(output, restrict=True, writable=True)
            x = emit_buf_to_tensor(x_arg, restrict=True)
            g1 = emit_buf_to_tensor(g1_arg, restrict=True); b1n = emit_buf_to_tensor(b1n_arg, restrict=True)
            wq = emit_buf_to_tensor(wq_arg, restrict=True); wk = emit_buf_to_tensor(wk_arg, restrict=True)
            wv = emit_buf_to_tensor(wv_arg, restrict=True)
            wp = emit_buf_to_tensor(wp_arg, restrict=True); bp = emit_buf_to_tensor(bp_arg, restrict=True)
            mask = emit_buf_to_tensor(mask_arg, restrict=True)
            ln1 = bld.layernorm(x, g1, b1n, T, C, eps)
            ln1_16 = bld.cast_f16(ln1, T, C)
            attn = bld.attention(ln1_16, wq, wk, wv, mask, scale, T, C, hs)
            attn16 = bld.cast_f16(attn, T, C)
            proj = bld.matmul(attn16, wp, T, C)
            proj = bld.bias(proj, bp, T, C, relu=False)
            bld.add(x, proj, T, C, out_buf=output)
            for b in bld.to_dealloc:
                gpu.dealloc(None, [], b)
        emit_gpu_util_funcs(f32, rank=2)
        emit_gpu_util_funcs(f32, rank=1)
        emit_gpu_util_funcs(f16, rank=2)
    return mod, bld.kinds


def numpy_ref_attn(x, g1, b1n, Wq, Wk, Wv, Wp, bp, mask, hs, eps=1e-5):
    scale = 1.0 / (hs ** 0.5)
    ln1 = _f16(_ln(x, g1, b1n, eps))
    q = _f16(ln1 @ Wq.astype(np.float32)); k = _f16(ln1 @ Wk.astype(np.float32)); v = _f16(ln1 @ Wv.astype(np.float32))
    scores = (q @ k.T) * scale + mask
    scores = scores - scores.max(-1, keepdims=True)
    e = np.exp(scores); w = e / e.sum(-1, keepdims=True)
    attn = _f16(w) @ v
    proj = _f16(attn) @ Wp.astype(np.float32) + bp
    return x + proj


def _emit_block(bld, x, w, mask, T, C, hidden, hs, eps, out_buf=None):
    """Emit ONE gpt.py Block on Builder `bld`, reading input tensor `x`, returning
    the output tensor (or writing into out_buf if given -> returns None).
    `w` is a dict of weight tensors: g1,b1n, wq,wk,wv, wp,bp, g2,b2n, w1,bb1,w2,bb2.
        a = x + proj(attn(ln1(x)))      # attention sublayer + residual
        y = a + ffn(ln2(a))             # FFN sublayer + residual
    """
    scale = 1.0 / (hs ** 0.5)
    # ---- attention sublayer: a = x + proj(attn(ln1(x))) ----
    ln1 = bld.layernorm(x, w["g1"], w["b1n"], T, C, eps)
    ln1_16 = bld.cast_f16(ln1, T, C)
    attn = bld.attention(ln1_16, w["wq"], w["wk"], w["wv"], mask, scale, T, C, hs)
    attn16 = bld.cast_f16(attn, T, C)
    proj = bld.matmul(attn16, w["wp"], T, C)
    proj = bld.bias(proj, w["bp"], T, C, relu=False)
    a = bld.add(x, proj, T, C)
    # ---- FFN sublayer: y = a + ffn(ln2(a)) ----
    ln2 = bld.layernorm(a, w["g2"], w["b2n"], T, C, eps)
    ln2_16 = bld.cast_f16(ln2, T, C)
    h = bld.matmul(ln2_16, w["w1"], T, hidden)
    h = bld.bias(h, w["bb1"], T, hidden, relu=True)
    h16 = bld.cast_f16(h, T, hidden)
    o = bld.matmul(h16, w["w2"], T, C)
    o = bld.bias(o, w["bb2"], T, C, relu=False)
    return bld.add(a, o, T, C, out_buf=out_buf)


def build_block_payload(func_name, T, C, hidden, n_head=1, eps=1e-5):
    """Full gpt.py Block (single head, hs=C):
        a = x + proj(attn(ln1(x)))      # attention sublayer + residual
        y = a + ffn(ln2(a))             # FFN sublayer + residual
    attn = single-head: q/k/v = Linear(C,hs) no bias; core; then proj Linear(hs,C)+bias.
    """
    assert n_head == 1 and "single-head (hs=C) for now"
    hs = C
    f32, f16 = F32(), F16()
    mod = ir.Module.create()
    x_t = ir.MemRefType.get((T, C), f32)
    g_t = ir.MemRefType.get((C,), f32)
    wqkv_t = ir.MemRefType.get((C, hs), f16)        # Wq,Wk,Wv
    wproj_t = ir.MemRefType.get((hs, C), f16)       # attn output proj
    bproj_t = ir.MemRefType.get((C,), f32)
    w1_t = ir.MemRefType.get((C, hidden), f16)
    b1_t = ir.MemRefType.get((hidden,), f32)
    w2_t = ir.MemRefType.get((hidden, C), f16)
    b2_t = ir.MemRefType.get((C,), f32)
    mask_t = ir.MemRefType.get((T, T), f32)
    out_t = ir.MemRefType.get((T, C), f32)
    scale = 1.0 / (hs ** 0.5)
    from lighthouse.utils.mlir import func_cif
    bld = Builder(T)
    with ir.InsertionPoint(mod.body):
        @func_cif(out_t, x_t, g_t, g_t,           # output, x, ln1_gamma, ln1_beta
                  wqkv_t, wqkv_t, wqkv_t,          # Wq, Wk, Wv
                  wproj_t, bproj_t,                # attn proj W, b
                  g_t, g_t,                        # ln2_gamma, ln2_beta
                  w1_t, b1_t, w2_t, b2_t,          # ffn
                  mask_t, name=func_name)
        def payload(output, x_arg, g1_arg, b1n_arg, wq_arg, wk_arg, wv_arg,
                    wp_arg, bp_arg, g2_arg, b2n_arg, w1_arg, bb1_arg, w2_arg, bb2_arg, mask_arg):
            emit_buf_to_tensor(output, restrict=True, writable=True)
            x = emit_buf_to_tensor(x_arg, restrict=True)
            g1 = emit_buf_to_tensor(g1_arg, restrict=True); b1n = emit_buf_to_tensor(b1n_arg, restrict=True)
            wq = emit_buf_to_tensor(wq_arg, restrict=True); wk = emit_buf_to_tensor(wk_arg, restrict=True)
            wv = emit_buf_to_tensor(wv_arg, restrict=True)
            wp = emit_buf_to_tensor(wp_arg, restrict=True); bp = emit_buf_to_tensor(bp_arg, restrict=True)
            g2 = emit_buf_to_tensor(g2_arg, restrict=True); b2n = emit_buf_to_tensor(b2n_arg, restrict=True)
            w1 = emit_buf_to_tensor(w1_arg, restrict=True); bb1 = emit_buf_to_tensor(bb1_arg, restrict=True)
            w2 = emit_buf_to_tensor(w2_arg, restrict=True); bb2 = emit_buf_to_tensor(bb2_arg, restrict=True)
            mask = emit_buf_to_tensor(mask_arg, restrict=True)
            w = dict(g1=g1, b1n=b1n, wq=wq, wk=wk, wv=wv, wp=wp, bp=bp,
                     g2=g2, b2n=b2n, w1=w1, bb1=bb1, w2=w2, bb2=bb2)
            _emit_block(bld, x, w, mask, T, C, hidden, hs, eps, out_buf=output)
            for b in bld.to_dealloc:
                gpu.dealloc(None, [], b)
        emit_gpu_util_funcs(f32, rank=2)
        emit_gpu_util_funcs(f32, rank=1)
        emit_gpu_util_funcs(f16, rank=2)
    return mod, bld.kinds


def build_gpt_payload(func_name, T, C, hidden, vocab, n_layer, eps=1e-5):
    """Full gpt.py forward as ONE module (embeddings done host-side):
        x (T,C) f32  [= token_emb + pos_emb, computed on host]
        -> n_layer x Block
        -> ln_f (final layernorm)
        -> lm_head: (T,C) @ (C,vocab) + bias  -> logits (T,vocab) f32

    Per-layer weights are passed as a flat arg list (in layer order); then ln_f
    gamma/beta, lm_head W (f16) + bias, and the causal mask.
    """
    hs = C
    f32, f16 = F32(), F16()
    mod = ir.Module.create()
    x_t = ir.MemRefType.get((T, C), f32)
    g_t = ir.MemRefType.get((C,), f32)
    wqkv_t = ir.MemRefType.get((C, hs), f16)
    wproj_t = ir.MemRefType.get((hs, C), f16)
    bvec_t = ir.MemRefType.get((C,), f32)
    w1_t = ir.MemRefType.get((C, hidden), f16)
    b1_t = ir.MemRefType.get((hidden,), f32)
    w2_t = ir.MemRefType.get((hidden, C), f16)
    lmw_t = ir.MemRefType.get((C, vocab), f16)
    lmb_t = ir.MemRefType.get((vocab,), f32)
    mask_t = ir.MemRefType.get((T, T), f32)
    out_t = ir.MemRefType.get((T, vocab), f32)
    # per-layer arg types: g1,b1n, wq,wk,wv, wp,bp, g2,b2n, w1,bb1,w2,bb2 (13)
    per_layer = [g_t, g_t, wqkv_t, wqkv_t, wqkv_t, wproj_t, bvec_t,
                 g_t, g_t, w1_t, b1_t, w2_t, bvec_t]
    from lighthouse.utils.mlir import func_cif
    fargs = [out_t, x_t]
    for _ in range(n_layer):
        fargs += per_layer
    fargs += [g_t, g_t, lmw_t, lmb_t, mask_t]   # ln_f gamma/beta, lm_head W,b, mask
    bld = Builder(T)
    with ir.InsertionPoint(mod.body):
        @func_cif(*fargs, name=func_name)
        def payload(*args):
            output = args[0]
            emit_buf_to_tensor(output, restrict=True, writable=True)
            x = emit_buf_to_tensor(args[1], restrict=True)
            idx = 2
            layer_w = []
            for _ in range(n_layer):
                keys = ["g1", "b1n", "wq", "wk", "wv", "wp", "bp",
                        "g2", "b2n", "w1", "bb1", "w2", "bb2"]
                w = {k: emit_buf_to_tensor(args[idx + i], restrict=True)
                     for i, k in enumerate(keys)}
                idx += len(keys)
                layer_w.append(w)
            gf_g = emit_buf_to_tensor(args[idx], restrict=True); idx += 1
            gf_b = emit_buf_to_tensor(args[idx], restrict=True); idx += 1
            lmw = emit_buf_to_tensor(args[idx], restrict=True); idx += 1
            lmb = emit_buf_to_tensor(args[idx], restrict=True); idx += 1
            mask = emit_buf_to_tensor(args[idx], restrict=True); idx += 1

            h = x
            for w in layer_w:
                h = _emit_block(bld, h, w, mask, T, C, hidden, hs, eps)
            # final layernorm
            hf = bld.layernorm(h, gf_g, gf_b, T, C, eps)
            # lm_head: (T,C) @ (C,vocab) + bias -> logits (T,vocab)
            hf16 = bld.cast_f16(hf, T, C)
            logits = bld.matmul(hf16, lmw, T, vocab)
            bld.bias(logits, lmb, T, vocab, relu=False, out_buf=output)
            for b in bld.to_dealloc:
                gpu.dealloc(None, [], b)
        emit_gpu_util_funcs(f32, rank=2)
        emit_gpu_util_funcs(f32, rank=1)
        emit_gpu_util_funcs(f16, rank=2)
    return mod, bld.kinds


# =============================================================================
# STAGE 2 -- SCHEDULE: describe HOW to lower the payload to the GPU
# =============================================================================
# A "schedule" here is itself an MLIR module written in the TRANSFORM dialect: a
# little program of rewrite ops that the transform interpreter runs over the
# payload module. It does NOT compute anything; it REWRITES the payload from
# high-level linalg ops down to GPU (XeGPU) kernels.
#
# We can't reuse the repo's per-op schedules (layer_norm_schedule, mlp_schedule,
# softmax_schedule) directly, because each assumes the module contains ONLY its
# op. Our module is mixed (matmul + layernorm + softmax + elementwise), so we
# build ONE COMBINED schedule that handles all op classes. The strategy:
#
#   (a) TILE each op into its own parallel loop nest (`scf.forall` = the GPU
#       work-group grid). Different op classes tile differently:
#         - matmul   -> `_tile_one_matmul`  (work-group tile + k-loop tile; the
#                        DPAS tile sizes come from `mm_params`)
#         - layernorm-> `_tile_one_layernorm` (tile rows, fuse the 2 reductions +
#                        2 zero-fills into the loop)
#         - softmax  -> `_tile_softmax`     (decompose into max/exp/sum/div, tile)
#         - elementwise -> a single `structured_tile_using_forall` over rows
#   (b) SHARED TAIL (same for every kernel): vectorize -> bufferize (tensors ->
#       memrefs) -> convert the forall grids to `gpu.launch` -> OUTLINE each into
#       its own `gpu.module`/`gpu.func` kernel -> attach the XeVM target.
#   (c) ANNOTATE each kernel with XeGPU layout attributes (how data maps to
#       sub-groups / DPAS tiles).
#
# `kinds` (from the Builder) tells the schedule the class and order of every
# kernel, so steps (a) and (c) can treat each one correctly. See memory.md
# parts 6-10 for the subtle correctness rules baked into the helpers below.
# =============================================================================
def _tile_one_matmul(matmul_op, anytype, mm_params):
    """Tile ONE matmul for DPAS: a work-group `forall` tile (wg_m x wg_n) with any
    elementwise consumer fused in, then an inner reduction (k) loop. Tile sizes
    come from `mm_params` (chosen by xegpu_parameter_selector for the GPU)."""
    wg_tile = [mm_params["wg_m"], mm_params["wg_n"]]
    consumers = transform_ext.get_tileable_consumers(matmul_op)
    leaf = transform_ext.extract_handle(consumers, -1)
    _, [wg_loop], _ = lh_transform.tile(
        leaf, tile_sizes=wg_tile, fuse_producers=True, use_forall=True, apply_cleanup=False)
    wg_matmul = match(wg_loop, ops={"linalg.matmul"})
    lh_transform.tile(wg_matmul, tile_sizes=[0, 0, mm_params["k_tile"]])


def _tile_softmax(sm_forall_target, anytype, wg_rows, rss):
    """Decompose + tile ONE softmax that has already been wrapped in a row `forall`.

    linalg.softmax expands into 5 ops (max-reduce, subtract+exp, sum-reduce,
    reciprocal, divide). We tile the divide, fuse the exp into it, then tile the
    sum and max reductions -- all SCOPED to this softmax's forall region so the
    matches never collide with other ops elsewhere in the (large) module.
    `rss` = reduction step size (how finely the row reductions are tiled)."""
    sm_to_decomp = structured.structured_match(anytype, sm_forall_target, ops=["linalg.softmax"])
    structured.structured_decompose_interface(anytype, sm_to_decomp)
    sm_ops = match_and_split(sm_forall_target, ops={"linalg.generic", "linalg.fill"}, nhandles=6)
    max_red, max_exp, sum_red, div = sm_ops[1], sm_ops[2], sm_ops[4], sm_ops[5]
    _, div_loop = structured.TileUsingForOp(div, sizes=[0, rss]).results
    structured.structured_fuse_into_containing_op(
        anytype, anytype, producer_op=max_exp, containing_op=div_loop)
    _, _, _, sum_loop = structured.structured_tile_reduction_using_for(
        [anytype], anytype, anytype, anytype, target=sum_red, tile_sizes=[0, rss])
    sm_gens = match_and_split(sm_forall_target, ops={"linalg.generic"}, nhandles=5)
    structured.structured_fuse_into_containing_op(
        anytype, anytype, producer_op=sm_gens[1], containing_op=sum_loop)
    structured.structured_tile_reduction_using_for(
        [anytype], anytype, anytype, anytype, target=sm_gens[0], tile_sizes=[0, rss])


def build_combined_schedule(mm_params, sm_params, kinds, stop_at_stage=""):
    """Build the transform-dialect schedule module for a payload with op classes
    `kinds`. Counts how many of each class there are, then delegates to `_bundle`
    (wrapped in transform boilerplate). `stop_at_stage` lets callers halt early
    for debugging (--dump <stage>)."""
    nkernels = len(kinds)
    n_mm = kinds.count("mm")
    n_ln = kinds.count("ln")
    n_sm = kinds.count("sm")
    n_ew = kinds.count("ew")
    with schedule_boilerplate() as (schedule, named_seq):
        anytype = transform.AnyOpType.get()
        func0 = match(named_seq.bodyTarget, ops={"func.func"})
        mod = transform.get_parent_op(anytype, func0, op_name="builtin.module", deduplicate=True)
        try:
            _bundle(mod, mm_params, sm_params, kinds, n_mm, n_ln, n_sm, n_ew, stop_at_stage)
        except PipelineInterrupt:
            pass
        finally:
            transform.yield_()
    return schedule


def _bundle(mod, mm_params, sm_params, kinds, n_mm, n_ln, n_sm, n_ew, stop_at_stage=""):
    """THE PASS ORCHESTRATOR -- emits the actual sequence of transform ops.

    Runs in 3 phases over the whole payload module:
      TILE   -- tile every op into a GPU work-group `forall` (per op class)
      SHARED TAIL -- vectorize, bufferize, forall->gpu.launch, outline kernels,
                     attach the XeVM target, lower vector ops to XeGPU
      ANNOTATE -- attach XeGPU sub-group/DPAS layout to each kernel
    `stop_at_stage` raises PipelineInterrupt to halt after a phase (for --dump).
    Reading the inline comments here is the best way to understand "which part of
    the code schedules the passes" -- it is this function, top to bottom."""
    anytype = transform.AnyOpType.get()
    rss = sm_params["reduction_step_size"]
    wg_rows = sm_params["wg_rows"]
    nkernels = len(kinds)

    if stop_at_stage == "initial":
        raise PipelineInterrupt()

    # ===== TILE each op-class into its own forall =====
    # KEY PROBLEM: match(linalg.generic) is NOT scoped -- once an op is tiled into
    # a forall, its generic is STILL matched (it's just nested), so we can't
    # re-match "the remaining bare generics" by count. SOLUTION: split ALL generic
    # handles ONCE up front (their build order is deterministic), then tile each
    # using its preserved handle. A handle to op X stays valid across tiling of
    # OTHER ops. We tile the simple EW generics first (no fusion/cleanup, so ln
    # handles survive), then the layernorms (which fuse + cleanup).
    #
    # Generic build order: each layernorm contributes [mean, var, normalize] (3),
    # in block build order; each elementwise contributes 1. We reconstruct the
    # per-op handle slices from `kinds`.
    ngen_total = 3 * n_ln + n_ew
    gen_handles = transform.split_handle((anytype,) * ngen_total, match(mod, ops={"linalg.generic"}))
    # Walk kinds to assign generic handles to ops.
    ln_slices, ew_handles = [], []
    gi = 0
    for k in kinds:
        if k == "ln":
            ln_slices.append((gen_handles[gi], gen_handles[gi + 1], gen_handles[gi + 2]))
            gi += 3
        elif k == "ew":
            ew_handles.append(gen_handles[gi]); gi += 1
        # mm / sm contribute no bare linalg.generic here

    # 1) Tile layernorms FIRST, using preserved (mean,var,normalize) handles.
    #    Doing this BEFORE EW/matmul tiling keeps the bare linalg.fill pool exactly
    #    predictable: 2*(untiled lns) + n_mm (matmul accumulator fills). EW tiling
    #    can introduce its own init fills, so we must finish ln fill-fusion first.
    for i, (mean_red, var_red, normalize) in enumerate(ln_slices):
        ln_untiled = n_ln - i
        _tile_one_layernorm(mod, anytype, wg_rows, rss, mean_red, var_red, normalize,
                            ln_untiled, n_mm, sm_params["T"])

    # 2) Tile EW generics into own foralls (handles preserved across ln tiling).
    for eg in ew_handles:
        structured.structured_tile_using_forall(
            anytype, anytype, eg, num_threads=[], tile_sizes=[], static_tile_sizes=(wg_rows,))

    # 3) Softmax(es). Split the handle so each is tiled as a SINGLE handle --
    #    structured_tile_using_forall requires exactly one target, but with >1
    #    block there are multiple linalg.softmax ops, so a bare match would return
    #    them all and fail at apply ("requires exactly one target handle").
    if n_sm:
        sms = match_and_split(mod, ops={"linalg.softmax"}, nhandles=n_sm)
        for sm in sms:
            _, sm_forall = structured.structured_tile_using_forall(
                anytype, anytype, sm, num_threads=[], tile_sizes=[], static_tile_sizes=(wg_rows,))
            _tile_softmax(sm_forall, anytype, wg_rows, rss)

    # 4) Matmuls (their EW producers already wrapped in foralls)
    mms = match_and_split(mod, ops={"linalg.matmul"}, nhandles=n_mm)
    for mm in mms:
        _tile_one_matmul(mm, anytype, mm_params)

    func = match(mod, ops={"func.func"})
    lh_transform.cleanup(func)
    if stop_at_stage == "tiled":
        raise PipelineInterrupt()

    # ===== SHARED TAIL =====
    func = structured.structured_vectorize_children_and_apply_patterns(
        anytype, func, fold_type_extensions_into_contract=True)
    lh_transform.cleanup(func)
    if stop_at_stage == "vectorized":
        raise PipelineInterrupt()

    mod = apply_registered_pass(mod, "eliminate-empty-tensors")
    mod = transform_bufferization.OneShotBufferizeOp(
        mod, allow_return_allocs_from_loops=True, bufferize_function_boundaries=True,
        function_boundary_type_conversion=LayoutMapOption.IdentityLayoutMap).result
    mod = apply_registered_pass(mod, "fold-memref-alias-ops")
    transform.apply_cse(mod)
    canonicalize(mod)

    func = match(mod, ops={"func.func"})
    func = apply_registered_pass(func, "promote-buffers-to-stack",
        options={"max-alloc-size-in-bytes": "8192", "max-rank-of-allocated-memref": "2"})
    if stop_at_stage == "bufferized":
        raise PipelineInterrupt()

    wg_loops = match_and_split(mod, ops={"scf.forall"}, nhandles=nkernels)
    for wg_loop in wg_loops:
        loop.loop_forall_to_parallel([anytype], wg_loop)
    func = match(mod, ops={"func.func"})
    func = apply_registered_pass(func, "gpu-map-parallel-loops")
    func = apply_registered_pass(func, "convert-parallel-loops-to-gpu")
    func = apply_registered_pass(func, "lower-affine")
    transform.apply_cse(func)
    canonicalize(func)

    # launch threads per kernel, in IR (build) order = `kinds`.
    launches = match_and_split(mod, ops={"gpu.launch"}, nhandles=nkernels)
    mm_threads = (mm_params["wg_m"] // mm_params["sg_m"]) * (mm_params["wg_n"] // mm_params["sg_n"]) * 16
    sm_threads = (sm_params["wg_rows"] // sm_params["sg_rows"]) * sm_params["subgroup_size"]
    for launch, kind in zip(launches, kinds):
        nt = mm_threads if kind == "mm" else sm_threads
        xegpu.set_gpu_launch_threads(launch, threads=[nt, 1, 1])

    func = apply_registered_pass(func, "lower-affine")
    canonicalize(func)
    func = apply_registered_pass(func, "gpu-launch-sink-index-computations")
    mod = apply_registered_pass(mod, "gpu-kernel-outlining")
    transform.apply_cse(mod)
    if stop_at_stage == "gpu-outlining":
        raise PipelineInterrupt()

    mod = apply_registered_pass(mod, "xevm-attach-target", options={"O": "3", "chip": "bmg"})

    # per-gpu.module convert-vector-to-xegpu. ONLY ln/sm need SLM allocas (their
    # cross-lane reductions go through shared local memory -> store_matrix). The
    # ew kernels (cast/bias/residual) are pure row-parallel: forcing their allocas
    # to SLM creates store_matrix paths with no valid layout -> "Expected layout
    # for non-1D vectors". So SLM-ify ln/sm only; leave ew (and mm) as store_nd.
    gpu_mods = match_and_split(mod, ops={"gpu.module"}, nhandles=nkernels)
    sg_layout = [sm_params["sg_rows"], 1]
    sg_data = [sm_params["sg_rows"], rss]
    for gm, kind in zip(gpu_mods, kinds):
        gf = match(gm, ops={"gpu.func"})
        if kind in ("ln", "sm"):
            allocas = match(gf, ops={"memref.alloca"})
            transform_ext.update_address_space(allocas, address_space=3)
        gf = apply_registered_pass(gf, "convert-vector-to-xegpu")
        transform.apply_cse(gf)
    transform.apply_cse(mod)
    canonicalize(mod)
    if stop_at_stage == "xegpu-initial":
        raise PipelineInterrupt()

    # ===== PER-KERNEL ANNOTATION =====
    #   mm -> full mlp wg annotation
    #   ln -> store_nd (1) + store_matrix (the SLM reduction stores)
    #   sm -> store_nd (1) + store_matrix (4)
    #   ew -> store_nd (1) only (pure row-parallel, no SLM)
    gpu_mods = match_and_split(mod, ops={"gpu.module"}, nhandles=nkernels)
    for gm, kind in zip(gpu_mods, kinds):
        gf = match(gm, ops={"gpu.func"})
        if kind == "mm":
            xegpu_wg_annotation_for_mlp_layer(gf, **mm_params)
        else:
            # ln/sm/ew: anchor-layout their store_nd, and (ln/sm) their SLM
            # store_matrix. Pass the whole match handle to set_anchor_layout (it
            # accepts a multi-handle) -- avoids guessing exact store counts.
            xegpu.set_anchor_layout(
                match(gf, ops={"xegpu.store_nd"}), sg_layout=sg_layout, sg_data=sg_data)
            if kind in ("ln", "sm"):
                xegpu.set_anchor_layout(
                    match(gf, ops={"xegpu.store_matrix"}), sg_layout=sg_layout, sg_data=sg_data)
    if stop_at_stage == "xegpu-wg":
        raise PipelineInterrupt()
    return mod


def _tile_one_layernorm(mod, anytype, wg_rows, rss, mean_red, var_red, normalize, ln_untiled, n_mm, T_ROWS):
    """Tile ONE layernorm into its own forall, using PRESERVED handles to its 3
    generics (mean_red, var_red, normalize). Handles to other ops stay valid.

    The 2 accumulator fills are matched by their producer relationship: we match
    all fills and fuse the ones that feed this ln. To avoid touching matmul fills,
    we rely on fuse_into_containing pulling only genuine producers of the forall.
    """
    _, ln_forall = structured.structured_tile_using_forall(
        anytype, anytype, normalize, num_threads=[], tile_sizes=[], static_tile_sizes=(wg_rows,))
    _, ln_forall = structured.structured_fuse_into_containing_op(
        anytype, anytype, producer_op=var_red, containing_op=ln_forall)
    _, ln_forall = structured.structured_fuse_into_containing_op(
        anytype, anytype, producer_op=mean_red, containing_op=ln_forall)
    # Fuse this ln's 2 accumulator fills into the forall. Robustly select ONLY the
    # layernorm accumulator fills (NOT matmul fills) by filtering on result type:
    # ln accumulators are rank-1 tensor<T x f32>; matmul accumulators are rank-2.
    # This avoids fragile positional counting across the whole block. There are
    # 2*ln_untiled such rank-1 fills (this ln + other untiled lns); this ln's are
    # the FIRST 2 in IR order.
    ln_func = transform.get_parent_op(anytype, ln_forall, op_name="func.func", deduplicate=True)
    reduce_t = ir.RankedTensorType.get((T_ROWS,), F32())  # ln accumulator type (T,)
    fill_match = structured.MatchOp(
        anytype, ln_func, ops=["linalg.fill"], filter_result_type=reduce_t)
    n_ln_fills = 2 * ln_untiled
    fills = transform.split_handle((anytype,) * n_ln_fills, fill_match.results[0])
    _, ln_forall = structured.structured_fuse_into_containing_op(
        anytype, anytype, producer_op=fills[1], containing_op=ln_forall)
    _, ln_forall = structured.structured_fuse_into_containing_op(
        anytype, anytype, producer_op=fills[0], containing_op=ln_forall)
    # Fusion leaves the full-size original fills DEAD at func scope (fusion only
    # slices a copy inside the forall). They must be removed or the next ln finds
    # too many. Use canonicalize (which does DCE of the dead originals) at FUNC
    # scope, but NEVER apply_cse at func scope -- CSE would merge the identical
    # live zero-fills ACROSS layernorms. CSE the duplicate GENERICS inside the
    # forall only (scoped), so the re-match below finds exactly 3.
    transform.apply_cse(ln_forall)
    canonicalize(ln_func)
    # tile this ln's reductions+normalize (now inside the forall). Re-match the
    # 3 generics INSIDE the forall (scoped to ln_forall, so unambiguous: exactly 3).
    g2 = match_and_split(ln_forall, ops={"linalg.generic"}, nhandles=3)
    structured.TileUsingForOp(g2[2], sizes=[0, rss])
    structured.structured_tile_reduction_using_for(
        [anytype], anytype, anytype, anytype, target=g2[1], tile_sizes=[0, rss])
    structured.structured_tile_reduction_using_for(
        [anytype], anytype, anytype, anytype, target=g2[0], tile_sizes=[0, rss])
    transform.apply_cse(ln_forall)
    canonicalize(ln_forall)


# =============================================================================
# NUMPY REFERENCE -- the same math in plain numpy, to CHECK the GPU result.
# These mirror what each variant computes. `_f16` rounds through float16 to model
# the GPU's f16 matmul precision, so the comparison tolerance can be tight.
# =============================================================================
def _ln(x, gamma, beta, eps=1e-5):
    mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * gamma + beta


def _f16(a):
    # round f32 -> f16 -> f32: models the precision loss of the GPU's f16 matmul.
    return a.astype(np.float16).astype(np.float32)


def numpy_ref_ffn(x, gamma, beta, W1, b1, W2, b2, eps=1e-5):
    ln = _ln(x, gamma, beta, eps)
    h = np.maximum(_f16(ln) @ W1.astype(np.float32) + b1, 0.0)
    o = _f16(h) @ W2.astype(np.float32) + b2
    return x + o


def numpy_ref_block(x, g1, b1n, Wq, Wk, Wv, Wp, bp, g2, b2n, W1, b1, W2, b2, mask, hs, eps=1e-5):
    scale = 1.0 / (hs ** 0.5)
    # attention sublayer
    ln1 = _f16(_ln(x, g1, b1n, eps))
    q = _f16(ln1 @ Wq.astype(np.float32))
    k = _f16(ln1 @ Wk.astype(np.float32))
    v = _f16(ln1 @ Wv.astype(np.float32))
    scores = (q @ k.T) * scale + mask
    scores = scores - scores.max(-1, keepdims=True)
    e = np.exp(scores); w = e / e.sum(-1, keepdims=True)
    attn = _f16(w) @ v                                  # (T,hs)
    proj = _f16(attn) @ Wp.astype(np.float32) + bp      # (T,C)
    a = x + proj
    # ffn sublayer
    ln2 = _f16(_ln(a, g2, b2n, eps))
    h = np.maximum(_f16(ln2) @ W1.astype(np.float32) + b1, 0.0)
    o = _f16(h) @ W2.astype(np.float32) + b2
    return a + o


# =============================================================================
# STAGE 3 -- DRIVER: build payload, apply schedule, run on GPU, check result.
# =============================================================================
def main():
    """Entry point. Flags:
      --variant ffn|attn|block|gpt : which sub-model to build (default ffn)
      --gpt-layers N               : number of transformer layers for --variant gpt
      --check                      : run on the GPU and compare to the numpy reference
      --dump STAGE                 : print IR at a stage and exit, one of
                                     initial | schedule | tiled | vectorized |
                                     bufferized | gpu-outlining | xegpu-initial |
                                     xegpu-wg | final  (great for seeing each pass)

    Flow: build payload module -> build schedule module -> TransformDriver applies
    schedule to payload (lowering it to XeGPU) + xegpu_to_binary makes the GPU
    binary -> Runner JIT-runs it on the GPU -> compare to numpy_ref_*."""
    dump = None
    check = "--check" in sys.argv
    variant = "ffn"
    if "--variant" in sys.argv:
        variant = sys.argv[sys.argv.index("--variant") + 1]
    if "--dump" in sys.argv:
        dump = sys.argv[sys.argv.index("--dump") + 1]

    # n_layer=6 matches gpt.py's depth. Other dims use the proven kernel-friendly
    # shapes (all matmuls clear the DPAS gate), NOT gpt.py's exact n_embd/n_head:
    # this single-module lowering is single-head (hs=C). True n_head=6
    # (head_size=64) needs multi-head attention + a fix for the head_size=64
    # scores@v matmul (n=64 < 256 fails the DPAS gate) -- a separate milestone.
    T, C, hidden = 256, 256, 1024
    vocab, n_layer = 256, 6                # n_layer per gpt.py
    if "--gpt-layers" in sys.argv:
        n_layer = int(sys.argv[sys.argv.index("--gpt-layers") + 1])
    mm_params = dict(xegpu_parameter_selector.get_matmul_parameters(T, C, C))
    sm_params = {"wg_rows": 64, "sg_rows": 8, "subgroup_size": 16,
                 "reduction_step_size": 16, "T": T}

    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        install_dialect_module_stubs()
        if variant == "ffn":
            mod, kinds = build_ffn_payload("payload", T, C, hidden)
        elif variant == "attn":
            mod, kinds = build_attn_payload("payload", T, C)
        elif variant == "block":
            mod, kinds = build_block_payload("payload", T, C, hidden)
        elif variant == "gpt":
            mod, kinds = build_gpt_payload("payload", T, C, hidden, vocab, n_layer)
        else:
            raise SystemExit(f"variant {variant} not yet implemented")
        if dump == "initial":
            print(mod); print("KINDS:", kinds); return

        sched = build_combined_schedule(dict(mm_params), dict(sm_params), kinds,
                                        stop_at_stage=(dump or ""))
        if dump == "schedule":
            print(sched); return
        schedules = [sched]
        if not dump or dump == "final":
            schedules.append(xegpu_to_binary())
        payload = TransformDriver(schedules).apply(mod)
        if dump:
            print(payload); return
        print(f"LOWERED OK: '{variant}' to {len(kinds)} kernels in one module")

        if not check:
            return
        runner = Runner(payload, mem_manager_cls=GPUMemoryManager,
                        shared_libs=["libmlir_levelzero_runtime.so"])
        np.random.seed(0)
        out_cols = vocab if variant == "gpt" else C
        out = np.zeros((T, out_cols), np.float32)
        cb = Runner.get_gpu_argument_access_callback(out, arg_index=0)

        if variant == "ffn":
            x = np.random.randn(T, C).astype(np.float32)
            gamma = np.random.randn(C).astype(np.float32)
            beta = np.random.randn(C).astype(np.float32)
            W1 = np.random.randint(-2, 3, (C, hidden)).astype(np.float16)
            b1 = np.random.randn(hidden).astype(np.float32)
            W2 = np.random.randint(-2, 3, (hidden, C)).astype(np.float16)
            b2 = np.random.randn(C).astype(np.float32)
            host = [out, x, gamma, beta, W1, b1, W2, b2]
            runner.execute(host_input_buffers=host, payload_function_name="payload",
                           argument_access_callback=cb)
            ref = numpy_ref_ffn(x, gamma, beta, W1, b1, W2, b2)
        elif variant == "attn":
            hs = C; sc = 0.05
            x = (np.random.randn(T, C) * 0.5).astype(np.float32)
            g1 = np.ones(C, np.float32); b1n = np.zeros(C, np.float32)
            Wq = (np.random.randn(C, hs) * sc).astype(np.float16)
            Wk = (np.random.randn(C, hs) * sc).astype(np.float16)
            Wv = (np.random.randn(C, hs) * sc).astype(np.float16)
            Wp = (np.random.randn(hs, C) * sc).astype(np.float16); bp = np.zeros(C, np.float32)
            mask = np.triu(np.full((T, T), -np.inf, np.float32), k=1)
            host = [out, x, g1, b1n, Wq, Wk, Wv, Wp, bp, mask]
            runner.execute(host_input_buffers=host, payload_function_name="payload",
                           argument_access_callback=cb)
            ref = numpy_ref_attn(x, g1, b1n, Wq, Wk, Wv, Wp, bp, mask, hs)
        elif variant == "block":
            hs = C
            # Small-scale weights (gpt.py uses std~0.02 init). With ±2 integer
            # weights the 2-matmul attention + FFN(hidden=1024) compound to O(1e3)
            # activations where f16 has only ~3 sig digits -> huge abs error. Keep
            # values O(1) so f16 is accurate; scale 0.05 keeps each matmul output
            # O(1) (sqrt(256)*0.05 ~ 0.8). Weights still f16-representable.
            sc = 0.05
            x = (np.random.randn(T, C) * 0.5).astype(np.float32)
            g1 = np.ones(C, np.float32); b1n = np.zeros(C, np.float32)
            Wq = (np.random.randn(C, hs) * sc).astype(np.float16)
            Wk = (np.random.randn(C, hs) * sc).astype(np.float16)
            Wv = (np.random.randn(C, hs) * sc).astype(np.float16)
            Wp = (np.random.randn(hs, C) * sc).astype(np.float16); bp = np.zeros(C, np.float32)
            g2 = np.ones(C, np.float32); b2n = np.zeros(C, np.float32)
            W1 = (np.random.randn(C, hidden) * sc).astype(np.float16); b1 = np.zeros(hidden, np.float32)
            W2 = (np.random.randn(hidden, C) * sc).astype(np.float16); b2 = np.zeros(C, np.float32)
            mask = np.triu(np.full((T, T), -np.inf, np.float32), k=1)
            host = [out, x, g1, b1n, Wq, Wk, Wv, Wp, bp, g2, b2n, W1, b1, W2, b2, mask]
            runner.execute(host_input_buffers=host, payload_function_name="payload",
                           argument_access_callback=cb)
            ref = numpy_ref_block(x, g1, b1n, Wq, Wk, Wv, Wp, bp, g2, b2n, W1, b1, W2, b2, mask, hs)
        else:  # gpt -- full model, n_layer blocks + ln_f + lm_head
            hs = C; sc = 0.05
            mask = np.triu(np.full((T, T), -np.inf, np.float32), k=1)
            # host embeddings: simulate token+pos embedding sum as the input x.
            x = (np.random.randn(T, C) * 0.5).astype(np.float32)
            layers = []
            host = [out, x]
            for _ in range(n_layer):
                lw = dict(
                    g1=np.ones(C, np.float32), b1n=np.zeros(C, np.float32),
                    wq=(np.random.randn(C, hs) * sc).astype(np.float16),
                    wk=(np.random.randn(C, hs) * sc).astype(np.float16),
                    wv=(np.random.randn(C, hs) * sc).astype(np.float16),
                    wp=(np.random.randn(hs, C) * sc).astype(np.float16), bp=np.zeros(C, np.float32),
                    g2=np.ones(C, np.float32), b2n=np.zeros(C, np.float32),
                    w1=(np.random.randn(C, hidden) * sc).astype(np.float16), bb1=np.zeros(hidden, np.float32),
                    w2=(np.random.randn(hidden, C) * sc).astype(np.float16), bb2=np.zeros(C, np.float32))
                layers.append(lw)
                host += [lw["g1"], lw["b1n"], lw["wq"], lw["wk"], lw["wv"], lw["wp"], lw["bp"],
                         lw["g2"], lw["b2n"], lw["w1"], lw["bb1"], lw["w2"], lw["bb2"]]
            gf_g = np.ones(C, np.float32); gf_b = np.zeros(C, np.float32)
            lmw = (np.random.randn(C, vocab) * sc).astype(np.float16); lmb = np.zeros(vocab, np.float32)
            host += [gf_g, gf_b, lmw, lmb, mask]
            runner.execute(host_input_buffers=host, payload_function_name="payload",
                           argument_access_callback=cb)
            # numpy ref: n_layer blocks -> ln_f -> lm_head
            h = x
            for lw in layers:
                h = numpy_ref_block(h, lw["g1"], lw["b1n"], lw["wq"], lw["wk"], lw["wv"],
                                    lw["wp"], lw["bp"], lw["g2"], lw["b2n"], lw["w1"], lw["bb1"],
                                    lw["w2"], lw["bb2"], mask, hs)
            hf = _ln(h, gf_g, gf_b)
            ref = _f16(hf) @ lmw.astype(np.float32) + lmb

        rel = np.abs(out - ref).max() / (np.abs(ref).max() + 1e-6)
        print(f"max abs diff={np.abs(out-ref).max():.4f}  rel={rel:.6f}")
        print("PASSED" if rel < 5e-2 else "FAILED")


if __name__ == "__main__":
    main()
