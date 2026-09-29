"""PyTorch Llama-3 block, written to be exportable by torch-mlir.

This is the SAME math as the hand-written `Builder` payload in llama3_payload.py
and its numpy reference `numpy_ref_block_llama` in llama3.py -- RMSNorm, real
half-split (HF/NeoX) RoPE, grouped-query causal attention, SwiGLU FFN, and an LM
head. The point is to obtain the linalg-on-tensors payload by lowering THIS model
through torch-mlir (see llama3_torch_mlir.py) instead of emitting linalg by hand,
so a lighthouse-vs-Inductor comparison starts from the same PyTorch source.

Deliberately avoids constructs the reference model uses that torch-mlir/export
handle poorly: complex-number RoPE (view_as_complex) and the KV-cache index_copy.
This is a full-prefill, cache-free forward -- exactly what the payload computes.

Dtype: f32 throughout. The payload's f16 casts are a DPAS-hardware lowering
detail, not part of the model's math; pass model_datatype=torch.float16 at import
time to reproduce them if needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Toy config -- matches the llama3.py default self-check
# (T=256, C=256, hidden=1024, vocab=256, H=4, n_kv=2, hs=64).
T = 256
C = 256
HIDDEN = 1024
VOCAB = 256
H = 4
N_KV = 2
EPS = 1e-5


def _rms_norm(x, weight, eps):
    """x * rsqrt(mean(x^2) + eps) * weight, over the last dim.

    Written to lower on XeGPU, not just to read naturally (see
    torch_mlir_rmsnorm_gpu.py for the full reasoning):
      - `x*x` instead of `pow(x, 2)`, so it legalizes to CPU libm.
      - the reciprocal square root is spelled `y ** -0.5`, and it is the ONLY spelling
        measured to work on both paths. The three alternatives:
          `1.0 / torch.sqrt(y)` and `torch.reciprocal(torch.sqrt(y))` -- torch-mlir guards
            tensor division with a `cf.assert` ("unimplemented: tensor with zero element")
            INSIDE the generic, and the vectorizer cannot vectorize a generic containing a
            side-effecting control-flow op: "Attempted to vectorize, but failed". This was
            the blocker that stopped the whole block from running.
          `torch.rsqrt(y)` -- vectorizes fine, but `math.rsqrt` does not legalize to libm,
            so the CPU oracle dies with "Unknown function main".
        `y ** -0.5` gives `math.powf`, which legalizes to libm `powf` AND has no division to
        guard. CPU oracle 6.0e-7, GPU rel 8.3e-4.
      - `sum(-1)` WITHOUT keepdim, giving a rank-1 accumulator with indexing map
        (d0,d1)->(d0). With keepdim the map is (d0,d1)->(d0,0), which is not a permuted
        projection and linalg refuses to tile it.
      - the sum is broadcast back to x's shape BEFORE the mean/eps/rsqrt/scale math, so
        that chain is full-rank elementwise and fuses into one generic. Doing the math on
        the reduced value leaves separate rank-1 generics, because elementwise fusion
        will not fuse a reduced producer into a full-rank consumer.
      - **the norm runs in f32** while the rest of the payload is f16, and that is REQUIRED for
        the model to predict the right token, not a nicety. Measured with REAL Llama-3.2-1B
        weights, 16 layers, prompt "The capital of France is":

            norm in f16 -> rel 0.67 vs f32,  greedy next token ' a'
            norm in f32 -> rel 7.2e-4,       greedy next token ' Paris'  <- correct

        The cause is the sum of squares: `ssq` is ~1 built from 2048 terms of ~5e-4, and near
        1.0 the f16 spacing is ~1e-3, so each addition rounds away much of the term being
        added. Both references we compare against do it in f32 -- the hand payload norms in f32
        (`llama3_weights.py` loads the norm gains as f32 for that reason), and transformers'
        `LlamaRMSNorm` upcasts to f32 internally on every fp16 inference path; Inductor's F0
        lists "FP32 accumulation of sum of squares" too.
        The GAIN stays in the input dtype and is applied AFTER the cast back: `weight.float()`
        instead is a rank-1 `(C,)` cast, and a rank-1 kernel has no 2-D tensor_desc to store
        through -- it lowers to a scattered `xegpu.store` and dies with "Failed to determine
        required layout for store scatter". transformers does the same
        (`self.weight * hidden_states.to(input_dtype)`), and the gain is ~1 so f16 costs
        nothing. Cost of the f32 norm: the `extf` becomes its own kernel per norm (+1 each)."""
    xf = x.float()
    ssq = (xf * xf).sum(-1)
    ssq = ssq.unsqueeze(-1).expand_as(xf)
    return (xf * (ssq / x.shape[-1] + eps) ** -0.5).to(x.dtype) * weight


def _rope(x, cos, sin, n_heads):
    """Half-split RoPE on (T, n_heads*hs). Mirrors Builder.rope / _numpy_rope:
    out[d] = a*cos - b*sin, out[d+half] = b*cos + a*sin, where a/b are the first/
    second half of each head.

    Kept RANK-2 and CONCAT-FREE, which is what XeGPU can lower -- see
    torch_mlir_rope_gpu.py for the full story and the measurements. In short: a rank-3
    head view leaves rank-3 *buffers* that `convert-vector-to-xegpu` cannot read out of,
    and `torch.cat` becomes a `tensor.concat`, which is data movement that lands outside
    every kernel (a host access over device memory, which faults). `(T, n_heads, hs)` is
    contiguous, so this works on a `(T*n_heads, hs)` view -- one row per (token, head) --
    and assigns the two rotated halves into slices of an `empty_like`, which the schedule
    can fold onto the real destination.

    `cos`/`sin` arrive pre-expanded to `(T*n_heads, half)`, one row per (token, head);
    build them with `expand_rope_tables`. They are expanded HOST-side on purpose: doing
    it in the payload means a broadcast that materializes as its own kernel, and the
    driver computes the tables anyway.
    """
    t, d = x.shape
    hs = d // n_heads
    half = hs // 2
    v = x.reshape(t * n_heads, hs)
    lo, hi = v[:, :half], v[:, half:]
    out = torch.empty_like(v)
    out[:, :half] = lo * cos - hi * sin
    out[:, half:] = hi * cos + lo * sin
    return out.reshape(t, d)


def expand_rope_tables(cos, sin, n_heads):
    """(T, half) RoPE tables -> (T*n_heads, half), one row per (token, head).

    Host-side helper, deliberately NOT part of `forward` -- see `_rope`. Q and K need
    separate packs because GQA gives them different head counts.
    """
    t, half = cos.shape

    def rep(m):
        return m[:, None, :].expand(t, n_heads, half).reshape(t * n_heads, half)

    return rep(cos).contiguous(), rep(sin).contiguous()


# Per-block weight order. Weights are passed as forward ARGUMENTS, not nn.Parameters,
# so torch-mlir emits them as function arguments instead of baked `dense_resource`
# constants. Constants break the matmul schedule (its operand prefetch cannot read from
# a constant) and make real weight loading impossible. The order matches the hand
# driver's host-buffer order in llama3.py so the two can share a weight pack.
BLOCK_WEIGHTS = (
    "attn_norm",
    "wq",
    "wk",
    "wv",
    "wo",
    "ffn_norm",
    "w1",
    "w2",
    "w3",
)
MODEL_WEIGHTS = ("final_norm", "lm_head")


class LlamaBlock(nn.Module):
    """One grouped-query + RoPE Llama transformer block (matches _emit_block_llama).

    Holds no parameters: weights arrive as forward arguments in `BLOCK_WEIGHTS` order.
    """

    def __init__(self, C, hidden, H, n_kv, eps=EPS):
        super().__init__()
        self.C, self.H, self.n_kv, self.eps = C, H, n_kv, eps
        self.hs = C // H
        self.kv_dim = n_kv * self.hs
        self.hidden = hidden

    def _attention(self, q, k, v):
        """Grouped-query attention. q:(T,C), k/v:(T,kv_dim) -> (T,C).

        THE CAUSAL MASK IS NOT SPELLED HERE, on purpose, and that makes this payload
        non-causal ON ITS OWN. Causality is applied by the schedule instead --
        `replace_with_fused_attention(causal=True)`, driven by `FA_PARAMS["causal"]` in
        `llama3_torch_schedule.py` -- which masks future keys inside the flash loop. The
        hand payload does the same thing, for the same reasons:

          * an explicit `scores + mask` materializes a T x T tensor of -inf purely to throw
            it away inside the fused kernel, and
          * it BREAKS the schedule. The mask-construction ops interleave between QK^T and
            P@V, which splits the attention region into two groups; and the extra add
            becomes the max reduction's nearest generic ancestor, defeating the scale
            search.

        Consequence to keep in mind: the CPU oracle (`llama3_torch_mlir_check.py`) lowers
        this payload without that schedule, so it validates NON-CAUSAL math. It is checking
        that torch-mlir lowers what PyTorch computes, which it still does; it is not
        checking Llama's autoregressive semantics.

        Softmax is spelled by hand, without `keepdim`, for two separate lowering reasons --
        see `torch_mlir_attention_gpu.py`: `torch.softmax` upcasts f16 to f32 and the
        attention XeGPU layouts are f16-only, and a `keepdim` reduction gets an output map
        that is not a permuted projection, which tiling refuses.
        """
        t = q.shape[0]
        hs, H, n_kv = self.hs, self.H, self.n_kv
        n_rep = H // n_kv
        scale = 1.0 / (hs**0.5)
        # (T, H, hs) -> (H, T, hs); KV heads expanded to query heads via GQA repeat.
        # K^T is formed INSIDE the matmul expression, i.e. as the last thing before the
        # contraction. That is deliberate: the fused-attention rewrite transposes K itself
        # and subsumes an adjacent K^T, so it must see K in (n_ctx, d_head). Pre-transposing
        # K instead -- e.g. `.transpose(0,1).transpose(1,2)` before the repeat, which
        # torch-mlir folds into one permutation and hoists into its own kernel -- hands the
        # rewrite an already-transposed operand and its Q/K layouts then land on the wrong
        # loads ("'xegpu.load_nd' op TensorDesc shape is not distributable"). Note that
        # pre-transposing IS what zero-copy GQA would want (`_gqa_probe.py`), so the two
        # wishes conflict; correctness wins until GQA materialization is revisited.
        qh = q.view(t, H, hs).transpose(0, 1)
        kh = k.view(t, n_kv, hs).transpose(0, 1).repeat_interleave(n_rep, dim=0)
        vh = v.view(t, n_kv, hs).transpose(0, 1).repeat_interleave(n_rep, dim=0)
        scores = torch.matmul(qh, kh.transpose(1, 2)) * scale  # (H, T, T)
        m = torch.amax(scores, dim=-1)
        e = torch.exp(scores - m.unsqueeze(-1))
        w = e / e.sum(dim=-1).unsqueeze(-1)
        out = torch.matmul(w, vh)  # (H, T, hs)
        return out.transpose(0, 1).reshape(t, H * hs)

    def forward(
        self,
        x,
        cos_q,
        sin_q,
        cos_k,
        sin_k,
        attn_norm,
        wq,
        wk,
        wv,
        wo,
        ffn_norm,
        w1,
        w2,
        w3,
    ):
        rms1 = _rms_norm(x, attn_norm, self.eps)
        q = _rope(rms1 @ wq, cos_q, sin_q, self.H)
        k = _rope(rms1 @ wk, cos_k, sin_k, self.n_kv)
        v = rms1 @ wv
        attn = self._attention(q, k, v)
        h = x + attn @ wo
        rms2 = _rms_norm(h, ffn_norm, self.eps)
        gate = F.silu(rms2 @ w1)
        up = rms2 @ w3
        o = (gate * up) @ w2
        return h + o


class Llama3(nn.Module):
    """n_layers Llama blocks + final RMSNorm + LM head (matches numpy_ref_llama).

    Parameter-free: `forward(x, cos_q, sin_q, cos_k, sin_k, *weights)` takes the whole
    weight pack as arguments, laid out as n_layers * BLOCK_WEIGHTS followed by
    MODEL_WEIGHTS. Build a matching pack with `make_weights`. Q and K get their own RoPE
    tables because GQA gives them different head counts; build them with
    `expand_rope_tables`.
    """

    def __init__(
        self, C=C, hidden=HIDDEN, vocab=VOCAB, H=H, n_kv=N_KV, n_layers=1, eps=EPS
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [LlamaBlock(C, hidden, H, n_kv, eps) for _ in range(n_layers)]
        )
        self.dims = (C, hidden, vocab, H, n_kv)
        self.n_layers = n_layers
        self.eps = eps

    def forward(self, x, cos_q, sin_q, cos_k, sin_k, *weights):
        n = len(BLOCK_WEIGHTS)
        expected = self.n_layers * n + len(MODEL_WEIGHTS)
        if len(weights) != expected:
            raise ValueError(f"expected {expected} weight tensors, got {len(weights)}")
        h = x
        for i, blk in enumerate(self.blocks):
            h = blk(h, cos_q, sin_q, cos_k, sin_k, *weights[i * n : (i + 1) * n])
        final_norm, lm_head = weights[self.n_layers * n :]
        hf = _rms_norm(h, final_norm, self.eps)
        return hf @ lm_head


def make_weights(
    C=C,
    hidden=HIDDEN,
    vocab=VOCAB,
    H=H,
    n_kv=N_KV,
    n_layers=1,
    scale=0.02,
    dtype=torch.float32,
):
    """Build a weight pack in the order `Llama3.forward` expects."""
    hs = C // H
    kv_dim = n_kv * hs
    shapes = {
        "attn_norm": (C,),
        "wq": (C, C),
        "wk": (C, kv_dim),
        "wv": (C, kv_dim),
        "wo": (C, C),
        "ffn_norm": (C,),
        "w1": (C, hidden),
        "w2": (hidden, C),
        "w3": (C, hidden),
        "final_norm": (C,),
        "lm_head": (C, vocab),
    }

    def make(name):
        if name.endswith("norm"):  # norm weights are gains, initialized to 1
            return torch.ones(shapes[name], dtype=dtype)
        return (torch.randn(shapes[name]) * scale).to(dtype)

    pack = []
    for _ in range(n_layers):
        pack += [make(n) for n in BLOCK_WEIGHTS]
    pack += [make(n) for n in MODEL_WEIGHTS]
    return tuple(pack)


# ---- Ingress hooks used by lighthouse.ingress.torch.import_model ----
def get_init_inputs():
    """Positional args for Llama3.__init__ (single block, toy dims)."""
    return (C, HIDDEN, VOCAB, H, N_KV, 1, EPS)


def get_inputs():
    """Sample forward inputs: (x, cos_q, sin_q, cos_k, sin_k, *weights).

    The RoPE tables are per-head-count (GQA), expanded host-side; half = (C // H) // 2.
    """
    half = (C // H) // 2
    cos, sin = torch.randn(T, half), torch.randn(T, half)
    act = (
        torch.randn(T, C),
        *expand_rope_tables(cos, sin, H),
        *expand_rope_tables(cos, sin, N_KV),
    )
    return act + make_weights(C, HIDDEN, VOCAB, H, N_KV, n_layers=1)
