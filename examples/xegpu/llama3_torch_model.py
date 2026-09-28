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
    """x * rsqrt(mean(x^2) + eps) * weight, over the last dim."""
    ms = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(ms + eps) * weight


def _rope(x, cos, sin, n_heads):
    """Half-split RoPE on (T, n_heads*hs). Mirrors Builder.rope / _numpy_rope:
    out[d] = a*cos - b*sin, out[d+half] = b*cos + a*sin, where a/b are the first/
    second half of each head. cos/sin are (T, half)."""
    t, d = x.shape
    hs = d // n_heads
    half = hs // 2
    v = x.view(t, n_heads, hs)
    a = v[:, :, :half]
    b = v[:, :, half:]
    c = cos[:, None, :]
    s = sin[:, None, :]
    out1 = a * c - b * s
    out2 = b * c + a * s
    return torch.cat([out1, out2], dim=-1).reshape(t, d)


class LlamaBlock(nn.Module):
    """One grouped-query + RoPE Llama transformer block (matches _emit_block_llama)."""

    def __init__(self, C, hidden, H, n_kv, eps=EPS):
        super().__init__()
        self.C, self.H, self.n_kv, self.eps = C, H, n_kv, eps
        self.hs = C // H
        self.kv_dim = n_kv * self.hs
        # Weights kept as (in, out) so the forward uses x @ W directly, matching the
        # Builder payload's matmul orientation (no implicit Linear transpose).
        self.attn_norm = nn.Parameter(torch.ones(C))
        self.wq = nn.Parameter(torch.randn(C, C) * 0.02)
        self.wk = nn.Parameter(torch.randn(C, self.kv_dim) * 0.02)
        self.wv = nn.Parameter(torch.randn(C, self.kv_dim) * 0.02)
        self.wo = nn.Parameter(torch.randn(C, C) * 0.02)
        self.ffn_norm = nn.Parameter(torch.ones(C))
        self.w1 = nn.Parameter(torch.randn(C, hidden) * 0.02)
        self.w2 = nn.Parameter(torch.randn(hidden, C) * 0.02)
        self.w3 = nn.Parameter(torch.randn(C, hidden) * 0.02)

    def _attention(self, q, k, v):
        """Grouped-query causal attention. q:(T,C), k/v:(T,kv_dim) -> (T,C)."""
        t = q.shape[0]
        hs, H, n_kv = self.hs, self.H, self.n_kv
        n_rep = H // n_kv
        scale = 1.0 / (hs**0.5)
        # (T, H, hs) -> (H, T, hs); KV heads expanded to query heads via GQA repeat.
        qh = q.view(t, H, hs).transpose(0, 1)
        kh = k.view(t, n_kv, hs).transpose(0, 1).repeat_interleave(n_rep, dim=0)
        vh = v.view(t, n_kv, hs).transpose(0, 1).repeat_interleave(n_rep, dim=0)
        scores = torch.matmul(qh, kh.transpose(1, 2)) * scale  # (H, T, T)
        mask = torch.triu(torch.full((t, t), float("-inf"), device=q.device), diagonal=1)
        scores = scores + mask
        w = torch.softmax(scores, dim=-1)
        out = torch.matmul(w, vh)  # (H, T, hs)
        return out.transpose(0, 1).reshape(t, H * hs)

    def forward(self, x, cos, sin):
        rms1 = _rms_norm(x, self.attn_norm, self.eps)
        q = _rope(rms1 @ self.wq, cos, sin, self.H)
        k = _rope(rms1 @ self.wk, cos, sin, self.n_kv)
        v = rms1 @ self.wv
        attn = self._attention(q, k, v)
        h = x + attn @ self.wo
        rms2 = _rms_norm(h, self.ffn_norm, self.eps)
        gate = F.silu(rms2 @ self.w1)
        up = rms2 @ self.w3
        o = (gate * up) @ self.w2
        return h + o


class Llama3(nn.Module):
    """n_layers Llama blocks + final RMSNorm + LM head (matches numpy_ref_llama)."""

    def __init__(self, C=C, hidden=HIDDEN, vocab=VOCAB, H=H, n_kv=N_KV,
                 n_layers=1, eps=EPS):
        super().__init__()
        self.blocks = nn.ModuleList(
            [LlamaBlock(C, hidden, H, n_kv, eps) for _ in range(n_layers)]
        )
        self.final_norm = nn.Parameter(torch.ones(C))
        self.lm_head = nn.Parameter(torch.randn(C, vocab) * 0.02)
        self.eps = eps

    def forward(self, x, cos, sin):
        h = x
        for blk in self.blocks:
            h = blk(h, cos, sin)
        hf = _rms_norm(h, self.final_norm, self.eps)
        return hf @ self.lm_head


# ---- Ingress hooks used by lighthouse.ingress.torch.import_model ----
def get_init_inputs():
    """Positional args for Llama3.__init__ (single block, toy dims)."""
    return (C, HIDDEN, VOCAB, H, N_KV, 1, EPS)


def get_inputs():
    """Sample forward inputs: (x, cos, sin). half = (C // H) // 2."""
    half = (C // H) // 2
    return (torch.randn(T, C), torch.randn(T, half), torch.randn(T, half))
