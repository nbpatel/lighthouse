"""Load real Llama-3.2 weights from a HuggingFace safetensors checkpoint into the
numpy arrays the XeGPU payload expects.

The payload (llama3_payload.build_llama_payload) computes each Linear as `x @ w`,
so it needs weights laid out [in, out]. HuggingFace/PyTorch store Linear weights
transposed, as [out, in]. So every projection is transposed here and made
C-contiguous (an XeGPU kernel silently reads garbage from a non-contiguous numpy
view -- transposes MUST be materialized with np.ascontiguousarray).

dtype: checkpoint tensors are bf16. RMSNorm weights -> f32 (the norm runs in f32);
projection weights -> f16 (the DPAS matmul units take f16). bf16->f16 goes via f32.

lm_head is tied to the token embedding table (config tie_word_embeddings=true), so
the output weight is embed_tokens transposed; the same table (un-transposed) is the
embedding lookup for the input.

HF name                                   shape            -> payload arg
  model.embed_tokens.weight               [vocab, C]          embeddings (+ lm_head = .T)
  model.norm.weight                       [C]                 fn_w (final RMSNorm)
  layers.N.input_layernorm.weight         [C]                 an
  layers.N.self_attn.q_proj.weight        [C, C]      -> .T   wq
  layers.N.self_attn.k_proj.weight        [kv_dim, C] -> .T   wk
  layers.N.self_attn.v_proj.weight        [kv_dim, C] -> .T   wv
  layers.N.self_attn.o_proj.weight        [C, C]      -> .T   wo
  layers.N.post_attention_layernorm.weight[C]                 fn
  layers.N.mlp.gate_proj.weight           [hidden, C] -> .T   w1
  layers.N.mlp.up_proj.weight             [hidden, C] -> .T   w3
  layers.N.mlp.down_proj.weight           [C, hidden] -> .T   w2
"""

import json
import numpy as np
import ml_dtypes  # noqa: F401 -- registers the bfloat16 numpy dtype safetensors returns
from safetensors import safe_open


def _f32(t):
    return np.ascontiguousarray(t.astype(np.float32))


def _f16_T(t):
    # transpose [out, in] -> [in, out], materialize contiguous, cast to f16.
    return np.ascontiguousarray(t.astype(np.float32).T).astype(np.float16)


def load_config(model_dir):
    with open(f"{model_dir}/config.json") as f:
        return json.load(f)


def rope_tables_from_config(cfg, T):
    """Build half-split RoPE (cos, sin) tables of shape (T, head_dim/2) using the
    checkpoint's real rope_theta AND llama3 NTK-by-parts frequency rescaling.

    Llama-3.2 sets rope_theta=500000 and rope_scaling={type:llama3, factor:32,
    low_freq_factor:1, high_freq_factor:4, original_max_position_embeddings:8192}.
    The scaling rescales per-frequency inverse frequencies (NEVER a no-op, even for
    short prompts): low-frequency components (long wavelength) are divided by
    `factor`, high-frequency ones are kept, and a smooth interpolation bridges the
    two. Mirrors transformers' _compute_llama3_parameters. Returns f32 (T, hs/2).
    """
    hs = cfg["head_dim"]
    half = hs // 2
    base = cfg["rope_theta"]
    # base inverse frequencies: inv_freq[i] = base^(-2i/hs), i in [0, half)
    inv_freq = base ** (-np.arange(0, half, dtype=np.float64) * 2.0 / hs)

    sc = cfg.get("rope_scaling")
    if sc and sc.get("rope_type", sc.get("type")) == "llama3":
        factor = sc["factor"]
        low_freq_factor = sc["low_freq_factor"]
        high_freq_factor = sc["high_freq_factor"]
        old_ctx = sc["original_max_position_embeddings"]
        low_freq_wavelen = old_ctx / low_freq_factor
        high_freq_wavelen = old_ctx / high_freq_factor
        wavelen = 2.0 * np.pi / inv_freq
        # low-freq (wavelen > low_freq_wavelen): divide by factor; high-freq: keep.
        inv_freq_llama = np.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
        # medium band: smooth interpolation between the scaled and unscaled freqs.
        smooth = (old_ctx / wavelen - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        smoothed = (1 - smooth) * inv_freq_llama / factor + smooth * inv_freq_llama
        is_medium = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq = np.where(is_medium, smoothed, inv_freq_llama)

    ang = np.outer(np.arange(T, dtype=np.float64), inv_freq)  # (T, half)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def load_llama_weights(model_dir, n_layers=None):
    """Read the checkpoint and return everything the driver needs.

    Returns a dict:
      cfg        -- parsed config.json
      embeddings -- (vocab, C) f32 token-embedding table (row lookup for input x)
      layers     -- list of per-layer weight dicts (keys an,wq,wk,wv,wo,fn,w1,w2,w3)
      fn_w       -- (C,) f32 final RMSNorm weight
      lmw        -- (C, vocab) f16 output weight (tied: embed_tokens.T)

    n_layers truncates to the first N transformer blocks (for a smaller test run);
    None loads all config['num_hidden_layers'].
    """
    cfg = load_config(model_dir)
    total = cfg["num_hidden_layers"]
    n = total if n_layers is None else min(n_layers, total)

    path = f"{model_dir}/model.safetensors"
    with safe_open(path, "numpy") as f:
        embed = f.get_tensor("model.embed_tokens.weight")  # (vocab, C) bf16
        embeddings = _f32(embed)
        lmw = np.ascontiguousarray(embeddings.T).astype(np.float16)  # tied lm_head
        fn_w = _f32(f.get_tensor("model.norm.weight"))

        layers = []
        for i in range(n):
            p = f"model.layers.{i}."
            layers.append(
                dict(
                    an=_f32(f.get_tensor(p + "input_layernorm.weight")),
                    wq=_f16_T(f.get_tensor(p + "self_attn.q_proj.weight")),
                    wk=_f16_T(f.get_tensor(p + "self_attn.k_proj.weight")),
                    wv=_f16_T(f.get_tensor(p + "self_attn.v_proj.weight")),
                    wo=_f16_T(f.get_tensor(p + "self_attn.o_proj.weight")),
                    fn=_f32(f.get_tensor(p + "post_attention_layernorm.weight")),
                    w1=_f16_T(f.get_tensor(p + "mlp.gate_proj.weight")),
                    w2=_f16_T(f.get_tensor(p + "mlp.down_proj.weight")),
                    w3=_f16_T(f.get_tensor(p + "mlp.up_proj.weight")),
                )
            )

    return dict(cfg=cfg, embeddings=embeddings, layers=layers, fn_w=fn_w, lmw=lmw)
