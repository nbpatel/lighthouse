# Llama-3 payload via torch-mlir (branch `llama3_actual_data_torch_mlir`)

## Why
The Inductor-vs-lighthouse comparison so far fed lighthouse a **hand-written**
linalg payload (the `Builder` in `llama3_payload.py`). Inductor starts from the
real PyTorch model. To compare backends fairly, lighthouse should ingest the same
source. This branch generates the linalg payload by lowering a PyTorch Llama-3
through **torch-mlir**, using lighthouse's existing torch ingress.

## Files
- `llama3_torch_model.py` — PyTorch Llama-3 (1 block + final norm + LM head).
  Same math as `numpy_ref_block_llama`: RMSNorm, real half-split RoPE, GQA causal
  attention, SwiGLU FFN. Avoids complex-number RoPE and the KV-cache `index_copy`
  (torch-mlir/export handle those poorly). Toy dims T=256, C=256, hidden=1024,
  H=4, n_kv=2. Exposes `get_init_inputs` / `get_inputs` for the ingress loader.
- `llama3_torch_mlir.py` — driver. Calls `lighthouse.ingress.torch.import_from_file`
  (torch-mlir FX importer) -> linalg-on-tensors. `-o` saves, `--f16` casts first,
  `--dialect` picks torch/tosa/linalg.

## Status: WORKING
`python examples/xegpu/llama3_torch_mlir.py -o /tmp/p.mlir` emits a linalg payload
that parses in lighthouse's MLIR context. Op histogram:
`batch_matmul:2, matmul:8, generic:51, fill:9, transpose:5`
(8 = wq/wk/wv/wo/w1/w2/w3/lm_head; 2 batch_matmul = attention QK^T and @V).

## Environment
torch-mlir is NOT in the default env. Installed the cp312-abi3 nightly wheel into
the uv venv with `--no-deps` (only needs numpy/packaging; does not touch torch):
`uv pip install --python .venv/bin/python3 --no-deps <torch_mlir wheel>`.
The intended clean path is `uv sync --extra ingress_torch_rocm` (pyproject pins
`torch-mlir==20260805.836`), but a full sync risks disturbing the pinned local
LLVM / MLIR bindings, so a scoped `--no-deps` install was used instead.
Ingress bridges torch-mlir's MLIR to lighthouse's via TEXT (str -> re-parse), so
the two MLIR builds need not be ABI-compatible.

## Next steps
- Feed this payload into the existing schedule. Blocker: the schedule matches ops
  by position in a hand-maintained `kinds` list; a torch-mlir payload has no
  `kinds`, and its op structure differs (attention is `batch_matmul` + separate
  softmax here vs the hand payload's fused flash attention; extra transpose/
  expand/collapse from head reshapes). Either derive `kinds` from the IR or adapt
  the schedule to discover op structure.
- Decide dtype policy: model is f32; payload's f16 DPAS casts are a lowering
  choice (`--f16` reproduces them end-to-end but changes numerics).
- Match the pinned torch-mlir version (20260805.836) if IR drift matters.
