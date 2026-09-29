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

## Status
### Ingress: WORKING
`python examples/xegpu/llama3_torch_mlir.py -o /tmp/p.mlir` emits a linalg payload
that parses in lighthouse's MLIR context. Op histogram:
`batch_matmul:2, matmul:8, generic:51, fill:9, transpose:5`
(8 = wq/wk/wv/wo/w1/w2/w3/lm_head; 2 batch_matmul = attention QK^T and @V).

### Milestone 0 -- CPU numerical validation: PASSING
`python examples/xegpu/llama3_torch_mlir_check.py` lowers the payload to CPU
(one-shot-bufferize -> convert-linalg-to-loops -> LLVM), JITs it, and checks the
output against the PyTorch model on the same inputs: `max|diff| = 8.3e-7`,
"Result matched!". This validates ingress+payload numerics before any GPU work.
Notes: RMSNorm uses `x*x` and `1/sqrt` (not pow/rsqrt) so it legalizes to libm;
`@main` returns a tensor, so after bufferization the C wrapper takes a result
memref descriptor as its first arg; link libmlir_c_runner_utils for `memrefCopy`.

## Environment
torch-mlir is NOT in the default env. Installed the cp312-abi3 nightly wheel into
the uv venv with `--no-deps` (only needs numpy/packaging; does not touch torch):
`uv pip install --python .venv/bin/python3 --no-deps <torch_mlir wheel>`.
The intended clean path is `uv sync --extra ingress_torch_rocm` (pyproject pins
`torch-mlir==20260805.836`), but a full sync risks disturbing the pinned local
LLVM / MLIR bindings, so a scoped `--no-deps` install was used instead.
Ingress bridges torch-mlir's MLIR to lighthouse's via TEXT (str -> re-parse), so
the two MLIR builds need not be ABI-compatible.

### Feasibility gate -- torch-mlir matmul on Intel GPU (PVC): TOOLCHAIN SURVIVES
`python examples/xegpu/torch_mlir_matmul_gpu.py` exports `x @ w` (f16) via
torch-mlir and lowers it with the EXISTING `matmul_schedule` + `xegpu_to_binary`.
Result: lowers all the way to `gpu.binary` + XeVM, NO segfault. Key facts learned:
- DPAS needs f16 A/B (f32 accumulate). f32 inputs -> "failed to legalize xegpu.dpas".
  With f16 inputs torch-mlir emits the exact HW form: matmul ins(f16,f16) outs(f32)
  -> f32, then truncf to f16.
- Weights must be a forward INPUT (function arg), not an nn.Parameter -- the matmul
  schedule prefetches operand buffers, which can't come from a baked constant.
- The existing per-op-class schedules consume a torch-mlir payload directly.
Device present: Intel Data Center GPU Max 1100 (PVC), Level Zero.

### Milestone 1a -- torch-mlir matmul EXECUTED on Intel GPU: PASSING
`torch_mlir_matmul_gpu.py` now runs the full loop and checks numerics:
rel=4.4e-4, "GPU PASSED". The complete torch-mlir -> XeGPU recipe:
1. ingress -> linalg (f16 inputs -> DPAS-ready matmul ins(f16,f16) outs(f32)).
2. `convert_function_results("main")` (lighthouse.schedule.func) -> DPS: return
   value becomes leading memref arg (output = arg 0), matching the runner.
3. `GPUMemoryManager.emit_memory_management_funcs(mod, host_inputs=[out,x,w])` ->
   injects gpu_alloc_/dealloc_/copy_ host wrappers before lowering.
4. `matmul_schedule` + `xegpu_to_binary()` -> gpu.binary + XeVM.
5. `Runner(mem_manager_cls=GPUMemoryManager, shared_libs=["libmlir_levelzero_runtime.so"])`
   .execute(host_input_buffers=[out,x,w], argument_access_callback=arg0 cb).

## Next steps
Chosen approach: **correctness-first generic GPU schedule** (classify ops by
walking the IR; no `kinds`, no flash-attention special-casing).
- Milestone 1 (GPU, correct but unoptimized): tile each op by type into a
  work-group `forall`, seed a default `sg_layout`/`sg_data` anchor per op type
  (matmul / batch_matmul / elementwise / reduction), then reuse `xegpu_to_binary()`
  (`gpu-lower-to-xevm-pipeline`, xegpu-op-level=workgroup) for the automatic
  wg->sg + layout propagation. NOTE from survey: lighthouse has NO fully-automatic
  layout assignment -- anchors must be seeded by hand per op. `examples/xegpu/
  kernel_bench.py` (`infer_parameters` + `lower_to_llvm`) is the nearest generic
  driver but only routes single-op-class payloads to 4 hand-written schedules.
- Milestone 2: same PyTorch model -> lighthouse and Inductor -> correctness + timing.
- Dtype: model is f32; payload's f16 DPAS casts are a lowering choice (`--f16`
  reproduces them end-to-end but changes numerics).
- Match pinned torch-mlir 20260805.836 if IR drift matters (installed 20260923).
