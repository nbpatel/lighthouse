"""Generate the Llama-3 linalg payload via torch-mlir (instead of by hand).

Lowers the PyTorch Llama-3 model in llama3_torch_model.py to linalg-on-tensors
MLIR using lighthouse's torch ingress (torch-mlir's FX importer). This replaces
the hand-emitted `Builder` payload with a compiler-generated one, so a
lighthouse-vs-Inductor comparison starts from the same PyTorch source.

Usage:
    python llama3_torch_mlir.py                 # print the linalg payload
    python llama3_torch_mlir.py -o payload.mlir # also save it
    python llama3_torch_mlir.py --dialect torch # emit the torch dialect instead
    python llama3_torch_mlir.py --f16           # cast the model to f16 first
"""

import argparse
from pathlib import Path

import torch

from lighthouse.ingress.torch import import_from_file

_HERE = Path(__file__).parent
_MODEL_FILE = _HERE / "llama3_torch_model.py"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", type=Path, default=None, help="save MLIR to this file")
    ap.add_argument(
        "--dialect",
        default="linalg-on-tensors",
        choices=["linalg-on-tensors", "torch", "tosa"],
        help="target dialect (default: linalg-on-tensors)",
    )
    ap.add_argument("--f16", action="store_true", help="cast model to float16 before import")
    args = ap.parse_args()

    mlir_text = import_from_file(
        _MODEL_FILE,
        model_class_name="Llama3",
        init_args_fn_name="get_init_inputs",
        sample_args_fn_name="get_inputs",
        model_datatype=torch.float16 if args.f16 else None,
        dialect=args.dialect,
    )

    if args.out is not None:
        args.out.write_text(mlir_text)
        print(f"wrote {args.dialect} payload -> {args.out} ({len(mlir_text)} bytes)")

    # Quick op histogram so the payload's shape is visible at a glance.
    counts = {}
    for line in mlir_text.splitlines():
        s = line.strip()
        for op in ("linalg.matmul", "linalg.generic", "linalg.fill",
                   "linalg.softmax", "linalg.transpose", "linalg.batch_matmul"):
            if op + " " in s or s.startswith(op) or ("= " + op) in s:
                counts[op] = counts.get(op, 0) + 1
    print("op histogram:", {k: counts[k] for k in sorted(counts)})

    if args.out is None:
        print(mlir_text)


if __name__ == "__main__":
    main()
