#!/usr/bin/env python3
"""Patch GGMLTensor dispatch for CUDA.

Problem: GGML_TENSOR_OP_TABLE only maps aten.linear.default when MPS is
available, so on the RTX 3090 any F.linear on a quantized weight raises
"all __torch_dispatch__ handlers returned NotImplemented".

Fix:
  1. Back up ggml_tensor.py to /workspace/.ro/backup-20260930/.
  2. Replace the __torch_dispatch__ fallback: unknown ops now dequantize
     GGMLTensor inputs and run the op on plain tensors (consistent with the
     existing design, where view/slice/addmm already dequant-and-run), with a
     one-time warning per op name instead of a hard failure.
  3. py_compile + a direct F.linear smoke test on a synthetic Q8_0 tensor.
"""
import hashlib
import py_compile
import shutil
import sys

SRC = "/venv/main/lib/python3.12/site-packages/invokeai/backend/quantization/gguf/ggml_tensor.py"
BACKUP_DIR = "/workspace/.ro/backup-20260930"

OLD = '''    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        # We will likely hit cases here in the future where a new op is encountered that is not yet supported.
        # The new op simply needs to be added to the GGML_TENSOR_OP_TABLE.
        if func in GGML_TENSOR_OP_TABLE:
            return GGML_TENSOR_OP_TABLE[func](func, args, kwargs)
        return NotImplemented
'''

NEW = '''    _warned_ops: set = set()

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        # Ops with special handling (keep quantized data / explicit dequant).
        if func in GGML_TENSOR_OP_TABLE:
            return GGML_TENSOR_OP_TABLE[func](func, args, kwargs)
        # Fallback for everything else (aten.linear, conv2d, group_norm, SDPA,
        # ...): dequantize GGMLTensor inputs and run the op on plain tensors.
        # This matches the existing design -- view/slice/addmm already
        # dequant-and-run -- and keeps new ops working without enumerating
        # them. One warning per op name for visibility.
        if func not in cls._warned_ops:
            cls._warned_ops.add(func)
            import sys as _sys

            print(f"[GGMLTensor] dequant fallback for op: {func}", file=_sys.stderr, flush=True)
        return dequantize_and_run(func, args, kwargs or {})
'''


def md5(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def main() -> int:
    src = open(SRC, encoding="utf-8").read()
    digest = md5(SRC)
    print(f"md5 before: {digest}")

    if "dequant fallback for op" in src:
        print("already patched")
    else:
        if OLD not in src:
            print("ANCHOR FAIL: __torch_dispatch__ body not found verbatim")
            return 1
        shutil.copy2(SRC, f"{BACKUP_DIR}/ggml_tensor.py.bak")
        print(f"backup -> {BACKUP_DIR}/ggml_tensor.py.bak")
        src = src.replace(OLD, NEW, 1)
        open(SRC, "w", encoding="utf-8", newline="").write(src)
        print("PATCHED")

    py_compile.compile(SRC, doraise=True)
    print("COMPILE OK")
    print(f"md5 after:  {md5(SRC)}")

    print("\nDISPATCH SMOKE TEST...")
    import numpy as np
    import torch

    import gguf
    from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor

    # Real Q8_0-quantized bytes (correct 32-elem block layout), as the
    # converter writes them.
    rows, cols = 64, 64
    w = np.random.randn(rows, cols).astype(np.float32)
    qb = gguf.quants.quantize(w, gguf.GGMLQuantizationType.Q8_0)
    data = torch.from_numpy(np.ascontiguousarray(qb))
    print(f"quantized bytes: {tuple(data.shape)} {data.dtype}")
    t = GGMLTensor(
        data,
        gguf.GGMLQuantizationType.Q8_0,
        torch.Size([rows, cols]),
        torch.float32,
    )
    x = torch.randn(4, cols)
    out = torch.nn.functional.linear(x, t)
    print(f"linear -> {tuple(out.shape)} {out.dtype}")
    assert out.shape == (4, rows)

    ref = torch.nn.functional.linear(x, t.get_dequantized_tensor())
    diff = (out - ref).abs().max().item()
    print(f"max abs diff vs manual dequant: {diff:.6f}")
    assert diff < 1e-4

    # bias path (addmm)
    b = torch.randn(rows)
    out2 = torch.nn.functional.linear(x, t, b)
    ref2 = torch.nn.functional.linear(x, t.get_dequantized_tensor(), b)
    assert (out2 - ref2).abs().max().item() < 1e-4
    print("bias path OK")
    print("ALL GREEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
