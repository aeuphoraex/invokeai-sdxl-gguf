#!/usr/bin/env python3
"""Dump GGUF headers locally (pure Python, no gguf-py needed).

Shows architecture KVs, all non-standard KVs, and tensor names/shapes so we
can see how the demo GGUFs embed runtime execution code alongside weights.
"""
import json
import struct
import sys
from pathlib import Path

KV_TYPES = {
    0: ("uint8", "<B", 1),
    1: ("int8", "<b", 1),
    2: ("uint16", "<H", 2),
    3: ("int16", "<h", 2),
    4: ("uint32", "<I", 4),
    5: ("int32", "<i", 4),
    6: ("float32", "<f", 4),
    7: ("bool", "<?", 1),
    8: ("str", None, None),
    9: ("array", None, None),
    10: ("uint64", "<Q", 8),
    11: ("int64", "<q", 8),
    12: ("float64", "<d", 8),
}

# llama.cpp standard keys we de-emphasize when listing
STD = {
    "general.architecture", "general.name", "general.file_type",
    "general.quantization_version", "general.alignment",
    "general.source.url", "general.source.huggingface.repository",
}


class R:
    """Streaming reader: only the header is touched, never tensor data.

    The demo GGUFs are up to 5GB, so we must not read the whole file.
    """

    def __init__(self, fh):
        self.fh = fh
        self.i = 0

    def take(self, n):
        v = self.fh.read(n)
        assert len(v) == n, f"EOF at {self.i} wanting {n}"
        self.i += n
        return v

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def len32(self):
        """GGUF v1 uses u32 for name/string/array lengths; v2+ uses u64."""
        return self.u32() if getattr(self, "ver", 3) < 2 else self.u64()

    def kv_value(self, t):
        name, fmt, size = KV_TYPES[t]
        if fmt:
            return struct.unpack(fmt, self.take(size))[0]
        if name == "str":
            n = self.len32()
            return self.take(n).decode("utf-8", "replace")
        if name == "array":
            et = self.u32()
            n = self.len32()
            if et == 8:
                return [self.kv_value(8) for _ in range(n)]
            _, fmt, size = KV_TYPES[et]
            if fmt:
                return list(struct.unpack(f"<{n}{fmt[1]}", self.take(n * size)))
            raise ValueError(f"nested array of type {et}")
        raise ValueError(f"unknown kv type {t}")


def dump(path: Path, max_kv=40, max_tensors=25) -> None:
    with open(path, "rb") as fh:
        r = R(fh)
        magic = r.take(4)
        if magic != b"GGUF":
            print(f"{path.name}: not a GGUF ({magic!r})")
            return
        ver = r.u32()
        r.ver = ver
        n_tensors = r.u64()
        n_kv = r.u64()
        print("=" * 74)
        print(f"{path.name}  ({path.stat().st_size:,} B, GGUF v{ver}, "
              f"{n_tensors} tensors, {n_kv} KVs)")
        kvs: dict = {}
        for _ in range(n_kv):
            klen = r.len32()
            key = r.take(klen).decode("utf-8", "replace")
            t = r.u32()
            kvs[key] = r.kv_value(t)
        _dump_tail(path, kvs, n_tensors, r, max_kv, max_tensors)


def _dump_tail(path, kvs, n_tensors, r, max_kv, max_tensors) -> None:
    std = {k: v for k, v in kvs.items() if k in STD}
    custom = {k: v for k, v in kvs.items() if k not in STD}
    print("-- standard KVs --")
    for k, v in std.items():
        print(f"  {k} = {str(v)[:100]}")
    print(f"-- custom KVs ({len(custom)}) --")
    for i, (k, v) in enumerate(custom.items()):
        if i >= max_kv:
            print(f"  ... {len(custom) - max_kv} more")
            break
        s = json.dumps(v) if not isinstance(v, str) else v
        print(f"  {k} = {s[:160]}{'...' if len(s) > 160 else ''}")

    # tensor directory
    align = int(kvs.get("general.alignment", 32))
    print(f"-- tensors ({n_tensors}) --")
    names = []
    for _ in range(n_tensors):
        nlen = r.len32()
        name = r.take(nlen).decode("utf-8", "replace")
        nd = r.u32()
        shape = struct.unpack(f"<{nd}Q", r.take(8 * nd))
        dtype = r.u32()
        off = r.u64()
        names.append((name, tuple(int(x) for x in reversed(shape)), dtype, off))
    for i, (name, shape, dtype, off) in enumerate(names):
        if i >= max_tensors:
            print(f"  ... {n_tensors - max_tensors} more")
            break
        print(f"  [{dtype:2d}] {name}  {shape}")

    # classify: code-bearing tensor name heuristics
    codey = [n for n, _, _, _ in names
             if any(s in n.lower() for s in
                    ("kernel", "code", "shader", "fsa", "grammar", "rule",
                     "program", "bytecode", "ops", "script", "wasm", "ir"))]
    if codey:
        print(f"-- code-like tensor names ({len(codey)}) --")
        for n in codey[:20]:
            print(f"  {n}")


if __name__ == "__main__":
    targets = sys.argv[1:]
    if not targets:
        targets = [
            "JinxOS-Kernel-Hyper.gguf",
            "JinxOS-WebOS-Kernel.gguf",
            "OMNI_MORPHEME_ZERO_MAC.gguf",
            "2026backup/4TH-DH-VISIONML.gguf",
            "2026backup/SKELETON_NEURAL.gguf",
        ]
    for t in targets:
        try:
            dump(Path(t))
        except Exception as e:  # noqa: BLE001
            print(f"{t}: ERROR {type(e).__name__}: {e}")
