#!/usr/bin/env python3
"""Apply the SDXL-GGUF patches to an installed InvokeAI (site-packages).

Patches (in order):
  01-configs-main.py.diff                  -> Main_GGUF_SDXL_Config (self-describing GGUF classifier)
  02-configs-factory.py.diff               -> import + AnyModelConfig union entry
  03-model_loaders-stable_diffusion.py.diff -> StableDiffusionXlGgufModel loader (portable source-ckpt fallback)
  04-gguf-ggml_tensor.py.diff              -> CUDA dequant fallback for aten.linear/conv2d/group_norm/layer_norm

Usage:
  python apply_patches.py [--site-packages PATH] [--dry-run]

Defaults to the site-packages dir of the currently running interpreter, so
run it with the same Python that runs InvokeAI (e.g. /venv/main/bin/python3).

Idempotent: files already containing the patch markers are skipped.
Backs up each file to <patch-dir>/orig/ before the first modification.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATCH_DIR = HERE / "patches"

# One marker per file: if present, that file is already patched.
PATCHES = [
    ("01-configs-main.py.diff", "backend/model_manager/configs/main.py", "Main_GGUF_SDXL_Config"),
    ("02-configs-factory.py.diff", "backend/model_manager/configs/factory.py", "Main_GGUF_SDXL_Config"),
    (
        "03-model_loaders-stable_diffusion.py.diff",
        "backend/model_manager/load/model_loaders/stable_diffusion.py",
        "StableDiffusionXlGgufModel",
    ),
    (
        "04-gguf-ggml_tensor.py.diff",
        "backend/quantization/gguf/ggml_tensor.py",
        "dequant fallback for op",
    ),
]


def find_site_packages(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    import site

    for p in site.getsitepackages():
        if (Path(p) / "invokeai").is_dir():
            return Path(p)
    sys.exit("could not locate site-packages containing invokeai/; pass --site-packages")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site-packages", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    sp = find_site_packages(a.site_packages)
    print(f"site-packages: {sp}")
    orig = PATCH_DIR / "orig"
    failures = 0

    for diff_name, rel, marker in PATCHES:
        target = sp / rel
        diff = PATCH_DIR / diff_name
        if not diff.is_file():
            print(f"MISSING DIFF: {diff}")
            failures += 1
            continue
        if not target.is_file():
            print(f"MISSING TARGET: {target}")
            failures += 1
            continue
        src = target.read_text(encoding="utf-8")
        if marker in src:
            print(f"SKIP (already patched): {rel}")
            continue
        if a.dry_run:
            print(f"WOULD PATCH: {rel}")
            continue
        orig.mkdir(parents=True, exist_ok=True)
        bak = orig / f"{rel.replace('/', '_')}.orig"
        if not bak.exists():
            shutil.copy2(target, bak)
        r = subprocess.run(["patch", "-p0", "-i", str(diff), str(target)], capture_output=True, text=True)
        if r.returncode != 0:
            # diff headers are backup-path -> site-packages-path; try -p with strip levels
            r2 = subprocess.run(
                ["patch", "-p3", "--no-backup-if-mismatch", "-i", str(diff), str(target)],
                capture_output=True,
                text=True,
            )
            if r2.returncode != 0:
                print(f"PATCH FAIL: {rel}\n{r.stdout}{r.stderr}{r2.stdout}{r2.stderr}")
                failures += 1
                continue
        src = target.read_text(encoding="utf-8")
        if marker not in src:
            print(f"MARKER MISSING AFTER PATCH: {rel}")
            failures += 1
            continue
        print(f"PATCHED: {rel}")

    if failures:
        print(f"\n{failures} failure(s)")
        return 1

    # Verify: py_compile every touched file, then registry smoke test.
    if not a.dry_run:
        for _, rel, _ in PATCHES:
            subprocess.run([sys.executable, "-m", "py_compile", str(sp / rel)], check=True)
        print("py_compile OK")
        from invokeai.backend.model_manager.configs.main import Main_GGUF_SDXL_Config
        from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
        from invokeai.backend.model_manager.taxonomy import ModelSourceType

        cfg = Main_GGUF_SDXL_Config(
            path="/tmp/dummy.gguf",
            hash="sha256:" + "0" * 64,
            file_size=1,
            name="dummy",
            source="/tmp/dummy.gguf",
            source_type=ModelSourceType.Path,
        )
        hits = ModelLoaderRegistry.get_implementation(cfg, None)
        assert "StableDiffusionXlGgufModel" in str(hits), hits
        print(f"registry OK: {hits[0].__name__}")
        print("\nALL GREEN — restart InvokeAI to load the new loader.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
