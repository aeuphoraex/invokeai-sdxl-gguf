#!/usr/bin/env python3
"""Repair stable_diffusion.py: insert runtime imports the patcher skipped.

The patcher's idempotence guard keys on the string "Main_GGUF_SDXL_Config",
so once the config-import replace landed, the follow-up runtime-import
replace (ModelLoader / gguf_sd_loader / TorchDevice) was skipped -> NameError
at class-definition time.

Idempotent: skips anything already present.
"""
import py_compile
import sys

BASE = "/venv/main/lib/python3.12/site-packages/invokeai"
SD = f"{BASE}/backend/model_manager/load/model_loaders/stable_diffusion.py"

WANTED = [
    "from invokeai.backend.model_manager.load.load_default import ModelLoader",
    "from invokeai.backend.quantization.gguf.loaders import gguf_sd_loader",
    "from invokeai.backend.util.devices import TorchDevice",
]
ANCHOR = "from invokeai.backend.util.silence_warnings import SilenceWarnings"


def main() -> int:
    src = open(SD, encoding="utf-8").read()
    missing = [w for w in WANTED if w not in src]
    if not missing:
        print("imports already present")
    else:
        if ANCHOR not in src:
            print(f"ANCHOR FAIL: {ANCHOR!r} not found")
            return 1
        block = "\n".join(missing) + "\n"
        src = src.replace(ANCHOR, block + ANCHOR, 1)
        open(SD, "w", encoding="utf-8", newline="").write(src)
        print(f"INSERTED {len(missing)} imports:")
        for m in missing:
            print(f"  {m}")

    py_compile.compile(SD, doraise=True)
    print("COMPILE OK")

    print("\nIMPORT SMOKE TEST...")
    from invokeai.backend.model_manager.configs.factory import AnyModelConfig  # noqa: F401
    from invokeai.backend.model_manager.configs.main import Main_GGUF_SDXL_Config
    from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
    from invokeai.backend.model_manager.taxonomy import ModelSourceType

    tag = Main_GGUF_SDXL_Config.get_tag()
    print(f"config tag: {tag}")
    cfg = Main_GGUF_SDXL_Config(
        path="/tmp/dummy.gguf",
        hash="sha256:0" * 8,
        file_size=1,
        name="dummy",
        source="/tmp/dummy.gguf",
        source_type=ModelSourceType.Path,
    )
    hits = ModelLoaderRegistry.get_implementation(cfg, None)
    print(f"loader registered: {hits[0]}")
    assert "StableDiffusionXlGgufModel" in str(hits), "loader not registered!"

    # And the module itself must import cleanly now.
    import importlib

    m = importlib.import_module(
        "invokeai.backend.model_manager.load.model_loaders.stable_diffusion"
    )
    assert hasattr(m, "StableDiffusionXlGgufModel")
    print("loader class importable:", m.StableDiffusionXlGgufModel)
    print("ALL GREEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
