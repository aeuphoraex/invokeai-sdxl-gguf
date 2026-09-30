#!/usr/bin/env python3
"""Repair factory.py: the idempotence guard skipped the AnyModelConfig union entry.

After the import replace landed, "Main_GGUF_SDXL_Config" existed in the file,
so the union-entry replace was skipped -> pydantic tagged union never learned
tag 'main.gguf_quantized.sdxl' -> install fails with union_tag_invalid.

Verifies by validating a dumped Main_GGUF_SDXL_Config through the union.
"""
import py_compile
import sys

BASE = "/venv/main/lib/python3.12/site-packages/invokeai"
FACTORY = f"{BASE}/backend/model_manager/configs/factory.py"

IMPORT_ANCHOR = "    Main_GGUF_ZImage_Config,\n"
TAG_ANCHOR = (
    "        Annotated[Main_GGUF_ZImage_Config, Main_GGUF_ZImage_Config.get_tag()],\n"
)
OUR_TAG_LINE = (
    "        Annotated[Main_GGUF_SDXL_Config, Main_GGUF_SDXL_Config.get_tag()],\n"
)


def main() -> int:
    src = open(FACTORY, encoding="utf-8").read()

    if OUR_TAG_LINE in src:
        print("union entry already present")
    else:
        if IMPORT_ANCHOR not in src:
            print(f"IMPORT ANCHOR FAIL: {IMPORT_ANCHOR!r}")
            return 1
        if TAG_ANCHOR not in src:
            print(f"TAG ANCHOR FAIL: {TAG_ANCHOR!r}")
            return 1
        src = src.replace(TAG_ANCHOR, TAG_ANCHOR + OUR_TAG_LINE, 1)
        open(FACTORY, "w", encoding="utf-8", newline="").write(src)
        print("INSERTED union entry after Main_GGUF_ZImage_Config")

    py_compile.compile(FACTORY, doraise=True)
    print("COMPILE OK")

    print("\nUNION VALIDATION TEST...")
    from pydantic import TypeAdapter

    from invokeai.backend.model_manager.configs.factory import AnyModelConfig
    from invokeai.backend.model_manager.configs.main import Main_GGUF_SDXL_Config

    cfg = Main_GGUF_SDXL_Config(
        path="/tmp/dummy.gguf",
        hash="sha256:" + "0" * 64,
        file_size=1,
        name="dummy",
        source="/tmp/dummy.gguf",
        source_type="path",
    )
    dumped = cfg.model_dump()
    adapter: TypeAdapter[AnyModelConfig] = TypeAdapter(AnyModelConfig)
    validated = adapter.validate_python(dumped)
    print(f"validated as: {type(validated).__name__}")
    assert isinstance(validated, Main_GGUF_SDXL_Config), (
        f"union resolved to {type(validated).__name__}"
    )
    print("ALL GREEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
