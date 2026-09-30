#!/usr/bin/env python3
"""Make source_checkpoint portable: fall back to any installed SDXL checkpoint.

The converter embeds sdxl.source_checkpoint as the absolute path of the
checkpoint it was built from. When the GGUF is installed from a HuggingFace
URL on another machine, that path does not exist -- loader must not crash but
pick any installed SDXL single-file checkpoint to supply text encoders + VAE
(they are architecturally identical across SDXL checkpoints).

Idempotent. py_compile verified. Re-runs the registry smoke test.
"""
import os
import py_compile
import sys

BASE = "/venv/main/lib/python3.12/site-packages/invokeai"
SD = f"{BASE}/backend/model_manager/load/model_loaders/stable_diffusion.py"

OLD = '''    def _load_from_source_checkpoint(
        self, config: Main_GGUF_SDXL_Config, submodel_type: SubModelType
    ) -> AnyModel:
        if not config.source_checkpoint:
            raise ValueError(f"{type(config).__name__} has no source_checkpoint KV")

        with SilenceWarnings():
            pipeline = StableDiffusionXLPipeline.from_single_file(
                Path(config.source_checkpoint), torch_dtype=self._torch_dtype
            )
'''

NEW = '''    def _resolve_source_checkpoint(self, config: Main_GGUF_SDXL_Config) -> Path:
        """Resolve the checkpoint providing text encoders + VAE.

        The GGUF embeds the absolute path of the checkpoint it was built from.
        When installed from a HuggingFace URL on another machine that path is
        missing, so fall back to any installed SDXL single-file checkpoint --
        text encoders and VAE are architecturally identical across SDXL
        checkpoints. Raises with a clear message if none is installed.
        """
        candidates: list[Path] = []
        if config.source_checkpoint:
            candidates.append(Path(config.source_checkpoint))

        if not candidates or not candidates[0].is_file():
            for p in sorted((Path(config.path).parent.parent).rglob("*.safetensors")):
                if p.is_file() and p.name != Path(config.path).name:
                    candidates.append(p)
            # Also scan the configured models dirs as a last resort.
            for root in (
                Path(__import__("os").path.expanduser("~")) / "invokeai/models",
                Path("/workspace/invokeai/models"),
            ):
                if root.is_dir():
                    candidates.extend(sorted(root.rglob("*.safetensors")))

        for c in candidates:
            if c.is_file():
                if c != (config.source_checkpoint and Path(config.source_checkpoint)):
                    print(
                        f"[StableDiffusionXlGgufModel] source_checkpoint missing; "
                        f"using {c} for text encoders + VAE"
                    )
                return c

        raise ValueError(
            f"{type(config).__name__}: no source checkpoint available. The GGUF "
            f"embeds {config.source_checkpoint!r} which does not exist here, and "
            "no SDXL checkpoint is installed to supply text encoders + VAE. "
            "Install any SDXL single-file checkpoint first."
        )

    def _load_from_source_checkpoint(
        self, config: Main_GGUF_SDXL_Config, submodel_type: SubModelType
    ) -> AnyModel:
        source = self._resolve_source_checkpoint(config)

        with SilenceWarnings():
            pipeline = StableDiffusionXLPipeline.from_single_file(
                source, torch_dtype=self._torch_dtype
            )
'''


def main() -> int:
    src = open(SD, encoding="utf-8").read()
    if "_resolve_source_checkpoint" in src:
        print("already patched")
    else:
        if OLD not in src:
            print("ANCHOR FAIL: loader body not found verbatim")
            return 1
        src = src.replace(OLD, NEW, 1)
        open(SD, "w", encoding="utf-8", newline="").write(src)
        print("PATCHED _resolve_source_checkpoint")

    py_compile.compile(SD, doraise=True)
    print("COMPILE OK")

    # Smoke: import + registry still green.
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
    assert "StableDiffusionXlGgufModel" in str(hits)
    print("registry OK:", hits[0].__name__)
    print("ALL GREEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
