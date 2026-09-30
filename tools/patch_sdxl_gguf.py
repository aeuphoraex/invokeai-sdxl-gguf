#!/usr/bin/env python3
"""Apply the SDXL-GGUF integration patch to InvokeAI site-packages (in place).

Three edits, each guarded by an exact anchor assertion:
  1. configs/main.py            -> helper + Main_GGUF_SDXL_Config class
  2. configs/factory.py         -> import + AnyModelConfig union entry
  3. load/model_loaders/stable_diffusion.py -> imports + loader class

Idempotent: re-running skips files already patched.
Verifies with py_compile. Backups already exist at /workspace/.ro/backup-20260930/.
"""
import py_compile
import sys

BASE = "/venv/main/lib/python3.12/site-packages/invokeai"

CONFIG_CLASS = '''def _read_sdxl_gguf_header(path):
    """Return (sdxl.* string KVs, tensor names) from a GGUF header.

    Only the GGUF header is parsed -- no tensor data is read, so classification
    stays cheap even for multi-GB files.
    """
    import gguf

    reader = gguf.GGUFReader(str(path))
    kvs: dict[str, str] = {}
    for key, field in reader.fields.items():
        if key.startswith("sdxl."):
            try:
                # STRING KVs store the value as the trailing uint8 part.
                kvs[key] = bytes(field.parts[-1]).decode("utf-8")
            except Exception:
                continue
    names = {t.name for t in reader.tensors}
    return kvs, names


class Main_GGUF_SDXL_Config(Checkpoint_Config_Base, Main_Config_Base, Config_Base):
    """Model config for GGUF-quantized SDXL UNet models.

    The GGUF is self-describing: converter-written string KVs carry the UNet
    architecture JSON (``sdxl.unet_config``) and the path of the original
    single-file checkpoint (``sdxl.source_checkpoint``) that provides the text
    encoders + VAE, because a GGUF holds the UNet only.
    """

    base: Literal[BaseModelType.StableDiffusionXL] = Field(default=BaseModelType.StableDiffusionXL)
    format: Literal[ModelFormat.GGUFQuantized] = Field(default=ModelFormat.GGUFQuantized)
    variant: ModelVariantType = Field(default=ModelVariantType.Normal)

    source_checkpoint: str | None = Field(
        default=None,
        description="Path to the original single-file checkpoint providing text encoders and VAE.",
    )
    unet_config_json: str | None = Field(
        default=None,
        description="UNet2DConditionModel architecture JSON embedded in the GGUF header (sdxl.unet_config KV).",
    )

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)
        raise_for_override_fields(cls, override_fields)

        kvs, names = _read_sdxl_gguf_header(mod.path)
        if "sdxl.unet_config" not in kvs or "sdxl.source_checkpoint" not in kvs:
            raise NotAMatchError("gguf file lacks sdxl.* self-description KVs")
        if not any(n.startswith("add_embedding.") for n in names):
            # SDXL-specific module (addition_embed_type="text_time");
            # SD1/2/FLUX/ZImage/Wan GGUFs never carry it.
            raise NotAMatchError("gguf does not look like an SDXL UNet")

        return cls(
            **override_fields,
            source_checkpoint=kvs["sdxl.source_checkpoint"],
            unet_config_json=kvs["sdxl.unet_config"],
        )


'''

LOADER_CLASS = '''


@ModelLoaderRegistry.register(base=BaseModelType.StableDiffusionXL, type=ModelType.Main, format=ModelFormat.GGUFQuantized)
class StableDiffusionXlGgufModel(ModelLoader):
    """SDXL GGUF: UNet from the .gguf file, text encoders + VAE from its source checkpoint.

    KV-driven and prefix-filtered by design so a future multi-component bundle
    (encoders/vae/vision tower under ``unet.*``/``text_encoder.*``/... prefixes
    plus a ``bundle.components`` KV) loads through this same class.
    """

    def _load_model(self, config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> AnyModel:
        if not isinstance(config, Main_GGUF_SDXL_Config):
            raise ValueError(f"Expected Main_GGUF_SDXL_Config, got {type(config).__name__}.")

        if submodel_type is None or submodel_type == SubModelType.UNet:
            return self._load_unet(config)
        return self._load_from_source_checkpoint(config, submodel_type)

    def _load_unet(self, config: Main_GGUF_SDXL_Config) -> AnyModel:
        import json

        import accelerate
        from diffusers import UNet2DConditionModel

        compute_dtype = TorchDevice.choose_bfloat16_safe_dtype(TorchDevice.choose_torch_device())
        sd = gguf_sd_loader(Path(config.path), compute_dtype=compute_dtype)

        # ComfyUI-style prefixes (our converter writes plain diffusers keys).
        for prefix in ("model.diffusion_model.", "diffusion_model."):
            if any(k.startswith(prefix) for k in sd):
                sd = {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in sd.items()}

        if not config.unet_config_json:
            raise ValueError(f"{type(config).__name__} has no unet_config_json KV")
        unet_config = json.loads(config.unet_config_json)

        with accelerate.init_empty_weights():
            model = UNet2DConditionModel.from_config(unet_config)
        model.load_state_dict(sd, assign=True)
        return model

    def _load_from_source_checkpoint(
        self, config: Main_GGUF_SDXL_Config, submodel_type: SubModelType
    ) -> AnyModel:
        if not config.source_checkpoint:
            raise ValueError(f"{type(config).__name__} has no source_checkpoint KV")

        with SilenceWarnings():
            pipeline = StableDiffusionXLPipeline.from_single_file(
                Path(config.source_checkpoint), torch_dtype=self._torch_dtype
            )

        # Prefetch everything EXCEPT the UNet: that cache slot must stay owned
        # by the GGUF model, never by the fp16 copy from the source checkpoint.
        for subtype in SubModelType:
            if subtype == submodel_type or subtype == SubModelType.UNet:
                continue
            if submodel := getattr(pipeline, subtype.value, None):
                self._apply_fp8_layerwise_casting(submodel, config, subtype)
                self._ram_cache.put(get_model_cache_key(config.key, subtype), model=submodel, prefetch=True)

        result = getattr(pipeline, submodel_type.value, None)
        if result is None:
            raise ValueError(f"Source checkpoint has no submodel {submodel_type.value}")
        return self._apply_fp8_layerwise_casting(result, config, submodel_type)
'''

FILES: list[str] = []


def replace(path: str, old: str, new: str, *, append: bool = False) -> None:
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    if "Main_GGUF_SDXL_Config" in src:
        print(f"SKIP (already patched): {path}")
        return
    n = src.count(old)
    if n != 1:
        print(f"ANCHOR FAIL: {path}: found {n} occurrences (expected 1) of {old[:70]!r}")
        sys.exit(1)
    src = src.replace(old, new + old if append else new.replace("{ANCHOR}", old), 1)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(src)
    print(f"PATCHED: {path}")
    FILES.append(path)


def main() -> int:
    # --- 1. configs/main.py: helper + config class before Ideogram4 config ---
    replace(
        f"{BASE}/backend/model_manager/configs/main.py",
        "class Main_Diffusers_Ideogram4_Config(Diffusers_Config_Base, Main_Config_Base, Config_Base):",
        CONFIG_CLASS + "{ANCHOR}",
    )

    # --- 2. configs/factory.py: import + union entry ---
    replace(
        f"{BASE}/backend/model_manager/configs/factory.py",
        "    Main_GGUF_ZImage_Config,\n",
        "    Main_GGUF_ZImage_Config,\n    Main_GGUF_SDXL_Config,\n",
    )
    replace(
        f"{BASE}/backend/model_manager/configs/factory.py",
        "        Annotated[Main_GGUF_ZImage_Config, Main_GGUF_ZImage_Config.get_tag()],\n",
        "        Annotated[Main_GGUF_ZImage_Config, Main_GGUF_ZImage_Config.get_tag()],\n"
        "        Annotated[Main_GGUF_SDXL_Config, Main_GGUF_SDXL_Config.get_tag()],\n",
    )

    # --- 3. stable_diffusion.py: config import, runtime imports, loader class ---
    replace(
        f"{BASE}/backend/model_manager/load/model_loaders/stable_diffusion.py",
        "    Main_Diffusers_SDXLRefiner_Config,\n)",
        "    Main_Diffusers_SDXLRefiner_Config,\n    Main_GGUF_SDXL_Config,\n)",
    )
    replace(
        f"{BASE}/backend/model_manager/load/model_loaders/stable_diffusion.py",
        "from invokeai.backend.util.silence_warnings import SilenceWarnings",
        "from invokeai.backend.model_manager.load.load_default import ModelLoader\n"
        "from invokeai.backend.quantization.gguf.loaders import gguf_sd_loader\n"
        "from invokeai.backend.util.devices import TorchDevice\n"
        "from invokeai.backend.util.silence_warnings import SilenceWarnings",
    )
    # append loader class at EOF
    path = f"{BASE}/backend/model_manager/load/model_loaders/stable_diffusion.py"
    if "StableDiffusionXlGgufModel" in open(path, encoding="utf-8").read():
        print(f"SKIP (already patched): {path}")
    else:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(LOADER_CLASS)
        print(f"APPENDED: {path}")
        FILES.append(path)

    # --- verify ---
    targets = FILES or [
        f"{BASE}/backend/model_manager/configs/main.py",
        f"{BASE}/backend/model_manager/configs/factory.py",
        f"{BASE}/backend/model_manager/load/model_loaders/stable_diffusion.py",
    ]
    for t in targets:
        py_compile.compile(t, doraise=True)
        print(f"COMPILE OK: {t}")

    # --- import smoke test: registry + union must see the new class ---
    print("\nIMPORT SMOKE TEST...")
    from invokeai.backend.model_manager.configs.factory import AnyModelConfig  # noqa: F401
    from invokeai.backend.model_manager.configs.main import Main_GGUF_SDXL_Config
    from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry

    tag = Main_GGUF_SDXL_Config.get_tag()
    print(f"config tag: {tag}")
    hits = ModelLoaderRegistry.lookup(
        base=Main_GGUF_SDXL_Config.model_fields["base"].default,
        type=Main_GGUF_SDXL_Config.model_fields["type"].default,
        format=Main_GGUF_SDXL_Config.model_fields["format"].default,
    )
    print(f"loader registered: {hits}")
    assert "StableDiffusionXlGgufModel" in str(hits), "loader not registered!"
    print("ALL GREEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
