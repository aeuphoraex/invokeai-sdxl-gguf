#!/usr/bin/env python3
"""Convert an SDXL single-file checkpoint's UNet into a quantized GGUF.

Only the UNet is quantized. Text encoders and VAE remain in the source
safetensors — InvokeAI's SDXL GGUF config references both files.

Key conversion goes through diffusers' own `from_single_file` so the mapping is
guaranteed correct rather than hand-rolled.

Usage:
  python3 sdxl_to_gguf.py <in.safetensors> <out.gguf> [--quant Q8_0|Q4_K|Q4_0]
"""
import argparse
import json
import os
import sys

import gguf
import torch

QUANTS = {
    "Q8_0": gguf.GGMLQuantizationType.Q8_0,
    "Q4_K": gguf.GGMLQuantizationType.Q4_K,
    "Q4_0": gguf.GGMLQuantizationType.Q4_0,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--quant", default="Q8_0", choices=sorted(QUANTS))
    ap.add_argument(
        "--pipe-load",
        action="store_true",
        default=True,
        help="load via StableDiffusionXLPipeline.from_single_file (default)",
    )
    a = ap.parse_args()

    print(f"[load] {a.src}", flush=True)
    from diffusers import StableDiffusionXLPipeline

    # torch_dtype=float16 keeps peak RAM around the checkpoint size.
    pipe = StableDiffusionXLPipeline.from_single_file(
        a.src,
        torch_dtype=torch.float16,
        local_files_only=False,
    )
    # Capture the UNet architecture BEFORE freeing the pipeline — embedded as
    # a GGUF string KV later so the loader can rebuild UNet2DConditionModel
    # without touching the source checkpoint (self-describing GGUF).
    unet_config_json = json.dumps(dict(pipe.unet.config))
    source_ckpt = os.path.abspath(a.src)
    unet_sd = {k: v.detach().cpu() for k, v in pipe.unet.state_dict().items()}
    print(f"[unet] {len(unet_sd)} diffusers-format tensors", flush=True)

    # Free everything that is not the state dict we are about to quantize.
    del pipe

    qtype = QUANTS[a.quant]
    print(f"[quant] {a.quant}", flush=True)

    w = gguf.GGUFWriter(a.dst, arch="stable-diffusion-xl-unet")
    w.add_name(os.path.basename(a.dst))
    w.add_description("SDXL UNet quantized from an InvokeAI checkpoint")
    # gguf-py has no add_quantization(); record it as a plain string KV.
    w.add_string("quantization.type", a.quant)
    # Self-describing KVs consumed by Main_GGUF_SDXL_Config.from_model_on_disk:
    #   sdxl.unet_config      -> UNet2DConditionModel architecture JSON
    #   sdxl.source_checkpoint-> where text encoders + VAE come from
    w.add_string("sdxl.unet_config", unet_config_json)
    w.add_string("sdxl.source_checkpoint", source_ckpt)

    n_params = 0
    counts: dict[str, int] = {}
    fails: list[str] = []
    # ROOT CAUSE of the all-F16 output (probed 2026-09-30 via q8_probe2.py):
    # gguf.quants.quantize(Q8_0) SUCCEEDS on matrix shapes, but
    # GGUFWriter.add_tensor() rejects the quantized uint8 payload with
    #   ValueError: Only F16, F32, F64, I8, I16, I32, I64 tensors are...
    # (gguf_writer.py:356) UNLESS raw_dtype=<GGMLQuantizationType> is passed —
    # the writer then re-derives the logical shape via
    # quant_shape_from_byte_shape.  Fix: raw_dtype=qt on add_tensor.
    # Also log EVERY fallback (the old first_fail only ever showed conv_in
    # and masked the real cause).
    # gguf-py quantizes along the LAST axis: k-quants need last % 256, Q8_0
    # needs last % 32. Conv weights (last dim 1/3/4/9) must stay F16.
    chain = [qtype] if qtype == gguf.GGMLQuantizationType.Q8_0 else [qtype, gguf.GGMLQuantizationType.Q8_0]
    chain.append(gguf.GGMLQuantizationType.F16)

    for name, t in unet_sd.items():
        if not t.is_floating_point():
            w.add_tensor(name, t.to(torch.int64).numpy())
            counts["int64"] = counts.get("int64", 0) + 1
            continue
        t32 = t.float().contiguous()
        arr = t32.numpy()
        n_params += t32.numel()
        placed = False
        for qt in chain:
            try:
                data = gguf.quants.quantize(arr.copy(), qt)
                # raw_dtype is REQUIRED for quantized (uint8) payloads.
                w.add_tensor(name, data, raw_dtype=qt)
                counts[qt.name] = counts.get(qt.name, 0) + 1
                placed = True
                break
            except Exception as e:  # noqa: BLE001
                if len(fails) < 20:
                    fails.append(f"{name} {tuple(t.shape)} -> {qt.name}: "
                                 f"{type(e).__name__}: {e}")
                continue
        if not placed:
            raise RuntimeError(f"could not quantize {name} ({t.shape})")

    print(f"[quant] breakdown: {counts}", flush=True)
    for line in fails:
        print(f"[quant] fallback: {line}", flush=True)

    # gguf-py API differs across versions; this build uses *_to_file names.
    if hasattr(w, "write"):
        w.write()
    else:
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()

    size = os.path.getsize(a.dst)
    src_size = os.path.getsize(a.src)
    print(f"[done] {a.dst}")
    print(f"       tensors={len(unet_sd)} params={n_params:,}")
    print(f"       src={src_size / 1e9:.2f} GB  dst={size / 1e6:.1f} MB")
    print(f"       ratio={size / src_size:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
