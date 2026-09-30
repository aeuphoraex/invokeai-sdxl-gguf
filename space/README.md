---
title: InvokeAI SDXL-GGUF
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# InvokeAI — full web UI running SDXL from a 3 GB GGUF

A complete [InvokeAI](https://github.com/invoke-ai/InvokeAI) 6.14.1 instance
(the regular browser UI at this URL) patched to load SDXL from a
**Q8_0 GGUF UNet** instead of a 5–7 GB checkpoint:

| | fp16/fp8 checkpoint | this GGUF |
|---|---|---|
| UNet weights on disk | 5–7 GB | **3.04 GB** |
| VRAM after generation (3090) | 15,412 MB | **5,172 MB** |

## What you get

- **Full InvokeAI UI** — graphs, canvas, queues, style presets, all nodes.
- **SDXL UNet as GGUF** (`AEUPH/sdxl-unet-q8-gguf`), self-describing:
  the file header carries `sdxl.unet_config` (architecture JSON) +
  `sdxl.source_checkpoint`, so the Model Manager classifies it from the
  header alone — no weight scan.
- **Text encoders + VAE** from `stabilityai/stable-diffusion-xl-base-1.0`
  (fp16, baked into the image; the GGUF never duplicates them).
- Install-from-HF-URL works too: paste
  `https://huggingface.co/AEUPH/sdxl-unet-q8-gguf/resolve/main/sdxl-unet-q8.gguf`
  into Model Manager → *Install from URL* on any patched InvokeAI.

## Patches applied at image build (Apache-2.0, see NOTICE)

1. `configs/main.py` — `Main_GGUF_SDXL_Config` header classifier
2. `configs/factory.py` — tagged-union entry
3. `model_loaders/stable_diffusion.py` — `StableDiffusionXlGgufModel` loader
   (portable TE/VAE source: diffusers-layout dirs or any SDXL checkpoint)
4. `gguf/ggml_tensor.py` — CUDA dequant fallback for `aten.linear`/`conv2d`/
   `group_norm`/`layer_norm` (upstream mapped `linear` on MPS only)

## Tips

- First boot installs the GGUF into the registry (~15 s after API is up).
- Free/CPU hardware: SDXL will be slow — 512×512 @ 6–10 steps is a
  reasonable smoke test; switch to a GPU hardware tier for real use.
- Weights stay quantized on the device and dequantize per-op — that's the
  VRAM saving. Speed cost vs checkpoint: ~1.6× (no fused dequant kernel yet).

Repo + converter + benchmarks: https://huggingface.co/AEUPH/sdxl-unet-q8-gguf
