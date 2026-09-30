# SDXL Q8_0 GGUF — ComfyUI workflow

ComfyUI workflow for the `sdxl-unet-q8.gguf` UNet (3.04 GB, Q8_0) from the
[InvokeAI-SDXL-GGUF](https://huggingface.co/AEUPH/sdxl-unet-q8-gguf) project.

## Files

| file | what it is |
|---|---|
| `sdxl-gguf-q8.json` | drag-and-drop UI workflow |
| `sdxl-gguf-q8.api.json` | API/prompt format (scripting, `POST /prompt`) |
| `README.md` | this file |

## Prerequisites

1. **[ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)** custom node
   (provides `UnetLoaderGGUF`):

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/city96/ComfyUI-GGUF
   pip install -r ComfyUI-GGUF/requirements.txt
   ```

2. Model files — this GGUF is **UNet only**; text encoders + VAE come from a
   standard SDXL setup:

   | file | put in |
   |---|---|
   | `sdxl-unet-q8.gguf` (3.04 GB) | `ComfyUI/models/unet/` |
   | `clip_l.safetensors` + `clip_g.safetensors` | `ComfyUI/models/text_encoders/` |
   | any SDXL VAE (`sdxl_vae.safetensors` / `vae.safetensors`) | `ComfyUI/models/vae/` |

   The GGUF is at `../hf-model/sdxl-unet-q8.gguf` in this bundle — copy it:

   ```bash
   cp ../hf-model/sdxl-unet-q8.gguf /path/to/ComfyUI/models/unet/
   # or download: https://huggingface.co/AEUPH/sdxl-unet-q8-gguf/resolve/main/sdxl-unet-q8.gguf
   ```

   `clip_l`/`clip_g` ship inside every SDXL checkpoint (e.g. extract with
   ComfyUI's `CheckpointLoaderSimple` → save, or grab the standalone files from
   `stabilityai/stable-diffusion-xl-base-1.0`).

## Run

1. Drag `sdxl-gguf-q8.json` onto the ComfyUI canvas (or *Workflow → Open*).
2. If a node shows red, re-pick the file names in the three loader nodes —
   widget values are machine-specific.
3. Queue prompt. Defaults: 1024×1024, 30 steps, CFG 7.0, euler/normal,
   denoise 1.0.

## Graph

```
UnetLoaderGGUF ──model──┐
DualCLIPLoader ──clip──┬─┴── CLIPTextEncode(pos/neg) ──cond──┐
                       └─────────────────────────────────────┤
VAELoader ──vae──────────────────────────────────────┐       │
EmptyLatentImage ──latent───────────────────────┐    │       ▼
                                                ▼    ▼   KSampler
                                                └──► latent
KSampler ──samples──► VAEDecode ◄──vae── VAELoader        (seed 42,
       │                                                   steps 30,
       └──► SaveImage                                      cfg 7, euler)
```

## Notes

- **VRAM**: quantized weights stay resident (≈3 GB) and dequantize per-op;
  measured 5.2 GB end-to-end on a 3090 (vs 15.4 GB for the fp8 checkpoint) —
  leaves headroom for larger batches / resolutions.
- **Q8_0 ≈ lossless** for SDXL; output matches the fp16 source within
  quantization noise.
- The file header carries `sdxl.unet_config` architecture JSON + the origin
  checkpoint path (`sdxl.*` KVs), so loaders can self-describe without a
  weight scan — see the main repo's `patches/`.
