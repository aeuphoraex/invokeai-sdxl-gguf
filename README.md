# InvokeAI SDXL-GGUF

Native SDXL **GGUF-quantized UNet** support for InvokeAI 6.x — self-describing
GGUF files, a new model config + loader, and a CUDA-safe dequant dispatch.
Install the model from a HuggingFace URL through the Model Manager, generate
images with the UNet resident at **~3 GB instead of ~7 GB** (measured end-to-end
VRAM: **5.2 GB vs 15.4 GB**).

```
GGUF file (Q8_0, 3.04 GB)
  ├─ quantized UNet weights           (tensors, plain diffusers keys)
  └─ self-describing header KVs
       ├─ sdxl.unet_config            (UNet2DConditionModel architecture JSON)
       ├─ sdxl.source_checkpoint      (path of origin checkpoint: TE + VAE)
       └─ quantization.type           (Q8_0)
                │
                ▼
Main_GGUF_SDXL_Config  ──classifies by header only (no weight read)──►
StableDiffusionXlGgufModel loader
  ├─ UNet        ← GGUF (quantized bytes stay resident, dequant-on-use)
  ├─ text encoders + VAE ← any installed SDXL checkpoint (portable fallback)
```

## Benchmarks (RTX 3090 24 GB, 1024×1024, 30 steps, CFG 7, euler)

| model | format | wall | compute | VRAM after |
|---|---|---|---|---|
| SDXL ckpt (fp8 storage) | checkpoint | 11.18 s | 9.76 s | 15,412 MB |
| **SDXL UNet Q8 GGUF**   | gguf_quantized | 17.52 s | 15.51 s | **5,172 MB** |

VRAM drops **74 %** (15.4 GB → 5.2 GB). Speed cost comes from the per-op dequant
fallback (every `aten.linear` dequantizes on the fly — no fused CUDA kernel);
see *Roadmap*.

## Install (as a patch)

```bash
python apply_patches.py            # auto-detects site-packages of this interpreter
# or
python apply_patches.py --site-packages /path/to/site-packages
```

The script is idempotent, backs originals up to `patches/orig/`, applies the
four diffs, py_compiles everything, and runs a registry smoke test
(`ALL GREEN` = loader registered). Then **restart InvokeAI**.

Patches:

| file | change |
|---|---|
| `configs/main.py` | `Main_GGUF_SDXL_Config` — classifies GGUFs by header KVs + `add_embedding.` tensor check |
| `configs/factory.py` | import + `AnyModelConfig` union entry |
| `load/model_loaders/stable_diffusion.py` | `StableDiffusionXlGgufModel` loader with portable source-checkpoint fallback |
| `quantization/gguf/ggml_tensor.py` | CUDA dequant fallback for `aten.linear` / `conv2d` / `group_norm` / `layer_norm` (upstream only mapped `linear` on MPS) |

## Install the model (HuggingFace URL)

Once the patch is applied, Model Manager → *Install from URL*:

```
https://huggingface.co/AEUPH/sdxl-unet-q8-gguf/resolve/main/sdxl-unet-q8.gguf
```

The classifier reads only the GGUF header (no weight scan), matches the
`sdxl.*` KVs, and registers it as `main.gguf_quantized.sdxl.normal`.
Text encoders + VAE are resolved from the embedded `sdxl.source_checkpoint`
path, falling back to **any installed SDXL single-file checkpoint** when that
path doesn't exist (e.g. installs on other machines).

## Convert your own checkpoint

```bash
python tools/sdxl_to_gguf.py --checkpoint /path/to/sdxl.safetensors --out sdxl-unet-q8.gguf
```

Quantizes every 2-D weight to Q8_0 (last-dim % 32 must hold; convs with
non-conforming shapes stay F16) and writes the `sdxl.*` self-description KVs.

## Layout

```
patches/           4 unified diffs + orig/ backups (generated against InvokeAI 6.14.1)
tools/             converter, patcher, bench harness, GGUF header dumper, remote-ops helpers
results/           bench JSONs + conversion/cluster logs
apply_patches.py   idempotent patch installer + smoke test
```

## Findings worth knowing

- **Q8_0 write path**: `gguf.quants.quantize()` works on matrices, but
  `GGUFWriter.add_tensor()` rejects uint8 — pass `raw_dtype=qt` so the writer
  accepts it and re-derives the dequantized shape.
- **Q4_K cannot be written** by the installed gguf-py (`quantize_blocks` raises
  `NotImplementedError`); reading is fine. Use llama.cpp or upgrade gguf-py.
- **Weight-space similarity does not index tensors cross-family** (cluster test
  v1+v2: at/below control) — bundle/model management must key on name + role
  metadata, which is exactly what the `sdxl.*` KVs provide.
- **GGUF v1** files use u32 KV lengths (v2+ use u64) — header dumpers must branch.

## Roadmap

- Row-block **sidecar patching** for LoRA on quantized modules
  (`force_sidecar_patching=True` — direct in-place add would materialize a
  second fp16 copy).
- Fused dequant+CUDA kernels (or cache the dequantized weights) to recover the
  ~1.6× speed gap.
- Deeper quant ladder: Q4_0 (write works today) → Q4_K via llama.cpp →
  AQLM/PVQ multi-vector → superposition/HDC bundling.
- Multi-component GGUF bundles (UNet + encoders + VAE + vision tower under
  `unet.*`/`text_encoder.*` prefixes, `bundle.components` KV).

## License

Apache-2.0 (same as InvokeAI). The model file on HuggingFace is released
under the license of its origin SDXL checkpoint.
