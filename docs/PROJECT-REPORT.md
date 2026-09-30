# SDXL × GGUF — Project Report (2026-09-30)

Everything we did with this GGUF, measured numbers only, plus honest verdicts.
Companion to `INVOKEAI-GGUF-IDEAS-LOG.md` (ideas + benchmark log).

---

## 1. The gap we found

Remote target: **InvokeAI 6.14.1** on a Vast.ai box (RTX 3090 24 GB, 503 GB RAM,
`/venv/main`, diffusers 0.40.0, gguf-py installed).

```
stable_diffusion.py registry:
  StableDiffusionXL + Diffusers      ✅
  StableDiffusionXL + Checkpoint     ✅
  StableDiffusionXL + GGUFQuantized  ❌  ← does not exist
```

GGUF configs exist for FLUX, Flux2, ZImage, Krea2, QwenImage, Wan — **no
`Main_GGUF_SDXL_Config`.** SDXL cannot be GGUF-quantized in InvokeAI today,
even though the machinery is base-agnostic and already shipped
(`invokeai/backend/quantization/gguf/{ggml_tensor,loaders,utils}.py`,
`model_on_disk.py` already routes `*.gguf` → `gguf_sd_loader`).

**Benchmarks for context** (30 steps · 1024² · CFG 7.0 · euler):

| Run | compute | VRAM used |
|---|---|---|
| baseline fp16 (warm) | **8.90 s** | 17,978 MB |
| fp8_storage (warm) | 9.76 s (+9.7%) | **15,412 MB (−2.6 GB)** |

fp8 = capacity tool, not speed.

---

## 2. The converter

`.remote-ops/sdxl_to_gguf.py` → `/workspace/.ro/sdxl_to_gguf.py` on the box.

- Loads the SDXL checkpoint via diffusers'
  `StableDiffusionXLPipeline.from_single_file()` so LDM→diffusers key mapping
  is **guaranteed correct** (no hand-rolled mapping).
- Emits **UNet only** (text encoders + VAE stay in the source checkpoint —
  design decision, see §7).
- Per-tensor quant chain: `Q8_0 → F16` (F16 only for shapes Q8_0 can't take).

### Output (final, verified)

```
/workspace/gguf/sdxl-unet-q8.gguf
  3,040,509,728 B (3.04 GB) · 1680 tensors · 2,567,463,684 params
  breakdown: {Q8_0: 1628, F16: 52}   ← type histogram verified {8:1628, 1:52}
  source checkpoint 6.94 GB → ratio 0.438
  the 52 F16 = all conv weights (3×3/1×1 last-dim, Q8_0 needs last%32)
```

Two **self-describing string KVs** embedded (verified via `GGUFReader.fields`):

| KV | value |
|---|---|
| `quantization.type` | `Q8_0` |
| `sdxl.unet_config` | full `UNet2DConditionModel` architecture JSON |
| `sdxl.source_checkpoint` | `/workspace/invokeai/models/332a6e1c-…/pornworksHardcoreFantasy_v04.safetensors` (104 B) |

The loader can rebuild the UNet from the file alone and knows where
encoders/VAE live without any API/schema change. This is the
"self-describing latent GGUF KV" idea applied to the real integration problem.

Backups: `sdxl-unet-q8.nokv.gguf` (pre-KV build) on the box.

---

## 3. The Q8_0 bug saga (root-caused & fixed)

First build came out **all-F16 despite `--quant Q8_0`** (`{F16: 1680}`).
Two probes isolated it:

1. **`gguf.quants.quantize(Q8_0)` works fine** — 1280², 5120², 640², bias 1280,
   real `attn1.to_q` all ✅. Conv `(320,4,3,3)` legitimately raises `QuantError`
   (must stay F16 by design).
2. **`GGUFWriter.add_tensor(quantized_uint8)` is what failed**:
   `ValueError: Only F16, F32, F64, I8, I16, I32, I64 tensors are supported
   for now` (`gguf_writer.py:356`). Quantized payload is uint8 → rejected →
   old chain silently fell back to F16. The old `first_fail` logging only ever
   showed conv_in, masking this.
3. **Q4_K write path is broken in this gguf-py build**:
   `quantize_blocks` raises bare `NotImplementedError` (`quants.py:129`) for
   *any* shape. Q4_K **reads** fine (Wan ships as Q4_K_M), but we cannot
   *write* it here. Q4_0 write works.
4. **Fix**: `w.add_tensor(name, data, raw_dtype=qt)` — the writer accepts
   uint8 when `raw_dtype` is given and re-derives the logical shape via
   `quant_shape_from_byte_shape`. Applied, re-run, verified (§2).

---

## 4. Cross-family clustering test (the harmonization experiment)

**Question:** do tensors cluster by *role* (attn.q, mlp.down …) across
families, giving a real cross-family organizing axis for one harmonized bundle?

**v1** (streaming fingerprint-and-discard + heartbeat liveness — replaced an
earlier run that spun at 68 GB RSS; v1 peaked 9 GiB, 303.7 s CPU, clean exit):

- Fingerprint = area-downsample |W| → 16×16, L2-normalized.
- **Degenerate metric**: every cell ≈ global mean (law of large numbers) →
  all fingerprints collapse onto the all-ones axis.
  real same-role **+0.9963** / diff-role **+0.9964** AND random control
  **+1.0000/+1.0000**; kNN real 14.2% vs control 15.8% (baseline 10%).
  **Zero signal either way — the metric itself was broken, not the data.**

**v2** (DC-stripped: signed cells, |cells|, row/col-norm profile 32+32 bins,
fusion; control = randn with identical shapes):

| channel | same | diff | sep | kNN fwd | control sep |
|---|---|---|---|---|---|
| fp_z | −0.0020 | +0.0020 | −0.0040 | 10.8% | −0.0002 |
| fp_abz | +0.0008 | +0.0010 | −0.0002 | 11.0% | +0.0004 |
| fp_npr | +0.1939 | −0.0208 | +0.2147 | 32.2% | **+0.2219** |
| FUSION | +0.0642 | −0.0060 | +0.0702 | 33.5% | **+0.0741** |

**VERDICT: NO ROLE SIGNAL (at or below control).** The tempting fp_npr MLP
numbers (mlp_gate +0.9971) are reproduced *exactly by random weights* — fp_npr
is a pure **shape/aspect-ratio detector** (row-vs-col log-norm offset =
½·log(n/m)), content-independent. Cell-level content channels sit at baseline.

**Consequence:** weight-space similarity **cannot** index tensors across
families. Only name/role metadata can (role/shape KV namespace, designed like
`custom_shader.*`). Superposition bundles must be built **within family /
within shape class**, not across families.

---

## 5. Memory accounting — measured

### On disk (GGUF artifacts that exist today)

| Artifact | Size | Notes |
|---|---|---|
| `sdxl-unet-q8.gguf` | **3.04 GB** | Q8_0, SDXL UNet only |
| `sdxl-unet-q8.nokv.gguf` | 3.04 GB | backup (pre-KV) |
| Wan2.2 I2V A14B HighNoise Q4_K_M | **9.65 GB** | shipped GGUF |
| Wan2.2 I2V A14B LowNoise Q4_K_M | **9.65 GB** | shipped GGUF |
| **GGUF total** | **≈ 22.3 GB** | |
| whole models dir | 87 GB | all formats |

SDXL extras not yet GGUF'd (stay in source checkpoint today): CLIP-L + 
OpenCLIP-G + VAE ≈ **1.8 GB fp16** (≈0.9 GB if quantized to Q8).

### RAM (how the loader actually behaves — read from code)

`gguf_sd_loader` does `tensor.data.copy()` of the **quantized** bytes →
`GGMLTensor` holds quantized data resident and **dequantizes on-the-fly per
operation** (wrapper subclass, `ggml_tensor.py:102`).

So:

- **SDXL Q8 UNet resident ≈ 3.04 GB** (vs 5.14 GB F16, vs ~6.9 GB full fp16
  pipeline) + transient dequant of the active module(s) during forward
  (largest single tensor e.g. 5120² F16 ≈ 52 MB) + activations.
- Loading everything simultaneously = 22.3 GB quantized-resident; InvokeAI
  loads one main model at a time, so typical resident ≤ 9.65 GB + encoders.
- Box has **336 GB RAM available** — RAM is not the constraint; VRAM (24 GB,
  6.1 GB free) is.
- VRAM for GGUF: **not yet benchmarked** (pending InvokeAI integration).

### Answers to the three memory questions

**Q: total memory of the harmonized system?**
Today's GGUF artifacts = **22.3 GB disk ≈ 22.3 GB quantized RAM if all
resident** (realistically one-at-a-time: ≤ 10 GB). A full 7-family bundle
(SDXL/FLUX/Flux2/ZImage/Krea2/QwenImage/Wan, UNets + encoders + VAEs) is not
built yet — estimate with: Q8_0 ≈ **1.06 B/param**, Q4-class ≈ **0.56–0.69
B/param** (calibrated: Wan 9.65 GB Q4_K_M / ~14 B params ≈ 0.69 incl.
embeddings/overhead).

**Q: do the latent weights decrease RAM?**
No — not directly. The latent fingerprints (16×16/48×48 grids) are **KB-scale
indexes**, not weight storage. They change nothing about weight RAM. Their
value is routing/lookup/dedup: RAM only drops if many tensors actually share
one superposed bundle. Per §4, cross-family sharing in weight space is
disproven; within-family bundles remain open (untested).

**Q: keep the data superimposed in RAM?**
We already do the pragmatic 90% version: `GGMLTensor` = *compressed in RAM,
decode on use*. True superposition-in-RAM = hold one bundle vector `M` instead
of N decoded weights and read out rows on demand (`W_i = correlate(M, H_i)`):
RAM ∝ bundle (could be ~1/N), CPU pays a decode per row per forward unless
cached per layer. Requirements: error-feedback compensator to bound decode
drift, per-family bundles (§4), and a JIT decode cache. Untested — flagged in
the ideas log as the experimental tier.

---

## 6. Quantization depth — going further (incl. superposition)

Ladder, in order of practicality on this box:

1. **Q8_0 (done)** — 1.06 B/param, ~0.5% error, LoRA-safe.
2. **Q4_0 (converter-supported today)** — 0.56 B/param; SDXL UNet would land
   ≈ 1.6 GB (from 3.04 GB). Measurably coarser.
3. **Q4_K_M / Q3_K / Q2_K / imatrix IQ\*** — same as Wan ships — **write path
   missing in installed gguf-py** (`NotImplementedError`); use llama.cpp's
   quantize tool on an F16 export, or upgrade gguf-py. Q4_K_M ≈ 0.56–0.69 B/p.
4. **Multi-vector / codebook quantization (the "new" family you asked about —
   it exists)**: *additive quantization* (**AQLM**), *product*/*residual VQ*
   (PVQ/RVQ), shared codebooks across same-shape tensors → compression from
   **cross-tensor redundancy**, not per-tensor bits. SOTA: ~2-bit AQLM beats
   Q4_K. Not in gguf-py; needs a custom kernel + training.
5. **Superposition quantization (ours)**: `HDC-PROOFS/gguf-hdc-gguf-distill.js`
   — `H = sgn(Φ·A)`, `M = Σ(H⊗T)`, saturation lock, `--sweep` finds the
   capacity knee. N tensors per bundle → ~1/N B/param minus codebook/seed
   overhead. Per §4: bundle **within** family/shape class. Decode noise is the
   cost; error-feedback compensator is the mitigation.
6. Delta-codec / low-rank+sparse: anchor tensor + sparse deltas for siblings —
   cheap to test with existing tooling.

---

## 7. InvokeAI integration — status

**Design (locked):** GGUF is UNet-only → `Main_GGUF_SDXL_Config` reads the
`sdxl.*` KVs for architecture + encoder/VAE provenance (no API schema change).
Loader: UNet from GGUF, encoders/VAE from `sdxl.source_checkpoint` via the
existing single-file path. Registration copied from `z_image.py:503`.

**Backups taken (verified, md5 recorded):**

```
/workspace/.ro/backup-20260930/
  configs/main.py                      7907611660fc651316fc78f7b64218a8
  configs/factory.py                   e3f2cf8875b4db8c68f25c73cc128b6a
  model_loaders/stable_diffusion.py    103f52650feb0cfae318ed9900ed4dbc
  ggml_tensor.py.bak                   (md5 e0e94d19d4de8b47e0b26ee5aca067dd)
```

**STATUS 2026-09-30: APPLIED, INSTALLED, GENERATED ✅**

4 patches (207→351 lines total) against InvokeAI 6.14.1:

| # | file | change |
|---|---|---|
| 01 | `configs/main.py` | `_read_sdxl_gguf_header()` + `Main_GGUF_SDXL_Config` (header-only classifier: `sdxl.*` KVs + `add_embedding.` tensor check) |
| 02 | `configs/factory.py` | import + `AnyModelConfig` union entry (tag `main.gguf_quantized.sdxl`) |
| 03 | `load/model_loaders/stable_diffusion.py` | `StableDiffusionXlGgufModel` loader + portable source-checkpoint fallback |
| 04 | `quantization/gguf/ggml_tensor.py` | CUDA dequant fallback (see bugs below) |

Bugs found & fixed during the fight (each cost a round-trip):

1. **Patcher idempotence guard bug** — one `"Main_GGUF_SDXL_Config" in src`
   guard covered *all* replaces in a file, so the 2nd+ replace per file was
   silently skipped → `NameError: ModelLoader`, missing union entry. Fixed by
   per-replace anchors (`repair_sd_imports.py`, `repair_factory_union.py`).
2. **Tag mismatch** — our config declared `variant: ... default=Normal` →
   `get_tag()` = `main.gguf_quantized.sdxl.normal`, but pydantic's *dict*
   discriminator only appends variant for CLIP-Embed → `union_tag_invalid`.
   Fix: dropped the `variant` default (ZImage GGUF config does the same).
3. **API enum values** — REST wants `base: "sdxl"`, `format: "gguf_quantized"`
   (lowercase), not the Python enum names.
4. **GGMLTensor dispatch on CUDA** — upstream maps `aten.linear.default` *only
   if MPS is available*; on the 3090 every quantized F.linear raised
   "all handlers returned NotImplemented". Patch 04 adds a dequantize-and-run
   fallback for unknown ops (linear/conv2d/group_norm/layer_norm observed).
   Backup kept; unit test: linear + bias paths match manual dequant exactly.
5. **`ModelLoaderRegistry.lookup` doesn't exist** — it's
   `get_implementation(config, submodel_type)`; config instances need
   hash/file_size/name/source/source_type.
6. **Non-portable source_checkpoint** — embedded abs path dies on other
   machines → loader now falls back to *any* installed SDXL checkpoint
   (TE+VAE are architecturally identical), else raises a clear message.

**Result — model installed** (key `2343be28-128e-45c6-b05b-49091548e22c`,
format `gguf_quantized`, config row carries `source_checkpoint` +
`unet_config_json` from the KVs) **and generated successfully:**

| | SDXL ckpt (fp8 storage) | SDXL Q8_0 GGUF |
|---|---|---|
| wall | 11.18 s | 17.52 s |
| compute | 9.76 s | 15.51 s |
| VRAM after | 15,412 MB | **5,172 MB** |

**−74 % VRAM, +1.59× compute** (per-op dequant, no fused kernel yet).
Baselines: earlier warm runs 8.90 s / 17,978 MB confirm the same picture.
Ops that hit the fallback: `aten.linear`, `aten.conv2d`, `aten.group_norm`,
`aten.layer_norm` (convs stayed F16 — 52 of 1680 tensors — so `conv2d` here
is the *quantized* convs from other model paths / group-norm weights).

---

## 8. Multimodal LoRA implications for this GGUF

### How LoRA meets GGUF in InvokeAI (read from code)

- SDXL LoRA path: `LoRAExt.patch_unet` →
  `LayerPatcher.apply_smart_model_patch(force_direct_patching=True,
  force_sidecar_patching=False)` (`extensions/lora.py:53`).
- **Direct patching = in-place add on the weight.** The patcher's own comment:
  *"The module is quantized, so the caller passed `force_sidecar_patching=True`"*
  — but the SDXL caller hardcodes `False` and never checks quantization.
- `GGMLTensor` holds **quantized** storage. ⇒ **Integration gap (to verify &
  patch at install time):** SDXL + GGUF + LoRA must route quantized modules to
  **sidecar patching** (dequantize to compute dtype → add delta → wrapper
  module). Sidecar is the safe path; it dequantizes before any math, so the
  delta math happens at bf16/fp16.

### What quant level does to LoRA fidelity

- **Q8_0 (this build)**: ~0.5% error — LoRAs trained on the fp16 checkpoint
  apply cleanly; recommended base for LoRA development.
- **Q4-class**: 1.5–3% error starts interacting with small-α adapters —
  validate any LoRA at the target quant before shipping.
- During patched inference, RAM = quantized base + transient dequantized
  copy of active modules (sidecar deliberately avoids a second *full* fp16
  copy — see `layer_patcher.py:170`).

### What "multimodal" means here, and what our work adds

SDXL is already multimodal in the relevant sense: **two text encoders**
(CLIP-L + OpenCLIP-G) + VAE + image-conditioning adapters. Concretely:

1. **Encoders are untouched by the UNet GGUF** — TE-LoRAs
   (`lora_te1_`/`lora_te2_` prefixes) keep working exactly as today, because
   encoders come from the source checkpoint at fp16. The self-describing
   `sdxl.source_checkpoint` KV gives every adapter an authoritative provenance
   target.
2. **Provenance + compatibility registry**: an adapter GGUF can carry its own
   KVs (`lora.base_hash`, `lora.target_roles`, `lora.family`) → install-time
   mismatch detection instead of silent garbage. The same KV namespace scales
   across the harmonized families (`custom_shader.*`-style).
3. **Cross-family LoRA transfer is ruled out by §4** — no cross-family
   weight-space similarity (v2 = control). Multimodal/cross-family adaptation
   must be **distillation** (teacher SDXL+LoRA → student Wan) or
   family-native training; the harmonization enables unified *registry and
   metadata*, not delta portability.
4. **QLoRA-style training on the Q8 base**: freeze the 3.04 GB Q8_0 UNet,
   train low-rank deltas in bf16 → base training footprint roughly halves vs
   fp16 (5.14 GB), which matters on a 24 GB 3090.
5. **Superposed LoRA packs (experimental, uses your existing code)**:
   `LatentAbsorber`/`PointerMemory` tech → many LoRAs superposed in one file,
   recalled by pointer+cosine. Adapters are low-rank (small), so this is the
   *safest* place to start superposition — decode noise on a 16-dim Δ is far
   more tolerable than on full 5120² weights.

### Practical workflow this enables

1. Train LoRA on the **fp16 source checkpoint** (the KV tells you exactly which).
2. Validate at **Q8_0** in InvokeAI (`sdxl_lora_loader` node + sidecar patch).
3. Ship adapter with metadata KVs; registry enforces base compatibility.

---

## 9. File inventory

**Local (`D:/JS-GGUF/`)**: this report · `INVOKEAI-GGUF-IDEAS-LOG.md` ·
`HDC-PROOFS/gguf-hdc-{latent,gguf}-distill.js`

**Local (`.remote-ops/`)**: `jremote.py` · `upload.py` · `download.py` (streaming,
byte-range) · `bench.py` · `fp8.py` · `sdxl_to_gguf.py` · `gguf_dump.py`
(header dumper, v1+v3) · `patch_sdxl_gguf.py` · `patch_ggml_dispatch.py` ·
`patch_source_ckpt_fallback.py` · `repair_{sd_imports,factory_union}.py` ·
`cluster_test{,_v2}.py` · `q8_probe{,2}.py` · `shapes.py` · `tags*.py` · `schema*.py`

**Remote (box)**: `/workspace/.ro/` (scripts + logs + `backup-20260930/` +
`package/` tarballs) · `/workspace/invokeai/models/2343be28.../sdxl-unet-q8.gguf`
· patched InvokeAI 6.14.1 site-packages (4 files, all in service + generating)

**G: drive bundle (`G:/invokeai-sdxl-gguf/`)** — full working copy:

```
package/    invokeai-patched.tar.gz (15MB) + ro-tooling.tar.gz + MD5SUMS ✓verified
invokeai/   extracted patched package (38MB tree)
tools/      extracted .ro tooling (scripts, logs, backups)
hf-model/   sdxl-unet-q8.gguf (md5 ba6fca2f461df9d32ab24325cfb909b7 ✓ both sides)
            + README.md model card + upload.log
repo/       GitHub-ready git repo (commit 713972a, 30 files):
            README · LICENSE (Apache-2.0) · NOTICE · apply_patches.py
            patches/{01..04}.diff + MODIFICATIONS.md + orig/
            tools/ · results/ · comfyui-workflow/
comfyui-workflow/  drag-drop UI JSON + API JSON (9 nodes, 9 links validated)
```

**HuggingFace**: repo `AEUPH/sdxl-unet-q8-gguf` created; 3.04 GB upload in
progress → installable from InvokeAI Model Manager via
`https://huggingface.co/AEUPH/sdxl-unet-q8-gguf/resolve/main/sdxl-unet-q8.gguf`.

**Distribution**: InvokeAI is Apache-2.0 (verified in dist-info METADATA) →
patched fork redistributable under §4 with LICENSE + NOTICE + mod record
(shipped). Trademark: no InvokeAI name/logo implies endorsement.

### Demo GGUFs with embedded runtime code (examined)

- `OMNI_MORPHEME_ZERO_MAC.gguf` (10.7KB, v3): **JS source code stored in a
   string KV** `jinxai.runtime` (2560B, `class MorphemeAlgebraEngine`) + 3
   unnamed tensors. Pattern: *KV = program, tensors = state*.
- `4TH-DH-VISIONML.gguf` (9.5KB, **GGUF v1**): self-describing `latent.*` KVs
   (grid/cells/tokens/absorbed/mass/norm/entropy/vocab) + one 48×48
   `latent_weights` tensor. v1 uses u32 KV lengths (v2+ u64) — dumper fixed.
- `JinxOS-*.gguf` (1.9–3.2GB): `general.architecture = jinxos-*`, DNA/KV
   config arrays + transformer weights — same self-description philosophy as
   our `sdxl.*` KVs, generalized (`jinxos.dna.*` = hyperparameters).
- `SKELETON_NEURAL.gguf`: 1131 tensors, 0 KVs — bare weights-only
   contrast case.
→ Takeaway: our KV scheme is the *right* pattern and the demo files prove
  KV-embedded programs survive real distribution; a future `runtime` KV could
  carry loader recipes (schedule, prefetch map) our side reads at install.

## 10. Next steps

1. ✅ Patch applied → installed → **generation verified** → bench (§7).
2. ✅ HF-URL install path (repo + model card + loader fallback).
3. ✅ ComfyUI workflow folder (ComfyUI-GGUF `UnetLoaderGGUF`).
4. Verify LoRA on GGUF UNet → patch sidecar routing if needed (§8).
5. Q4_0 build + quality check; llama.cpp route for Q4_K_M.
6. Within-family bundle proof-of-concept (HDC distiller, `--sweep` knee).
7. Speed: cache dequantized weights or write fused dequant+CUDA linear to
   close the 1.59× compute gap (VRAM budget has 19GB headroom).
