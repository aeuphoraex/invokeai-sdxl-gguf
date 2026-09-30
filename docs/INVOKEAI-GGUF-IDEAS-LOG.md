# JS-GGUF → InvokeAI Ideas Log

**Date:** 2026-09-30
**Source:** `D:\JS-GGUF` (full survey of ANALYSIS/audit docs + on-disk artifacts)
**Target:** InvokeAI hosted on Jupyter (`151.237.25.16:26112`)
**Purpose:** Identify novel + high-efficiency ideas worth porting, and rate them honestly before spending test time.

---

## READ THIS FIRST — honesty rating

The repo contains excellent **self-audits** (`ANALYSIS.TERNARY.MD`, `*/ANALYSIS.MD`,
`GGUF-INFERENCE-AUDIT.md`). Those audits are accurate and I verified their key claims:

| Claim in audits | My verification | Result |
|---|---|---|
| "AVX2 / SIMD / sub-100ns names are branding, bodies are scalar JS" | `grep -rlE "wasm_simd128\|v128\.\|i32x4\|f32x4\|__m256\|__m128\|asm\.js" --include=*.js` | **0 files** repo-wide |
| BitNet WASM is invalid (110 B) | `.wasm` files found: only `gguf-bitnet-kernels.wasm` + a sandbox copy | consistent with audit |
| "90 JINXAI model components documented" | `MODELS/` contains **1** `.js` model file | catalog is aspirational |
| No InvokeAI work exists yet | `grep -ril invokeai --include=*.md` | **0 hits** |

**Score key** — `READY` (real, test soon) · `PORT` (sound concept, needs real impl) ·
`SKIP` (branding / broken, do not spend time).

---

## TIER A — READY: real code, direct InvokeAI value

### A1. GGUF ↔ SafeTensors bridge  ★ the compatibility doorway
- **Where:** `GGUF-SURGERY/gguf-safetensors-writer.js` (+ `gguf-core.js`)
- **What it really does:** Reads GGUF *or* safetensors, writes `.safetensors`.
  Dequantizes `Q4_0..Q6_K` → F16/F32/BF16. Supports `--tensors`, `--exclude`,
  `--rename`, `--dtype`, `--align`.
- **Why it matters:** InvokeAI ingests **safetensors**, not GGUF. This is the single
  most valuable file in the repo — it is the conversion doorway that makes *every*
  other GGUF idea testable on InvokeAI at all.
- **InvokeAI test:** convert `WAN 4QUANT.gguf` → `.safetensors`, drop into InvokeAI
  model manager, confirm load + a smoke generation.
- **Risk:** dequant of block-quantized tensors is lossy-by-design; check output quality.

### A2. `WAN 4QUANT.gguf` — the one large real artifact
- **Where:** `D:\JS-GGUF\WAN 4QUANT.gguf` — **3,433,116,000 B (3.43 GB)**
- **Why it matters:** by far the largest GGUF in the repo; WAN is an actual
  image/video arch InvokeAI supports. Everything else in the repo is KB-scale stubs.
- **InvokeAI test:** inspect tensor inventory + quant types → A1 conversion → run.

### A3. FLUX / SD3.5 LoRA GGUF writer (comfyui_kohya format)
- **Where:** `TINY-LORA/gguf-tiny-flux-lora.js` · artifact `TINY-LORA/tiny_flux_lora.gguf` (36,179,392 B)
- **What it really does:** Emits structurally valid LoRA GGUFs with a correct
  Flux/SD3.5 double-block module table (`img_attn_qkv` 3072→9216, `img_mlp_0`
  3072→12288, `img_mod_lin` 3072→18432, txt_* mirrors), `lora_unet_*` tensor names,
  rank/alpha scalars, BF16/F16/F32 dtypes.
- **Why it matters:** real, correct structure — not branding. LoRA is InvokeAI's
  fastest-moving surface.
- **InvokeAI test:** `gguf-safetensors-writer.js` → FLUX LoRA in InvokeAI.

### A4. FLUX hybrid variants — flow / consistency / score axes
- **Where:** `FLUX-HYBRID-VARIANTS/gguf-flux-hybrid-variants.js`
- **The three axes (real ML, not branding):**
  - **F** = flow matching / rectified **reflow** → best speed:quality
  - **C** = **consistency / 1–4 step distillation** → cheapest per output
  - **S** = score/SDE + latent prior (VAE/hierarchical) → most controllable
- **Honest caveat:** tensors are deterministic *seeded placeholders*; the value is
  the **recipe carried as KV metadata**, not trained weights.
- **Why it matters:** step distillation is the single biggest speed lever in
  diffusion. A 4-step model is ~10× cheaper than a 30-step one at equal class.
- **InvokeAI test:** apply the C-axis recipe to InvokeAI schedulers; compare
  steps-to-convergence vs baseline at fixed CFG.

### A5. Chunked GGUF V2/V3 loader + layer router + worker pool
- **Where:** `NEURAL LATENT PIPELINE ENGINE/`
  - `gguf-chunked-loader.js` — `readGGUFHeaderChunk`, `readGGUFDirectoryChunk`,
    `groupTensorsIntoLayers`, `inferBufferForTensor`, `tensorRange`
  - `gguf-layer-router.js`, `gguf-inline-worker-host.js`
- **Architecture:** header (24 B) + KV + tensor directory parsed in main thread;
  **per-tensor `ArrayBuffer.slice()`** copies dispatched to workers. Zero whole-file
  load.
- **Why it matters:** streamed/lazy weight loading → lower peak RSS, faster startup,
  enables models larger than RAM. Directly analogous to InvokeAI's model-manager
  offload path.
- **InvokeAI test:** prototype lazy per-tensor load; measure cold-start + RSS.

---

## TIER B — PORT: novel concept, needs a real implementation

### B1. Custom Shader Protocol — model file carries its own kernels  ★ most novel
- **Where:** `CUSTOM-SHADER-PROTOCOL/protocol-spec.md`, `embedding-methods.md`, `SHADER-LIBRARY/`
- **What it proposes:** extend GGUF v3 with a `custom_shader.` KV namespace:
  - GLSL compute/vertex/fragment → KV strings (`custom_shader.glsl.<name>`)
  - **SPIR-V binary → stored as `I32` (gType 27) *tensor*, `[word_count]`**
  - WGSL (`custom_shader.wgsl.*`) + MSL (`custom_shader.msl.*`)
  - protocol registration via `custom_shader.meta.protocol_version = "1.0.0"`
- **Why it matters:** genuinely novel — a checkpoint that transports its own fused
  kernels alongside its weights. Would let an InvokeAI model ship custom ops.
- **Honest caveat:** spec-level. `STATUS.md` claims the shader library is "COMPLETED"
  but I have not verified any of it compiles. **Verify one shader end-to-end first.**
- **InvokeAI test:** take one `SHADER-LIBRARY/compute/` kernel, compile → SPIR-V →
  confirm it runs as a PyTorch custom op. If that fails, drop B1.

### B2. Ternary / BitNet b1.58 packing + LUT-GEMV ("zero-multiplication")
- **Where:** `ANALYSIS.TERNARY.MD` + 13 files across `HYPERTHEORY`, `HYPER-PRECISION`,
  `AVX2-OPTIMIZED`, `VARIANTS`, `V2`, `ADDINS`
- **The sound idea underneath:** 2-bit `{-1,0,+1}` weights, 4 values/byte, evaluated
  by a **precomputed look-up table** so the inner loop does *adds and table reads
  instead of multiplies* (T-MAC-style LUT-GEMV). This is a real, published technique
  with genuine CPU latency/energy wins — **especially relevant for InvokeAI's CPU
  offload / low-VRAM path.**
- **Honest caveat:** current implementations are **not usable as-is**:
  - 6 confirmed correctness bugs (truncation, infinite loop, asymmetric scale,
    0.5-threshold, 4× bloat, Markov desync)
  - 10+ independent re-declarations of the same codec, no canonical module
  - 0 SIMD, 0 tests, 0 measurements; "16:1 vs Q4_K" unverified
- **InvokeAI test:** port the *LUT-GEMV idea* to Python/C, benchmark vs `Q4_K` on
  identical weights. Do **not** port the JS.

### B3. Quantization error-feedback compensator  ← real code, underused idea
- **Where:** `GGUF-SURGERY/gguf-quantization-error-feedback-compensator.js`
- **What it really does:** quantizes F32 → N-bit, computes per-tensor mean error,
  stores it back as KV `jinxai.error_bias_<tensor>`, emits compensated GGUF.
- **Why it matters:** idea is sound and cheap — persist quant-error statistics as
  metadata so the *consumer* can compensate at dequant time. Naive today (uniform
  scalar quant, mean-bias only) but a clean extension point.
- **InvokeAI test:** does applying the stored bias measurably improve PSNR vs
  naive dequant? Cheap to test, potentially free quality.

### B4. Tiered cache L1/L2/L3 (SRAM / DDR / VRAM, LRU)
- **Where:** `vram_cache/vram_cache.js` — L1 64 MB SRAM / L2 256 MB DDRAM /
  L3 dynamic VRAM, LRU eviction, priority weights, 60 FPS frame budget.
- **Why it matters:** the *tiering policy* maps cleanly onto InvokeAI's
  load/offload/unload model-management tiers.
- **Honest caveat:** it is a **disk-backed simulation**, not real VRAM management.
  Port the policy, not the code.

### B5. HDC — hyperdimensional computing (most verified subsystem here)
- **Where:** `HDC-PROOFS/` — `_lib/hdc-lib.js` + 12 method tools
- **What is genuinely verified:** round-trip GGUF v3 codec debugged; bipolar recover
  fidelity **1.0000**, FHRR unbind **1.0000**; all 12 tools report `ok=True`
  (HRR style algebra, compositional scene synthesis, one-pass few-shot style
  learning, conformal sets, EP-HDC privacy, temporal memory, algebraic video,
  zero-step generation, federated, neuromorphic, crossmodal text+audio+image).
- **Why it matters:** deterministic ±1 hypervectors with XOR bind / majority
  bundle / popcount similarity — extremely cheap. Candidate for **fast conditioning
  indexing, prompt/asset dedupe, and crossmodal retrieval** around InvokeAI.
- **InvokeAI test:** use HDC similarity as a cheap pre-filter for prompt/LoRA lookup.

### B6. GGUF surgery toolbox (verify before use)
- `gguf-kv-cache-quant-injector.js` — inject quantization into KV cache
- `gguf-mixed-precision-quant-scheduler.js` — per-layer precision scheduling
- `gguf-quantized-cache-lru-evictor.js` — cache eviction
- `gguf-quant-block-layout-reorder.js` — block layout reorder (memory locality)
- `gguf-quantization-error-feedback-compensator.js` — see B3
- `gguf-attention-head-pruning-stitch.js` — attention head pruning
- `gguf-bench.js`, `gguf-core.js` (the reliable reader/writer, ~70% complete)
- **Status:** names are purposeful (unlike the ternary family), but I have not
  verified behavior. **Run each on a small GGUF and check round-trip before trusting.**

---

## TIER C — SKIP: branding or broken. Do not spend test time.

| Item | Evidence |
|---|---|
| All "AVX2 / SIMD / sub-100ns / sub-µs" filenames | **0 SIMD markers** repo-wide (grep verified) |
| `gguf-bitnet-kernels.wasm` (110 B) | fails `WebAssembly.validate()` per audit |
| Ternary family as-is | 6 confirmed correctness bugs, no canonical codec, no tests |
| "90 JINXAI model components" | 1 actual file on disk |
| `quantized_flux-pro.gguf` (1.6 MB), `quantized_midjourney-v7.gguf` (813 KB), `quantized_nano-banana.gguf` (613 KB) | far too small to be real models — metadata stubs, **not** runnable checkpoints |
| `VARIANTS/*simd*`, `*hyper-bitpack*` | scalar loops; one has a 4× bloat bug, one an infinite loop |

---

## Suggested test order on InvokeAI

1. **A1** — prove the GGUF→safetensors bridge works. Everything else depends on it.
2. **A2** — `WAN 4QUANT.gguf` through A1 → first real generation.
3. **A3** — FLUX LoRA conversion → load.
4. **A4** — step-distillation recipe vs baseline (biggest expected speed win).
5. **B3** — error-feedback compensator (cheapest quality experiment).
6. **A5** — chunked loader (startup / RSS win).
7. **B1** — one shader compiled end-to-end, then decide.
8. **B2** — LUT-GEMV ported to Python, benchmarked against Q4_K.

---

# RESULTS LOG (2026-09-30 session)

## Target actually reached

InvokeAI is a **Vast.ai** container, not the local PC.

| Item | Value |
|---|---|
| InvokeAI | **6.14.1**, `127.0.0.1:9090` internal, external `http://151.237.25.16:26114/` |
| GPU | RTX 3090 24 GB (only **~6–9 GB free** while SDXL resident) |
| Env | `/venv/main`, Python 3.12, torch CUDA 12.8, diffusers 0.40.0, `gguf` installed |
| API | 145 REST paths, 877 schemas, 303 invocations |
| SDXL models | 4 mains @6.94 GB (2 checkpoint, 2 diffusers) + 1 refiner + 2 lycoris LoRAs |
| Remote shell | `.remote-ops/jremote.py` (Jupyter terminal WebSocket) |
| Script upload | `.remote-ops/upload.py` (Jupyter contents API PUT) |

> **Gotcha:** Git Bash mangles POSIX args (`/venv/...` → `C:/Program Files/Git/...`).
> Always prefix remote commands with `MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*'`.

## Benchmarks (30 steps · 1024×1024 · CFG 7.0 · euler · SDXL checkpoint)

| Run | compute_s | wall_s | VRAM used | VRAM free | Notes |
|---|---|---|---|---|---|
| baseline (warm) | **8.90** | 9.05 | 17,978 MB | 6,149 MB | fresh prompt |
| baseline (cold) | 0.72 | 2.85 | — | — | **session-cache hit, invalid** |
| fp8_storage (cold) | 15.12 | 16.14 | 15,106 MB | 9,021 MB | includes forced cache eviction |
| **fp8_storage (warm)** | **9.76** | 11.18 | **15,412 MB** | 8,715 MB | fair comparison |

### fp8_storage verdict
- **−2,566 MB VRAM** (17,978 → 15,412, **−14 %**)
- **+9.7 % slower** (8.90 → 9.76 s)
- Toggle: `PATCH /api/v2/models/i/{key}` with `default_settings.fp8_storage=true`; InvokeAI
  auto-evicts the model cache (`_LOAD_AFFECTING_SETTINGS`).
- Use it for **capacity**, not speed.

> Benchmarking trap: InvokeAI caches by node inputs. **Always change the prompt/seed**
> or you will measure a cache hit, not generation.

## GGUF × SDXL — the headline gap (CONFIRMED)

```
stable_diffusion.py registry:
  StableDiffusionXL + Diffusers      ✅
  StableDiffusionXL + Checkpoint     ✅
  StableDiffusionXL + GGUFQuantized  ❌  NOT registered
```
GGUF configs exist for **FLUX, Flux2, ZImage, Krea2, QwenImage, Wan** —
**there is no `Main_GGUF_SDXL_Config`.** SDXL cannot be GGUF-quantized today.

But the machinery is base-agnostic and already shipped:
- `invokeai/backend/quantization/gguf/{ggml_tensor.py,loaders.py,utils.py}`
- `model_on_disk.py:140` already routes `*.gguf` → `gguf_sd_loader(path)`
- Pattern to copy: `z_image.py:503` (`@ModelLoaderRegistry.register(... GGUFQuantized)`)

### Why SDXL is harder than Wan/ZImage
SDXL single-file checkpoints bundle **everything**:
- `model.diffusion_model.*` (1680 tensors) — the UNet
- `conditioner.*` (587) — CLIP-L + OpenCLIP-G
- `first_stage_model.*` (248) — VAE

`StableDiffusionDiffusersModel._load_from_singlefile` calls diffusers'
`from_single_file()` on the whole file. A GGUF holds **UNet only**, so
`Main_GGUF_SDXL_Config` needs a **second path** to the source checkpoint for
text encoders + VAE. That is the missing design decision.

## Converter built: `.remote-ops/sdxl_to_gguf.py`

Deployed to `/workspace/.ro/sdxl_to_gguf.py`. Uses diffusers' own
`StableDiffusionXLPipeline.from_single_file()` so LDM→diffusers key mapping is
guaranteed correct (no hand-rolled mapping).

**Output: `/workspace/gguf/sdxl-unet-q8.gguf`** — 5,135,085,536 B, **1680 tensors,
verified readable by `GGUFReader`**. Source checkpoint 6.94 GB → ratio 0.740.

### Q8_0 bug — ROOT CAUSE FOUND & FIXED (q8_probe.py / q8_probe2.py)

v1 output was **all-F16 despite `--quant Q8_0`** (breakdown `{F16:1680}`).
Two probes isolated it:

1. `gguf.quants.quantize(Q8_0)` **works**: 1280²✅ 5120²✅ 640²✅ bias✅,
   real `attn1.to_q` ✅. Conv `(320,4,3,3)` legitimately fails (`QuantError`) —
   must stay F16 by design.
2. `GGUFWriter.add_tensor(quantized_uint8)` **fails**:
   `ValueError: Only F16, F32, F64, I8, I16, I32, I64 tensors are supported
   for now` (`gguf_writer.py:356`). Quantized payloads are uint8 → rejected
   → old chain caught it → silent F16 fallback. Old `first_fail` logging
   only ever showed conv_in, masking this.
3. **Q4_K is unusable in this gguf-py build**: `quantize_blocks` raises bare
   `NotImplementedError` (`quants.py:129`) for *any* shape — no K-quant write
   path exists here. Q4_0 works on matrices (needs last%32? no — last%8).
4. **Fix**: `w.add_tensor(name, data, raw_dtype=qt)` — writer accepts uint8
   when `raw_dtype` is given, re-derives logical shape via
   `quant_shape_from_byte_shape`.  Converter patched accordingly.

**FIX VERIFIED — real Q8_0 output produced:**

```
[quant] breakdown: {'F16': 52, 'Q8_0': 1628}     # was {F16: 1680}
[done] /workspace/gguf/sdxl-unet-q8.gguf
       tensors=1680 params=2,567,463,684
       src=6.94 GB  dst=3040.5 MB   ratio=0.438   # was 5.14 GB / 0.740
```
1628/1680 tensors Q8_0; the 52 F16 are all conv `(C,C,3,3)`/`(C,C,1,1)` —
legitimately unquantizable (Q8_0 needs last%32), every logged fallback is a
`QuantError` on a conv. File size halved vs the F16 mistake.

## Cross-family clustering test (cluster_test.py → v2)

**v1 (fingerprint-and-discard streaming, heartbeat liveness):**
- Ran clean 303.7 s CPU, RSS peaked 9 GiB (previous run OOM-spun at 68 GiB).
- **Degenerate metric**: area-downsample of `|W|` → every cell ≈ global mean
  (LLN) → all fingerprints collapse to all-ones axis.
  real same +0.9963 / diff +0.9964 **AND random control +1.0000/+1.0000**;
  kNN real 14.2% vs control 15.8% (baseline 10%) → **zero signal either way**.

**v2 (DC-stripped: fp_z signed cells, fp_abz |cells|, fp_npr log row/col-norm
profile 32+32 bins, fusion; control = randn same shapes):**

| channel | same | diff | sep | kNN fwd/rev | control sep |
|---|---|---|---|---|---|
| fp_z | −0.0020 | +0.0020 | −0.0040 | 10.8%/12.5% | −0.0002 |
| fp_abz | +0.0008 | +0.0010 | −0.0002 | 11.0%/5.8% | +0.0004 |
| fp_npr | **+0.1939** | −0.0208 | **+0.2147** | **32.2%**/3.0% | **+0.2219** |
| FUSION | +0.0642 | −0.0060 | +0.0702 | 33.5%/5.0% | +0.0741 |

**VERDICT: NO ROLE SIGNAL (at or below control).** The tempting fp_npr
MLP numbers (mlp_gate +0.9971, mlp_down +0.9876) are **reproduced exactly by
random weights** — fp_npr is a pure *shape/aspect-ratio* detector: the
row-vs-col log-norm offset is 0.5·log(n/m), independent of content.
Cell-level content channels sit at baseline. **Weight-space similarity
cannot index tensors across families; only name/role metadata can.**
Role/shape KV namespace design (custom_shader-style) stands.

### OPEN BUG — quantization did not engage
Per-tensor breakdown: `{F16: 1680}` — every tensor fell back to F16, so the
file is a **valid but unquantized** F16 GGUF (5.14 GB ≈ 2.57 B params × 2 B).

Isolated tests on this box (`gguf-py`):

| shape | Q4_K | Q8_0 |
|---|---|---|
| (1280,1280) | FAIL (empty msg) | **OK** → (1280,1360) |
| (640,1280) | FAIL | **OK** |
| (320,320) | FAIL | **OK** |
| (1280,320) | FAIL | **OK** |
| (320,4,3,3) conv | FAIL | FAIL (last dim 3, needs %32) |

- gguf-py quantizes along the **LAST** axis: Q8_0 needs `last % 32`, Q4_K needs `last % 256`.
- **Q4_K is unusable in this build** (fails even on (1280,1280)). Use **Q8_0**.
- Conv tensors legitimately stay F16 — but matrix weights should have taken Q8_0 and did not.
  Root cause **not yet found**; next step is to log the *actual* exception for a
  known-good shape like `model.diffusion_model.*.attn1.to_q.weight` inside the
  converter (the `first_fail` capture only records the first failure, `conv_in`,
  which masks the real reason for the others).
- Writer API in this gguf-py: `write_header_to_file()`, `write_kv_data_to_file()`,
  `write_tensors_to_file()` — **not** `write_header()`.
- `gguf.quants.quantize()` takes **numpy**, not torch; pass `.numpy()` (and a
  fresh copy per attempt).

## Remaining work (in order)
1. Fix Q8_0 fallback so matrix weights actually quantize (~2.6 GB → ~1.3 GB).
2. Add `Main_GGUF_SDXL_Config` to `configs/main.py` (needs `source_path` design).
3. Register `StableDiffusionXL + GGUFQuantized` in `stable_diffusion.py`.
4. Build UNet from `gguf_sd_loader()`; text encoders/VAE from `source_path`.
5. Install GGUF model, benchmark vs safetensors baseline **8.90 s / 17,978 MB**.
6. Not started: error-feedback compensator port, step-distillation custom node.

> InvokeAI has **not** been patched yet. The only change made to it is the
> reversible `fp8_storage` model setting on `332a6e1c-…` (bench used neutral prompts).

## Environment notes

- Preflight now **4/4 green**: `AGENT_SHELL_SKILL_FILE` ✅ · `AGENT_SHELL_RESONANCE_HTML`
  (`file:///D:/TENSORMATRIX/index.html`) ✅ · `AGENT_SHELL_SKILLS_ROOT` ✅ · `JS_GGUF_TOOLS_ROOT` (`D:/JS-GGUF`) ✅
- Local stack: Python 3.14.2 (`C:/Python314`), torch **2.11.0+cu126**, diffusers 0.36.0,
  `invoke` shim at `AppData/Roaming/Python/Python314/Scripts/invoke.exe`.
  The `invokeai` package did **not** resolve from `C:/Python314` — confirm which
  interpreter/env actually hosts it before testing.
- `robocopy` is banned in `~/.agents/AGENTS.md`. Use `cp -a` / `rsync -a` and verify.
