# Modified InvokeAI 6.14.1 files

All diffs generated against stock `invokeai==6.14.1` (site-packages).
Originals are preserved in `orig/`.

| diff | file | what changed and why |
|---|---|---|
| `01-configs-main.py.diff` | `backend/model_manager/configs/main.py` | adds `Main_GGUF_SDXL_Config` + `_read_sdxl_gguf_header()`: classifies SDXL GGUFs from header KVs (`sdxl.unet_config`, `sdxl.source_checkpoint`) and an `add_embedding.` tensor-name check — no weight read during classification. |
| `02-configs-factory.py.diff` | `backend/model_manager/configs/factory.py` | imports the new config and adds it to the `AnyModelConfig` tagged union (tag `main.gguf_quantized.sdxl`). |
| `03-model_loaders-stable_diffusion.py.diff` | `backend/model_manager/load/model_loaders/stable_diffusion.py` | adds `StableDiffusionXlGgufModel` loader: UNet from the GGUF (quantized bytes resident, dequant-on-use), text encoders + VAE from the embedded source checkpoint with a portable fallback to any installed SDXL checkpoint. |
| `04-gguf-ggml_tensor.py.diff` | `backend/quantization/gguf/ggml_tensor.py` | `GGML_TENSOR_OP_TABLE` only mapped `aten.linear` on MPS; on CUDA every quantized `F.linear`/`conv2d`/`group_norm`/`layer_norm` raised `NotImplemented`. Adds a dequantize-and-run fallback for unknown ops (one warning per op name). |

Files are **modified versions** under Apache-2.0 §4 — original license text
shipped in `../LICENSE`, modification record in `../NOTICE`.
