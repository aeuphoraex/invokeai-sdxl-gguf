FROM python:3.12-slim-bookworm

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# patch(1) for apply_patches.py; curl for healthchecks; opencv runtime libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        patch curl libgl1 libglib2.0-0 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# InvokeAI 6.14.1 == the version our diffs were generated against.
# Default (CUDA) torch so any Space hardware tier works without a rebuild.
RUN pip install invokeai==6.14.1

# --- apply the SDXL-GGUF patches (idempotent, self-verifying) -------------
COPY patches /opt/sdxl-gguf/patches
COPY apply_patches.py /opt/sdxl-gguf/apply_patches.py
RUN python /opt/sdxl-gguf/apply_patches.py

# --- bake models ----------------------------------------------------------
# 1) The self-describing Q8_0 UNet GGUF (moved into the model registry by the
#    entrypoint on first boot; install MOVES the source file).
RUN python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download('AEUPH/sdxl-unet-q8-gguf', 'sdxl-unet-q8.gguf', local_dir='/opt/models')"

# 2) Text encoders + VAE + tokenizers (diffusers layout, fp16) -- NO unet:
#    the GGUF owns the unet. Our loader finds this dir via
#    models/*/model_index.json and loads components individually.
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('stabilityai/stable-diffusion-xl-base-1.0', \
  local_dir='/invokeai/models/sdxl-base', \
  allow_patterns=['model_index.json', \
    'text_encoder/config.json', 'text_encoder/model.fp16.safetensors', \
    'text_encoder_2/config.json', 'text_encoder_2/model.fp16.safetensors', \
    'tokenizer/*', 'tokenizer_2/*', \
    'vae/config.json', 'vae/diffusion_pytorch_model.fp16.safetensors'])"

# Fresh-root InvokeAI config: serve UI + API on the Space's app port.
RUN mkdir -p /invokeai && printf 'schema_version: 4.0.3\nhost: 0.0.0.0\nport: 7860\n' > /invokeai/invokeai.yaml

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 7860
ENTRYPOINT ["/entrypoint.sh"]
