#!/bin/bash
set -euo pipefail

ROOT=/invokeai
API=http://127.0.0.1:7860

echo "[entrypoint] starting invokeai-web (root=$ROOT, port=7860)"
invokeai-web --root "$ROOT" &
IPID=$!

# Wait for the API (HF already sees the port open; this gates model install).
for i in $(seq 1 150); do
    if curl -sf "$API/openapi.json" > /dev/null 2>&1; then
        echo "[entrypoint] API up after ~$((i * 2))s"
        break
    fi
    if ! kill -0 "$IPID" 2>/dev/null; then
        echo "[entrypoint] invokeai-web exited during startup" >&2
        exit 1
    fi
    sleep 2
done

# Idempotent GGUF install (fresh container = fresh DB, so usually runs once).
if curl -s "$API/api/v2/models/?base=sdxl&type=main" | grep -q gguf_quantized; then
    echo "[entrypoint] SDXL GGUF already installed"
else
    echo "[entrypoint] installing SDXL UNet Q8 GGUF from /opt/models ..."
    JOB=$(curl -s -X POST "$API/api/v2/models/install?source=/opt/models/sdxl-unet-q8.gguf" \
        -H 'Content-Type: application/json' \
        -d '{"name":"SDXL UNet Q8 GGUF","base":"sdxl","type":"main","format":"gguf_quantized","description":"Q8_0 SDXL UNet - self-describing GGUF (sdxl.* header KVs)"}')
    echo "[entrypoint] install job: $JOB"
    ID=$(printf '%s' "$JOB" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("id",""))' 2>/dev/null || true)
    if [ -n "$ID" ]; then
        for j in $(seq 1 100); do
            S=$(curl -s "$API/api/v2/models/install/$ID" | \
                python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("status",""), d.get("error_reason") or "")' 2>/dev/null || echo "")
            case "$S" in
                completed*) echo "[entrypoint] install completed"; break ;;
                error*|canceled*) echo "[entrypoint] install FAILED: $S" >&2; break ;;
            esac
            sleep 3
        done
    fi
fi

echo "[entrypoint] InvokeAI ready on :7860"
wait "$IPID"
