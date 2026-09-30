#!/usr/bin/env python3
"""Baseline SDXL benchmark on InvokeAI: enqueue a graph, time it, record VRAM.

Usage:  python3 bench.py [--steps N] [--cfg F] [--size WxH] [--scheduler NAME]
                         [--label NAME] [--model KEY] [--out /path.json]
"""
import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

API = "http://127.0.0.1:9090"
QUEUE = "default"


def req(method, path, body=None, timeout=60):
    url = f"{API}{path}"
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method)
    if data:
        r.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:1200]
        raise RuntimeError(f"{method} {path} -> HTTP {e.code}: {detail}") from None


def vram_mb():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=15,
        )
        used, free = [int(x.strip()) for x in out.strip().splitlines()[0].split(",")]
        return used, free
    except Exception:  # noqa: BLE001
        return None, None


def pick_sdxl_model(override=None):
    if override:
        d = req("GET", "/api/v2/models/?model_type=main")
        for m in d.get("models", []):
            if m.get("key") == override:
                return m
        raise SystemExit(f"model key {override} not found")
    d = req("GET", "/api/v2/models/?model_type=main")
    sdxl = [m for m in d.get("models", []) if m.get("base") == "sdxl"]
    # prefer a checkpoint (single-file) as it exercises the standard loader path
    sdxl.sort(key=lambda m: (m.get("format") != "checkpoint", m.get("name", "")))
    if not sdxl:
        raise SystemExit("no sdxl main model installed")
    return sdxl[0]


def model_ident(m, submodel=None):
    ident = {
        "key": m["key"],
        "hash": m.get("hash"),
        "name": m.get("name"),
        "base": m.get("base"),
        "type": m.get("type"),
    }
    if submodel:
        ident["submodel_type"] = submodel
    return ident


def build_graph(m, prompt, negative, w, h, steps, cfg, scheduler, seed):
    mi = lambda sub=None: model_ident(m, sub)  # noqa: E731
    return {
        "id": f"bench-{int(time.time())}",
        "nodes": {
            "loader": {
                "id": "loader",
                "type": "SDXLModelLoaderInvocation",
                "model": model_ident(m),
            },
            "pos": {
                "id": "pos",
                "type": "SDXLCompelPromptInvocation",
                "prompt": prompt,
                "clip": {"placeholder": None},
            },
            "noise": {
                "id": "noise",
                "type": "NoiseInvocation",
                "seed": seed,
                "width": w,
                "height": h,
            },
            "denoise": {
                "id": "denoise",
                "type": "DenoiseLatentsInvocation",
                "steps": steps,
                "cfg_scale": cfg,
                "scheduler": scheduler,
                "positive_conditioning": {},
                "negative_conditioning": {},
                "noise": {},
                "unet": {},
            },
            "l2i": {
                "id": "l2i",
                "type": "LatentsToImageInvocation",
                "latents": {},
                "vae": {},
            },
        },
        "edges": [
            # placeholder, replaced by real connection dicts below
        ],
    }


def conn(node_id, field):
    return {"node_id": node_id, "field": field}


def build_graph_full(m, prompt, negative, w, h, steps, cfg, scheduler, seed):
    g = {
        "id": f"bench-{int(time.time())}",
        "nodes": {
            "loader": {
                "id": "loader",
                "type": "sdxl_model_loader",
                "model": model_ident(m),
            },
            "pos": {
                "id": "pos",
                "type": "sdxl_compel_prompt",
                "prompt": prompt,
            },
            "neg": {
                "id": "neg",
                "type": "sdxl_compel_prompt",
                "prompt": negative,
            },
            "noise": {
                "id": "noise",
                "type": "noise",
                "seed": seed,
                "width": w,
                "height": h,
            },
            "denoise": {
                "id": "denoise",
                "type": "denoise_latents",
                "steps": steps,
                "cfg_scale": cfg,
                "scheduler": scheduler,
            },
            "l2i": {"id": "l2i", "type": "l2i"},
        },
        "edges": [
            {"source": {"node_id": "loader", "field": "clip"}, "destination": {"node_id": "pos", "field": "clip"}},
            {"source": {"node_id": "loader", "field": "clip2"}, "destination": {"node_id": "pos", "field": "clip2"}},
            {"source": {"node_id": "loader", "field": "clip"}, "destination": {"node_id": "neg", "field": "clip"}},
            {"source": {"node_id": "loader", "field": "clip2"}, "destination": {"node_id": "neg", "field": "clip2"}},
            {"source": {"node_id": "pos", "field": "conditioning"}, "destination": {"node_id": "denoise", "field": "positive_conditioning"}},
            {"source": {"node_id": "neg", "field": "conditioning"}, "destination": {"node_id": "denoise", "field": "negative_conditioning"}},
            {"source": {"node_id": "noise", "field": "noise"}, "destination": {"node_id": "denoise", "field": "noise"}},
            {"source": {"node_id": "loader", "field": "unet"}, "destination": {"node_id": "denoise", "field": "unet"}},
            {"source": {"node_id": "denoise", "field": "latents"}, "destination": {"node_id": "l2i", "field": "latents"}},
            {"source": {"node_id": "loader", "field": "vae"}, "destination": {"node_id": "l2i", "field": "vae"}},
        ],
    }
    return g


def snapshot_ids():
    try:
        d = req("GET", f"/api/v1/queue/{QUEUE}/list_all?limit=100")
    except Exception:  # noqa: BLE001
        return set()
    items = d if isinstance(d, list) else d.get("items", [])
    return {i.get("item_id") for i in items}


def wait_new_item(before_ids, timeout=900, poll=2.0):
    """Poll until an item_id not present in before_ids appears and finishes."""
    deadline = time.time() + timeout
    mine = None
    while time.time() < deadline:
        try:
            d = req("GET", f"/api/v1/queue/{QUEUE}/list_all?limit=100")
        except Exception:  # noqa: BLE001
            time.sleep(poll)
            continue
        items = d if isinstance(d, list) else d.get("items", [])
        new = [i for i in items if i.get("item_id") not in before_ids]
        if not new:
            time.sleep(poll)
            continue
        # newest first is not guaranteed; take the lowest id (first enqueued)
        mine = sorted(new, key=lambda x: x.get("item_id") or 0)[0]
        st = mine.get("status")
        if st in ("completed", "failed", "canceled"):
            return mine, st
        time.sleep(poll)
    return mine, "timeout"


def duration_s(item):
    """Wall time for the item itself, from InvokeAI's own timestamps."""
    if not item:
        return None
    try:
        from datetime import datetime

        fmt = "%Y-%m-%d %H:%M:%S.%f"
        s, e = item.get("started_at"), item.get("completed_at")
        if s and e:
            return round((datetime.strptime(e, fmt) - datetime.strptime(s, fmt)).total_seconds(), 2)
    except Exception:  # noqa: BLE001
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--scheduler", default="euler")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--model", default=None)
    ap.add_argument("--prompt", default="a red cube on a white table, studio lighting")
    ap.add_argument("--negative", default="blurry, low quality")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", default="/workspace/.ro/bench.json")
    a = ap.parse_args()

    w, h = (int(x) for x in a.size.lower().split("x"))

    used0, free0 = vram_mb()
    m = pick_sdxl_model(a.model)
    print(f"[model] {m.get('name')} base={m.get('base')} fmt={m.get('format')} key={m['key']}")

    g = build_graph_full(m, a.prompt, a.negative, w, h, a.steps, a.cfg, a.scheduler, a.seed)

    before = snapshot_ids()
    t0 = time.time()
    # `batch` is a single Batch object: {"graph": ..., "runs": N} — not a list.
    resp = req(
        "POST",
        f"/api/v1/queue/{QUEUE}/enqueue_batch",
        {"batch": {"graph": g, "runs": 1}},
    )
    batch_id = None
    if isinstance(resp, dict):
        for k in ("batch_ids", "batch_id"):
            v = resp.get(k)
            if isinstance(v, list) and v:
                batch_id = v[0]
            elif isinstance(v, str):
                batch_id = v
        if batch_id is None:
            print(f"[enqueue raw] {json.dumps(resp)[:400]}")
    print(f"[enqueue] batch={batch_id}")

    item, status = wait_new_item(before, timeout=a.timeout)
    elapsed = time.time() - t0

    result = {
        "label": a.label,
        "model": m.get("name"),
        "model_key": m["key"],
        "format": m.get("format"),
        "steps": a.steps,
        "cfg": a.cfg,
        "size": f"{w}x{h}",
        "scheduler": a.scheduler,
        "status": status,
        "wall_s": round(elapsed, 2),
        "compute_s": duration_s(item),
        "item_id": (item or {}).get("item_id"),
        "vram_used_before_mb": used0,
        "vram_free_before_mb": free0,
    }

    used1, free1 = vram_mb()
    result["vram_used_after_mb"] = used1
    result["vram_free_after_mb"] = free1

    if item:
        result["error"] = item.get("error_type")
        result["error_msg"] = (item.get("error_message") or "")[:500]
        if item.get("status") != "completed":
            result["traceback"] = (item.get("error_traceback") or "")[:1500]

    print(json.dumps(result, indent=2))
    with open(a.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[saved] {a.out}")


if __name__ == "__main__":
    main()
