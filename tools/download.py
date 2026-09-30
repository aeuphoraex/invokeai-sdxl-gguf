#!/usr/bin/env python3
"""Download a file from the remote Jupyter instance (raw /files/ endpoint).

Usage:
    JUP_TOKEN=<token> python download.py <remote_path> <local_path> [--part N --total M]

Remote paths are relative to the Jupyter server root ("/" on the box),
e.g. "workspace/.ro/package/invokeai-patched.tar.gz".
Streams to disk; safe for multi-GB files. Resumable via --part/--total
(byte-range chunks) for very large files.
"""
import argparse
import os
import ssl
import sys
import urllib.request

BASE = os.environ.get("JUP_BASE_HTTP", "https://151.237.25.16:26112")
TOKEN = os.environ.get("JUP_TOKEN", "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("remote")
    ap.add_argument("local")
    ap.add_argument("--part", type=int, default=None, help="chunk index (0-based)")
    ap.add_argument("--total", type=int, default=None, help="total chunk count")
    a = ap.parse_args()

    if not TOKEN:
        print("JUP_TOKEN is not set", file=sys.stderr)
        return 2

    url = f"{BASE}/files/{a.remote}?token={TOKEN}"
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    headers = {}
    mode = "wb"
    if a.part is not None and a.total is not None:
        # Fetch size first so we can compute the range.
        head = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(head, context=ctx, timeout=60) as r:
            size = int(r.headers["Content-Length"])
        chunk = (size + a.total - 1) // a.total
        start = a.part * chunk
        end = min(start + chunk, size) - 1
        headers["Range"] = f"bytes={start}-{end}"
        mode = "r+b" if os.path.exists(a.local) else "wb"
        print(f"part {a.part}/{a.total}: bytes {start}-{end} of {size}")

    req = urllib.request.Request(url, headers=headers)
    done = 0
    with urllib.request.urlopen(req, context=ctx, timeout=600) as resp, open(a.local, mode) as fh:
        if a.part is not None and a.total is not None and mode == "r+b":
            fh.seek(start)
        while True:
            buf = resp.read(1 << 20)
            if not buf:
                break
            fh.write(buf)
            done += len(buf)
            if done % (100 << 20) < (1 << 20):
                print(f"  {done >> 20} MiB", flush=True)
    print(f"downloaded {a.local} ({done} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
