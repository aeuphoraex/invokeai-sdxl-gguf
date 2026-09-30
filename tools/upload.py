#!/usr/bin/env python3
"""Upload a local file to the remote Jupyter instance via the contents API.

Usage:
    JUP_TOKEN=<token> python upload.py <local_path> <remote_path>

Remote paths are relative to the Jupyter server root (which is `/` on the box),
so e.g. "workspace/.ro/graph.py".
"""
import json
import os
import ssl
import sys
import urllib.request

BASE = os.environ.get("JUP_BASE_HTTP", "https://151.237.25.16:26112")
TOKEN = os.environ.get("JUP_TOKEN", "")


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    local, remote = sys.argv[1], sys.argv[2]

    with open(local, "r", encoding="utf-8") as fh:
        content = fh.read()

    body = json.dumps(
        {"type": "file", "format": "text", "content": content}
    ).encode("utf-8")

    url = f"{BASE}/api/contents/{remote}?token={TOKEN}"
    req = urllib.request.Request(
        url, data=body, method="PUT", headers={"Content-Type": "application/json"}
    )
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    print(f"uploaded {local} -> {data.get('path')} ({len(content)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
