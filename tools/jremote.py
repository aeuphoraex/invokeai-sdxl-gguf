#!/usr/bin/env python3
"""Run a shell command on a remote Jupyter terminal and print its output.

Usage:
    JUP_TOKEN=<token> python jremote.py "uname -a"
    JUP_TOKEN=<token> python jremote.py --term 1 "curl -s localhost:11111/capabilities/services"

Drives the Jupyter terminal WebSocket protocol:
    ["stdin", data] / ["stdout", data] / ["set_size", rows, cols]
Output is collected until the marker quiet period elapses or --wait expires.
"""
import argparse
import json
import os
import re
import ssl
import sys
import time

import websocket

BASE = os.environ.get("JUP_BASE", "wss://151.237.25.16:26112")
TOKEN = os.environ.get("JUP_TOKEN", "")
END = "\n###JREMOTE_DONE###\n"


def run(cmd: str, term: str, wait: float, cols: int, rows: int) -> str:
    url = f"{BASE}/terminals/websocket/{term}?token={TOKEN}"
    # The fronting Caddy proxy uses a self-signed/short-lived cert, so disable
    # verification for this local tooling connection.
    ws = websocket.create_connection(
        url,
        sslopt={"cert_reqs": ssl.CERT_NONE, "check_hostname": False},
        timeout=30,
    )
    ws.settimeout(3)

    # Normalise the terminal geometry so line wrapping is predictable.
    ws.send(json.dumps(["set_size", rows, cols]))
    time.sleep(0.3)
    # Discard the screen replay: it contains tags from *previous* runs, which
    # would otherwise make us break out of the collect loop before this run's
    # output has even arrived.
    _drain(ws, 1.5)

    # Interrupt anything in flight, then run the command bracketed by tags.
    ws.send(json.dumps(["stdin", "\x03"]))
    time.sleep(0.4)
    _drain(ws, 0.5)

    # Tags are built REMOTELY from $RANDOM so their digit form only ever exists
    # in real output. The echoed command line still shows the literal
    # "$RANDOM" text, which does not match the \d+ pattern below — otherwise
    # we would break out of the collect loop on our own echoed payload.
    # Single line only: a newline inside the brace group would put the remote
    # bash into continuation mode (">") and mangle everything.
    payload = (
        f"S=__S_$RANDOM$RANDOM__; E=__E_$RANDOM$RANDOM__; "
        f"printf '%s\\n' \"$S\"; {{ {cmd}; }}; printf '%s\\n' \"$E\"\n"
    )
    ws.send(json.dumps(["stdin", payload]))

    buf: list[str] = []
    deadline = time.time() + wait
    # NB: no trailing "__" in the pattern. The remote assignment
    # "__E_$RANDOM$RANDOM__" has its closing "__" swallowed as part of the
    # variable name "$RANDOM__" (unset -> empty), so real tags end in digits.
    end_re = re.compile(r"__E_\d+")
    while time.time() < deadline:
        try:
            raw = ws.recv()
        except websocket.WebSocketTimeoutException:
            continue
        except Exception:
            break
        if not isinstance(raw, str):
            continue
        try:
            msg = json.loads(raw)
        except Exception:
            continue
        if isinstance(msg, list) and len(msg) >= 2 and msg[0] == "stdout":
            buf.append(msg[1])
            if end_re.search("".join(buf)):
                break
    ws.close()

    out = _strip_ansi("".join(buf))
    # Real output is the LAST digit-tag occurrence of each kind.
    start_re = re.compile(r"__S_\d+")
    starts = list(start_re.finditer(out))
    ends = list(end_re.finditer(out))
    if starts and ends:
        # take the last start that precedes the last end
        j = ends[-1]
        prior = [m for m in starts if m.start() < j.start()]
        if prior:
            i = prior[-1]
            out = out[i.end() : j.start()]
    return out.strip()


def _drain(ws, seconds: float) -> None:
    """Read and discard terminal output for `seconds` (screen replay / echoes)."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            ws.recv()
        except websocket.WebSocketTimeoutException:
            continue
        except Exception:
            return


def _strip_ansi(s: str) -> str:
    import re

    s = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", s)
    s = re.sub(r"\x1b\][^\x07]*\x07", "", s)
    s = re.sub(r"[\r\x08]", "", s)
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--term", default="1")
    ap.add_argument("--wait", type=float, default=30.0)
    ap.add_argument("--cols", type=int, default=400)
    ap.add_argument("--rows", type=int, default=50)
    a = ap.parse_args()

    if not TOKEN:
        print("JUP_TOKEN is not set", file=sys.stderr)
        return 2

    print(run(a.cmd, a.term, a.wait, a.cols, a.rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
