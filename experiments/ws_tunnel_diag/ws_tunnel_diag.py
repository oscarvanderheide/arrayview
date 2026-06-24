"""Diagnostic: does WebSocket survive VS Code Remote Tunnel's devtunnel?

Run on the REMOTE side, inside a VS Code terminal that is connected via
Remote Tunnel (``code tunnel`` on the remote, client on your laptop).

    python experiments/ws_tunnel_diag/ws_tunnel_diag.py
    # or, if fastapi/uvicorn aren't on the default path:
    uv run --with fastapi --with uvicorn python experiments/ws_tunnel_diag/ws_tunnel_diag.py

What happens:
  1. A tiny FastAPI server starts on port 8765 with an HTTP and a WS endpoint.
  2. A signal file is written to ~/.arrayview/open-request-v0900.json so the
     installed arrayview-opener extension picks it up, calls asExternalUri,
     and opens a webview tab whose iframe points at the devtunnel URL.
  3. The page inside that iframe runs two self-tests (HTTP fetch + WS echo)
     and prints the results on the page — no DevTools required.

Read the four lines at the top of the opened tab:
  - asExternalUri URL    : https://HOST-8765.euw.devtunnels.ms/  (or FAIL)
  - HTTP through tunnel  : OK  /  FAIL
  - WS through tunnel    : OK  /  FAIL
  - page loaded from     : <the URL the iframe actually used>

Interpretation:
  - HTTP OK + WS OK        → devtunnel forwards WS; the fast WS transport
                             is viable over Remote Tunnel. Proceed with the
                             "asExternalUri + WS" plan (Phase 1A).
  - HTTP OK + WS FAIL      → devtunnel strips WS upgrades; WS is impossible
                             through the devtunnel. Use HTTP transport or
                             the WSS + self-signed cert path (Phase 1B).
  - Tab never opened       → asExternalUri failed/timed out. Forward port
                             8765 manually in the Ports tab, then open
                             http://localhost:<forwarded>/  in a browser to
                             confirm the server itself works (both HTTP and
                             WS should show OK — that's the baseline).

The server keeps running so you can reload the tab or try the manual
fallback. Ctrl+C to stop.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

PORT = 8765
SIGNAL_DIR = Path.home() / ".arrayview"
SIGNAL_FILE = SIGNAL_DIR / "open-request-v0900.json"
SIGNAL_MAX_AGE_MS = 60_000

app = FastAPI()


@app.get("/ping")
def ping():
    return {"ok": True}


@app.websocket("/ws/echo")
async def ws_echo(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_text()
            await ws.send_text(f"echo:{msg}")
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


def _detect_request_host() -> str:
    """Best-effort guess of the host the browser will see.

    For the iframe path the page is loaded from the devtunnel URL, so
    ``location.host`` (used by the page JS) is authoritative. This value is
    only a fallback shown on the server-side stdout.
    """
    try:
        import socket

        host = socket.gethostname()
        return f"{host}:{PORT}"
    except Exception:
        return f"localhost:{PORT}"


def _serve_local_url() -> str:
    return f"http://localhost:{PORT}/"


def _write_signal_file() -> None:
    """Write the fallback signal file the arrayview-opener extension watches.

    Mirrors the payload shape produced by arrayview._vscode_signal
    ._open_via_signal_file (action=open-preview, url, maxAgeMs, requestId).
    The extension calls vscode.env.asExternalUri on the port and opens a
    webview tab whose iframe points at the resulting devtunnel URL.
    """
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "action": "open-preview",
        "url": _serve_local_url(),
        "title": "WS Tunnel Diag",
        "maxAgeMs": SIGNAL_MAX_AGE_MS,
        "sentAtMs": int(time.time() * 1000),
        "requestId": uuid.uuid4().hex,
    }
    tmp = SIGNAL_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, SIGNAL_FILE)
    print(f"[diag] signal written → {SIGNAL_FILE}")
    print(f"[diag] payload url field: {payload['url']}")
    print(
        "[diag] if a tab does not open within ~10s, the extension did not "
        "pick up the signal or asExternalUri failed.\n"
        "       fallback: forward port "
        f"{PORT} manually in the VS Code Ports tab, then open "
        f"{_serve_local_url()} in a browser."
    )


@app.get("/", response_class=HTMLResponse)
def index():
    """Self-testing page.

    Loaded either through the devtunnel (iframe inside the extension's
    webview tab) or directly via http://localhost:<port>/ (manual fallback).
    Runs HTTP + WS probes against its own origin and renders the result.
    """
    return HTMLResponse(
        """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>WS Tunnel Diag</title>
<style>
  body { background:#101010; color:#e6e6e6; font-family: ui-monospace, Menlo, Consolas, monospace; padding:32px; line-height:1.6; }
  h1 { font-size:18px; margin:0 0 16px; }
  .row { display:flex; align-items:center; gap:12px; padding:8px 0; border-bottom:1px solid #222; }
  .label { min-width:260px; color:#9a9a9a; }
  .value { word-break:break-all; }
  .ok  { color:#80ed99; font-weight:600; }
  .bad { color:#f4845f; font-weight:600; }
  .pending { color:#f5c842; }
  .note { margin-top:24px; color:#777; font-size:13px; max-width:680px; }
  code { background:#1c1c1c; padding:2px 6px; border-radius:3px; }
</style>
</head>
<body>
<h1>WS Tunnel Diagnostic</h1>
<div class="row"><span class="label">asExternalUri URL</span><span class="value pending" id="ext-url">(waiting)</span></div>
<div class="row"><span class="label">HTTP through tunnel</span><span class="value pending" id="http-status">(testing…)</span></div>
<div class="row"><span class="label">WS through tunnel</span><span class="value pending" id="ws-status">(testing…)</span></div>
<div class="row"><span class="label">page loaded from</span><span class="value" id="origin"></span></div>
<p class="note">
  The iframe's <code>src</code> was set by the extension to the
  <code>asExternalUri</code> result. If "asExternalUri URL" shows
  <code>(no parent signal)</code> you loaded this page directly
  (manual fallback) — both probes still run against this origin.
  Copy the three statuses above into the opencode session.
</p>
<script>
const origin = location.origin;
const host = location.host;
document.getElementById('origin').textContent = origin;

// The parent webview (arrayview-opener extension) injects the final
// URL as the iframe src. We can't read the parent's asExternalUri
// value directly (cross-origin), but the iframe src IS that value,
// so if we're on a devtunnels.ms host we know asExternalUri worked.
const isDevtunnel = /\\.devtunnels\\.ms$/.test(host) || /\\.devtunnels\\.ms:/.test(host);
const extUrlEl = document.getElementById('ext-url');
if (isDevtunnel) {
  extUrlEl.textContent = origin + '  (devtunnel — asExternalUri succeeded)';
  extUrlEl.className = 'value ok';
} else if (window.parent && window.parent !== window) {
  extUrlEl.textContent = origin + '  (iframe, non-devtunnel host — asExternalUri may have fallen back to localhost)';
  extUrlEl.className = 'value bad';
} else {
  extUrlEl.textContent = origin + '  (no parent signal — direct load)';
  extUrlEl.className = 'value';
}

// HTTP probe
(async () => {
  const el = document.getElementById('http-status');
  try {
    const r = await fetch(origin + '/ping');
    const j = await r.json();
    if (j && j.ok) { el.textContent = 'OK  (GET /ping → {"ok":true})'; el.className = 'value ok'; }
    else { el.textContent = 'FAIL  (unexpected body)'; el.className = 'value bad'; }
  } catch (e) {
    el.textContent = 'FAIL  (' + (e && e.message ? e.message : e) + ')'; el.className = 'value bad';
  }
})();

// WS probe — use wss:// if the page is https, ws:// if http.
(() => {
  const el = document.getElementById('ws-status');
  const wsProto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = wsProto + '//' + host + '/ws/echo';
  let ws;
  try { ws = new WebSocket(wsUrl); } catch (e) {
    el.textContent = 'FAIL  (constructor: ' + (e && e.message ? e.message : e) + ')'; el.className = 'value bad'; return;
  }
  const tid = setTimeout(() => {
    el.textContent = 'FAIL  (timeout — no open within 6s)'; el.className = 'value bad';
    try { ws.close(); } catch (_) {}
  }, 6000);
  ws.onopen = () => {
    clearTimeout(tid);
    el.textContent = 'OK  (connected to ' + wsUrl + ')'; el.className = 'value ok';
    try { ws.send('hello'); } catch (_) {}
  };
  ws.onmessage = (e) => {
    if (e.data === 'echo:hello') {
      el.textContent = 'OK  (connected + echo round-trip)'; el.className = 'value ok';
    }
    try { ws.close(); } catch (_) {}
  };
  ws.onerror = () => {
    clearTimeout(tid);
    el.textContent = 'FAIL  (onerror — see browser console for detail)'; el.className = 'value bad';
  };
  ws.onclose = (e) => {
    clearTimeout(tid);
    if (el.className.indexOf('ok') === -1) {
      el.textContent = 'FAIL  (closed code=' + e.code + ' reason=' + JSON.stringify(e.reason || '') + ')';
      el.className = 'value bad';
    }
  };
})();
</script>
</body>
</html>
"""
    )


@app.get("/_diag_info")
def diag_info():
    """Server-side context for debugging."""
    return JSONResponse(
        {
            "port": PORT,
            "local_url": _serve_local_url(),
            "signal_file": str(SIGNAL_FILE),
            "signal_exists": SIGNAL_FILE.exists(),
            "guessed_remote_host": _detect_request_host(),
        }
    )


def main() -> None:
    print("=" * 72)
    print("  WS Tunnel Diagnostic — see opened VS Code tab for results.")
    print("  Server keeps running. Ctrl+C to stop.")
    print("=" * 72)
    print(f"[diag] local url : {_serve_local_url()}")
    print(f"[diag] signal dir: {SIGNAL_DIR}")
    _write_signal_file()
    print()
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
