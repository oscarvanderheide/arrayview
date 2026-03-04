#!/usr/bin/env python3
"""
ArrayView Tunnel Diagnostics
=============================
Run this from a VS Code tunnel terminal to diagnose the setup:
    uv run python diagnostics.py
"""
import json
import os
import subprocess
import sys
import glob
from pathlib import Path
from datetime import datetime


def section(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def ok(msg):
    print(f"  ✅ {msg}")


def warn(msg):
    print(f"  ⚠️  {msg}")


def fail(msg):
    print(f"  ❌ {msg}")


def info(msg):
    print(f"  ℹ️  {msg}")


# ------------------------------------------------------------------
section("1. Environment")
# ------------------------------------------------------------------
ipc_hook = os.environ.get("VSCODE_IPC_HOOK_CLI", "")
term_program = os.environ.get("TERM_PROGRAM", "")
ssh_conn = os.environ.get("SSH_CONNECTION", "")
display = os.environ.get("DISPLAY", "")

print(f"  Platform:              {sys.platform}")
print(f"  Python:                {sys.executable} ({sys.version.split()[0]})")
print(f"  Home:                  {os.path.expanduser('~')}")
print(f"  TERM_PROGRAM:          {term_program or '(not set)'}")
print(f"  VSCODE_IPC_HOOK_CLI:   {ipc_hook or '(not set)'}")
print(f"  SSH_CONNECTION:         {ssh_conn or '(not set)'}")
print(f"  DISPLAY:               {display or '(not set)'}")

if ipc_hook:
    if os.path.exists(ipc_hook):
        ok(f"IPC socket exists: {ipc_hook}")
    else:
        fail(f"IPC socket does NOT exist: {ipc_hook}")
elif term_program == "vscode":
    warn("VSCODE_IPC_HOOK_CLI not set — will try process-tree walk")
else:
    warn("Not in a VS Code terminal (TERM_PROGRAM != vscode)")

# ------------------------------------------------------------------
section("2. VS Code Remote Detection (from _app.py)")
# ------------------------------------------------------------------
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    from arrayview._app import (
        _is_vscode_remote,
        _find_code_cli,
        _find_vscode_ipc_hook,
        _VSCODE_EXT_VERSION,
    )

    remote = _is_vscode_remote()
    code_cli = _find_code_cli()
    ipc_found = _find_vscode_ipc_hook()

    print(f"  _is_vscode_remote():   {remote}")
    print(f"  _find_code_cli():      {code_cli or '(None)'}")
    print(f"  _find_vscode_ipc_hook(): {ipc_found or '(None)'}")
    print(f"  _VSCODE_EXT_VERSION:   {_VSCODE_EXT_VERSION}")

    if remote:
        ok("Detected as VS Code remote")
    else:
        warn("NOT detected as VS Code remote — signal file path won't be used")
        info("For tunnel: VSCODE_IPC_HOOK_CLI must be set (check process tree)")

    if code_cli:
        ok(f"Found code CLI: {code_cli}")
    else:
        fail("Could not find 'code' CLI")

    if ipc_found:
        ok(f"Found IPC hook: {ipc_found}")
    else:
        warn("Could not find IPC hook (needed for code --install-extension)")

except Exception as e:
    fail(f"Could not import arrayview: {e}")
    _VSCODE_EXT_VERSION = "0.1.3"

# ------------------------------------------------------------------
section("3. Tunnel Configuration")
# ------------------------------------------------------------------
tunnel_config = None
for p in [
    os.path.expanduser("~/.vscode/cli/code_tunnel.json"),
    os.path.expanduser("~/.vscode-server/cli/code_tunnel.json"),
]:
    try:
        with open(p) as f:
            tunnel_config = json.load(f)
        ok(f"Tunnel config: {p}")
        print(f"  Name:    {tunnel_config.get('name', '?')}")
        print(f"  Cluster: {tunnel_config.get('cluster', '?')}")
        break
    except Exception:
        pass
if not tunnel_config:
    warn("No code_tunnel.json found — not a tunnel setup?")

# ------------------------------------------------------------------
section("4. Extension Installation")
# ------------------------------------------------------------------
ext_dirs = [
    os.path.expanduser("~/.vscode-server/extensions/"),
    os.path.expanduser("~/.vscode/extensions/"),
]
found_ext = False
for d in ext_dirs:
    pattern = os.path.join(d, "arrayview.arrayview-opener-*")
    matches = glob.glob(pattern)
    for m in matches:
        ver = m.split("@")[-1] if "@" in m else m.rsplit("-", 1)[-1]
        found_ext = True
        # Check if it has the right extension.js
        ext_js = os.path.join(m, "extension.js")
        if os.path.isfile(ext_js):
            with open(ext_js) as f:
                first_line = f.readline().strip()
            ok(f"Extension at {m}")
            info(f"  extension.js header: {first_line}")
        else:
            warn(f"Extension dir exists but no extension.js: {m}")

if not found_ext:
    fail("Extension NOT installed in any known location")
    info("Install manually: code --install-extension src/arrayview/arrayview-opener.vsix --force")

# ------------------------------------------------------------------
section("5. Extension Log")
# ------------------------------------------------------------------
log_file = os.path.expanduser("~/.arrayview/extension.log")
if os.path.isfile(log_file):
    with open(log_file) as f:
        lines = f.readlines()
    
    # Show last 10 lines
    print("  Last entries:")
    for line in lines[-10:]:
        print(f"    {line.rstrip()}")
    
    # Check for today's activation
    today = datetime.now().strftime("%Y-%m-%d")
    today_activations = [l for l in lines if today in l and "activate" in l]
    if today_activations:
        ok(f"Extension activated today ({len(today_activations)} time(s))")
        for a in today_activations:
            info(f"  {a.strip()}")
    else:
        fail("Extension has NOT activated today")
        info("→ Reload VS Code window: Ctrl+Shift+P → 'Developer: Reload Window'")
else:
    warn(f"No extension log at {log_file}")
    info("Extension may never have been activated")

# ------------------------------------------------------------------
section("6. Signal File Test")
# ------------------------------------------------------------------
signal_dir = os.path.expanduser("~/.arrayview")
signal_file = os.path.join(signal_dir, "open-request.json")

if os.path.exists(signal_file):
    age_s = os.time() - os.path.getmtime(signal_file)
    fail(f"Stale signal file exists (age: {age_s:.0f}s) — extension not consuming it!")
    info("This means the extension is NOT watching the signal dir")
    with open(signal_file) as f:
        print(f"  Contents: {f.read().strip()}")
else:
    ok("No stale signal file (good)")

# ------------------------------------------------------------------
section("7. Tunnel Server Processes")
# ------------------------------------------------------------------
try:
    r = subprocess.run(
        ["ps", "aux"],
        capture_output=True, text=True, timeout=5,
    )
    tunnel_procs = [l for l in r.stdout.splitlines() if "code-tunnel" in l.lower() or "code tunnel" in l.lower()]
    ext_host_procs = [l for l in r.stdout.splitlines() if "extensionHost" in l and "transformURIs" in l]
    
    if tunnel_procs:
        ok(f"Tunnel daemon running ({len(tunnel_procs)} process(es))")
    else:
        warn("No tunnel daemon found")
    
    if ext_host_procs:
        ok(f"Extension hosts running ({len(ext_host_procs)})")
    else:
        warn("No tunnel extension hosts found")
except Exception as e:
    warn(f"Could not check processes: {e}")

# ------------------------------------------------------------------
section("8. Port Check")
# ------------------------------------------------------------------
import socket
for port in [8000, 8123]:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            info(f"Port {port} is in use (something is already listening)")
    except (ConnectionRefusedError, OSError):
        ok(f"Port {port} is free")

# ------------------------------------------------------------------
section("Summary")
# ------------------------------------------------------------------
print()
issues = []
if not ipc_hook and not (term_program == "vscode"):
    issues.append("Not running in VS Code terminal")
if not found_ext:
    issues.append("Extension not installed")
if os.path.isfile(log_file):
    with open(log_file) as f:
        content = f.read()
    today = datetime.now().strftime("%Y-%m-%d")
    if today not in content or "activate" not in content.split(today)[-1] if today in content else True:
        issues.append("Extension not activated today — reload VS Code window")

if not issues:
    ok("Everything looks good! Try: uv run arrayview small_array.npy --browser")
else:
    print("  Issues to fix:")
    for i, issue in enumerate(issues, 1):
        print(f"    {i}. {issue}")
    print()
    info("After fixing, re-run this script to verify")
