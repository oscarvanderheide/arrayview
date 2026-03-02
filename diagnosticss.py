#!/usr/bin/env python3
"""Run this on the remote machine to diagnose the tunnel environment."""
import os, sys, subprocess, glob

print("=== Environment Variables ===")
for k, v in sorted(os.environ.items()):
    kl = k.lower()
    if any(x in kl for x in ["vscode", "term_program", "remote", "ssh", "display"]):
        print(f"  {k}={v}")

print("\n=== code CLI ===")
import shutil
code = shutil.which("code")
print(f"  which code: {code}")
if code:
    real = os.path.realpath(code)
    print(f"  realpath: {real}")

print("\n=== VS Code Server ===")
for pattern in [
    os.path.expanduser("~/.vscode-server/bin/*/bin/remote-cli/code"),
    os.path.expanduser("~/.vscode-server/cli/servers/*/server/bin/remote-cli/code"),
]:
    matches = glob.glob(pattern)
    for m in matches:
        print(f"  {m}")

print("\n=== IPC sockets ===")
for pattern in [
    "/tmp/vscode-ipc-*.sock",
    os.path.expanduser("~/.vscode-server/*.sock"),
]:
    for m in glob.glob(pattern):
        print(f"  {m}")
