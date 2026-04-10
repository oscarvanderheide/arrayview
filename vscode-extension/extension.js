const vscode = require('vscode');
const fs = require('fs');
const path = require('path');
const os = require('os');
const crypto = require('crypto');
const net = require('net');
const { spawnSync } = require('child_process');

const SIGNAL_DIR = path.join(os.homedir(), '.arrayview');
const SIGNAL_FILE = path.join(SIGNAL_DIR, 'open-request-v0900.json');  // fallback
const LOG_FILE = path.join(SIGNAL_DIR, 'extension.log');

// Per-window targeted signal file: Python writes to a file named by the SHA256
// of VSCODE_IPC_HOOK_CLI, which is unique per VS Code window on the remote.
// The extension checks its own targeted file first, then the fallback.
//
// On LOCAL VS Code desktop the extension host may not inherit VSCODE_IPC_HOOK_CLI
// directly, so we also walk parent processes to find it — the same approach
// Python uses in _platform._find_vscode_ipc_hook().
function _findVscodeIpcHook() {
    const direct = process.env.VSCODE_IPC_HOOK_CLI || '';
    if (direct && fs.existsSync(direct)) return direct;
    // Walk up to 8 ancestor processes looking for VSCODE_IPC_HOOK_CLI.
    let pid = process.pid;
    for (let i = 0; i < 8; i++) {
        const ppidRes = spawnSync('ps', ['-p', String(pid), '-o', 'ppid='],
            { encoding: 'utf8', timeout: 2000 });
        const ppid = parseInt((ppidRes.stdout || '').trim(), 10);
        if (!ppid || ppid <= 1) break;
        const envRes = spawnSync('ps', ['ewwww', '-p', String(ppid)],
            { encoding: 'utf8', timeout: 3000 });
        for (const token of (envRes.stdout || '').split(/\s+/)) {
            if (token.startsWith('VSCODE_IPC_HOOK_CLI=')) {
                const val = token.slice('VSCODE_IPC_HOOK_CLI='.length);
                if (val && fs.existsSync(val)) return val;
            }
        }
        pid = ppid;
    }
    return '';
}

const OWN_IPC_HOOK = _findVscodeIpcHook();
const OWN_HOOK_TAG = OWN_IPC_HOOK
    ? crypto.createHash('sha256').update(OWN_IPC_HOOK).digest('hex').slice(0, 16)
    : '';
// Targeted signal file: prefer IPC hook-based, fallback to PID-based for local desktop.
// This enables multi-window targeting even when VSCODE_IPC_HOOK_CLI isn't available.
const TARGETED_SIGNAL_FILE = OWN_HOOK_TAG
    ? path.join(SIGNAL_DIR, `open-request-ipc-${OWN_HOOK_TAG}.json`)
    : path.join(SIGNAL_DIR, `open-request-pid-${process.pid}.json`);

// Collect ancestor PIDs for cross-process window matching.
// Python can identify which VS Code window spawned a given terminal by finding
// the window whose extension host shares a common ancestor with the terminal process.
// Records up to 8 levels: [ppid, pppid, ...] stopping before PID 1.
function _getAncestorPids(pid, depth) {
    const result = [];
    let p = pid;
    for (let i = 0; i < depth; i++) {
        const res = spawnSync('ps', ['-p', String(p), '-o', 'ppid='],
            { encoding: 'utf8', timeout: 1000 });
        const ppid = parseInt((res.stdout || '').trim(), 10);
        if (!ppid || ppid <= 1) break;
        result.push(ppid);
        p = ppid;
    }
    return result;
}
const EXT_PPIDS = _getAncestorPids(process.pid, 8);

// ---------------------------------------------------------------------------
// PythonBridge: spawns `python -m arrayview --mode stdio <filepath>` and
// bridges length-prefixed binary responses from stdout / JSON requests on stdin.
// ---------------------------------------------------------------------------
class PythonBridge {
    constructor(filePath, onSessionReady, pythonPath, shmParams, extraArgs) {
        this.filePath = filePath;
        this.shmParams = shmParams || null;  // {name, shape, dtype}
        this.extraArgs = extraArgs || [];    // additional CLI args (e.g. --vectorfield)
        this.process = null;
        this.sid = null;
        this._buffer = Buffer.alloc(0);
        this._pendingCallbacks = [];
        this.onSessionReady = onSessionReady;
        this.pythonPath = pythonPath || null;
    }

    start() {
        return new Promise((resolve, reject) => {
            this._startResolve = resolve;
            this._startReject = reject;
            // Candidate order:
            // 1. Explicit pythonPath (from signal file — always correct)
            // 2. Workspace .venv (covers uv sync / editable installs during dev)
            // 3. System python3/python (might have arrayview via pip)
            // 4. uv run --with arrayview (ephemeral env from PyPI — always works)
            if (this.pythonPath) {
                this._candidates = [this.pythonPath];
            } else {
                this._candidates = ['python3', 'python', 'uv run --with arrayview python'];
                // Prepend workspace .venv if it exists
                const folders = vscode.workspace.workspaceFolders;
                if (folders) {
                    for (const f of folders) {
                        const venvPy = path.join(f.uri.fsPath, '.venv', 'bin', 'python');
                        if (fs.existsSync(venvPy)) {
                            this._candidates.unshift(venvPy);
                            break;
                        }
                    }
                }
            }
            this._tryNextCandidate();
        });
    }

    _tryNextCandidate() {
        if (this._candidates.length === 0) {
            const msg = 'Python with arrayview not found. Install with: pip install arrayview (or uv pip install arrayview)';
            log(`PYTHON: all candidates exhausted`);
            if (this._startReject) { this._startReject(new Error(msg)); this._startReject = null; }
            return;
        }
        const cmd = this._candidates.shift();
        this._spawn(cmd);
    }

    _spawn(cmd) {
        const { spawn } = require('child_process');
        let arrayviewArgs;
        if (this.shmParams) {
            arrayviewArgs = ['-m', 'arrayview', '--mode', 'stdio',
                '--shm-name', this.shmParams.name,
                '--shm-shape', this.shmParams.shape,
                '--shm-dtype', this.shmParams.dtype];
            if (this.shmParams.arrayName) {
                arrayviewArgs.push('--name', this.shmParams.arrayName);
            }
        } else {
            arrayviewArgs = ['-m', 'arrayview', '--mode', 'stdio', this.filePath];
        }
        if (this.extraArgs.length > 0) {
            arrayviewArgs.push(...this.extraArgs);
        }

        // Support compound commands like "uv run python"
        let spawnCmd, spawnArgs;
        const parts = cmd.split(/\s+/);
        if (parts.length > 1) {
            spawnCmd = parts[0];
            spawnArgs = [...parts.slice(1), ...arrayviewArgs];
        } else {
            spawnCmd = cmd;
            spawnArgs = arrayviewArgs;
        }

        log(`PYTHON: spawning ${spawnCmd} ${spawnArgs.join(' ')}`);
        this.process = spawn(spawnCmd, spawnArgs, { stdio: ['pipe', 'pipe', 'pipe'] });

        this._currentCmd = cmd;

        this.process.on('error', (err) => {
            if (err.code === 'ENOENT' && this._candidates.length > 0) {
                log(`PYTHON: ${spawnCmd} not found, trying next candidate`);
                this._tryNextCandidate();
                return;
            }
            log(`PYTHON: spawn error: ${err.message}`);
            if (this._startReject) {
                this._startReject(new Error(err.message));
                this._startReject = null;
            }
        });

        // Handle stderr for SESSION: lines and errors
        this.process.stderr.on('data', (data) => {
            const lines = data.toString().split('\n');
            for (const line of lines) {
                if (line.startsWith('SESSION:')) {
                    try {
                        const info = JSON.parse(line.slice(8));
                        this.sid = info.sid;
                        this.sessionInfo = info;  // preserve full session info (overlay_sid, etc.)
                        if (this.onSessionReady) this.onSessionReady(info);
                        if (this._startResolve) {
                            this._startResolve(info);
                            this._startResolve = null;
                        }
                    } catch (_) {}
                } else if (line.trim()) {
                    log(`PYTHON: ${line}`);
                }
            }
        });

        // Handle stdout: read length-prefixed responses
        this.process.stdout.on('data', (chunk) => {
            this._buffer = Buffer.concat([this._buffer, chunk]);
            this._processBuffer();
        });

        this.process.on('exit', (code) => {
            log(`PYTHON: ${this._currentCmd} exited with code ${code}`);
            // If we haven't started yet and there are more candidates, try next
            if (code !== 0 && this._startResolve && this._candidates && this._candidates.length > 0) {
                log(`PYTHON: trying next candidate after exit code ${code}`);
                this._tryNextCandidate();
                return;
            }
            // Reject any pending request callbacks
            for (const cb of this._pendingCallbacks) {
                cb(Buffer.from(JSON.stringify({ error: `process exited with code ${code}` })));
            }
            this._pendingCallbacks = [];
            // Reject start promise if still pending
            if (this._startReject) {
                this._startReject(new Error(`Python process exited with code ${code}`));
                this._startReject = null;
            }
        });
    }

    _processBuffer() {
        while (this._buffer.length >= 4) {
            const len = this._buffer.readUInt32LE(0);
            if (this._buffer.length < 4 + len) break;

            const payload = this._buffer.slice(4, 4 + len);
            this._buffer = this._buffer.slice(4 + len);

            const cb = this._pendingCallbacks.shift();
            if (cb) {
                cb(payload);
            } else {
                log(`WARNING: response (${payload.length} bytes) with no pending callback — FIFO misaligned`);
            }
        }
    }

    sendRequest(msg) {
        return new Promise((resolve) => {
            this._pendingCallbacks.push(resolve);
            const ok = this.process.stdin.write(JSON.stringify(msg) + '\n');
            if (!ok) log(`WARNING: stdin.write returned false (backpressure) for ${msg.type}, pending=${this._pendingCallbacks.length}`);
        });
    }

    destroy() {
        if (this.process) {
            this.process.kill();
            this.process = null;
        }
    }
}

// ---------------------------------------------------------------------------
// Direct webview: embeds the viewer HTML directly (no iframe/server)
// ---------------------------------------------------------------------------

/**
 * Wire up a webview panel to a PythonBridge: set HTML, bridge messages,
 * clean up on dispose.  Shared by openDirectWebview and the custom editor.
 */
async function setupArrayViewPanel(panel, filePath, pythonPath, shmParams, extraArgs) {
    const bridge = new PythonBridge(filePath, (sessionInfo) => {
        log(`SESSION READY: sid=${sessionInfo.sid} name=${sessionInfo.name}`);
    }, pythonPath, shmParams, extraArgs);
    const sessionInfo = await Promise.race([
        bridge.start(),
        new Promise((_, reject) => setTimeout(() => reject(new Error('Timeout waiting for Python session (30s)')), 30000)),
    ]);

    // Get the rendered viewer HTML from the Python subprocess.
    // Forward the full sessionInfo so the stdio server can build the correct
    // query string (overlay_sid, etc.) without the extension needing to know
    // about each feature.
    const htmlPayload = await bridge.sendRequest({
        type: 'get-viewer-html',
        sid: bridge.sid,
        ...(bridge.sessionInfo || {}),
    });

    let viewerHtml;
    try {
        const response = JSON.parse(htmlPayload.toString());
        viewerHtml = response.html;
    } catch (e) {
        bridge.destroy();
        throw new Error(`Failed to get viewer HTML: ${e.message}`);
    }

    panel.webview.html = viewerHtml;

    // Bridge messages between webview and Python subprocess
    panel.webview.onDidReceiveMessage(async (msg) => {
        if (msg.type === 'ws-send') {
            // Slice request from viewer -> forward to Python.
            // The entire block is wrapped in try/catch so that any unexpected
            // error still sends a synthetic frame — preventing the viewer's
            // isRendering flag from getting stuck forever.
            try {
                const request = {
                    type: 'slice',
                    sid: msg.sid || bridge.sid,
                    ...msg.data,
                };
                const payload = await bridge.sendRequest(request);

                // Check if this is a binary slice response or a JSON error
                // JSON responses start with '{' (0x7b)
                if (payload[0] === 0x7b) {
                    let errorMsg = 'Unknown slice error';
                    try {
                        const err = JSON.parse(payload.toString());
                        errorMsg = err.error || errorMsg;
                        log(`SLICE ERROR: ${errorMsg}`);
                    } catch (_) {}
                    // Synthesize a minimal 1×1 transparent frame so the viewer
                    // resets isRendering.  Header: seq(u32)+width(u32)+height(u32)
                    // +vmin(f32)+vmax(f32) = 20 bytes, then 4 bytes RGBA.
                    const frame = Buffer.alloc(24);
                    frame.writeUInt32LE(msg.data.seq || 0, 0);
                    frame.writeUInt32LE(1, 4);   // width
                    frame.writeUInt32LE(1, 8);   // height
                    // vmin, vmax, RGBA all zeros (transparent)
                    const bytes = new Uint8Array(frame.buffer, frame.byteOffset, frame.byteLength);
                    const delivered = await panel.webview.postMessage({
                        type: 'slice-data',
                        channelId: msg.channelId,
                        buffer: bytes,
                    });
                    if (!delivered) log(`WARNING: error-frame postMessage not delivered for ${msg.channelId}`);
                    return;
                }

                // VS Code postMessage uses structured clone which handles
                // Uint8Array but NOT raw ArrayBuffer reliably across the IPC bridge.
                // Send as Uint8Array — the viewer side reconstructs the ArrayBuffer.
                const bytes = new Uint8Array(payload.buffer, payload.byteOffset, payload.byteLength);

                const delivered = await panel.webview.postMessage({
                    type: 'slice-data',
                    channelId: msg.channelId,
                    buffer: bytes,
                });
                if (!delivered) log(`WARNING: slice-data postMessage not delivered for ${msg.channelId}`);
            } catch (e) {
                log(`SLICE RELAY ERROR: ${e.message}`);
                // Synthesize a 1×1 frame so the viewer can recover
                try {
                    const frame = Buffer.alloc(24);
                    frame.writeUInt32LE(msg.data.seq || 0, 0);
                    frame.writeUInt32LE(1, 4);
                    frame.writeUInt32LE(1, 8);
                    const bytes = new Uint8Array(frame.buffer, frame.byteOffset, frame.byteLength);
                    await panel.webview.postMessage({ type: 'slice-data', channelId: msg.channelId, buffer: bytes });
                } catch (_) {}
            }
        } else if (msg.type === 'fetch-proxy') {
            // Proxied fetch from viewer -> forward to Python
            const url = msg.url;
            const parts = url.replace(/^\//, '').split('/');
            const route = parts[0];
            const sid = parts[1] || bridge.sid;

            let request;
            if (route === 'metadata') {
                request = { type: 'metadata', sid };
            } else if (route === 'clearcache') {
                request = { type: 'clearcache', sid };
            } else if (route === 'sessions') {
                request = { type: 'sessions' };
            } else {
                request = { type: 'fetch-proxy', endpoint: url };
            }

            try {
                const payload = await bridge.sendRequest(request);
                const data = JSON.parse(payload.toString());

                // Binary responses (e.g. lebesgue) are base64-encoded by the
                // stdio server with a _binary flag and custom headers.
                if (data._binary && data.data) {
                    const buf = Buffer.from(data.data, 'base64');
                    const bytes = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
                    panel.webview.postMessage({
                        type: 'fetch-response',
                        id: msg.id,
                        ok: true,
                        status: 200,
                        data: bytes,
                        headers: data.headers || {},
                    });
                    return;
                }

                panel.webview.postMessage({
                    type: 'fetch-response',
                    id: msg.id,
                    ok: !data.error,
                    status: data.error ? 500 : 200,
                    data,
                    headers: data.headers || {},
                });
            } catch (e) {
                panel.webview.postMessage({
                    type: 'fetch-response',
                    id: msg.id,
                    ok: false,
                    status: 500,
                    data: { error: e.message },
                    headers: {},
                });
            }
        }
    });

    // Clean up on panel close
    panel.onDidDispose(() => {
        bridge.destroy();
        // Delete relay temp files when their panel is closed
        if (relayTempFiles.has(filePath)) {
            try { fs.unlinkSync(filePath); } catch (_) {}
            relayTempFiles.delete(filePath);
            log(`RELAY: cleaned up temp file ${path.basename(filePath)}`);
        }
    });

    log(`DIRECT: setup complete for ${filePath}`);
    return bridge;
}

async function openDirectWebview(filePath, title, pythonPath, shmParams, extraArgs) {
    const label = title || (filePath ? `ArrayView: ${path.basename(filePath)}` : (shmParams && shmParams.arrayName) || 'Array');

    const viewColumn = vscode.window.activeTextEditor
        ? vscode.ViewColumn.Beside
        : vscode.ViewColumn.Active;

    const panel = vscode.window.createWebviewPanel(
        'arrayview.viewer',
        label,
        { viewColumn, preserveFocus: false },
        {
            enableScripts: true,
            retainContextWhenHidden: true,
        }
    );

    try {
        await setupArrayViewPanel(panel, filePath, pythonPath, shmParams, extraArgs);
    } catch (e) {
        panel.dispose();
        throw e;
    }
    log(`DIRECT: opened "${label}" for ${filePath}`);
}

// ---------------------------------------------------------------------------
// Custom editor provider: "Open With..." from Explorer
// ---------------------------------------------------------------------------
class ArrayViewEditorProvider {
    static viewType = 'arrayview.arrayEditor';

    openCustomDocument(uri, _openContext, _token) {
        // Minimal document — just wraps the URI.  No data loading here;
        // PythonBridge handles everything in resolveCustomEditor.
        return { uri, dispose: () => {} };
    }

    async resolveCustomEditor(document, webviewPanel, _token) {
        const filePath = document.uri.fsPath;
        log(`CUSTOM-EDITOR: resolveCustomEditor for ${filePath}`);
        webviewPanel.webview.options = { enableScripts: true };
        webviewPanel.webview.html = `<html><body style="background:#1e1e1e;color:#ccc;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;font-family:system-ui">
            <div>Loading ${path.basename(filePath)}…</div></body></html>`;
        try {
            await setupArrayViewPanel(webviewPanel, filePath);
            log(`CUSTOM-EDITOR: setup complete for ${filePath}`);
        } catch (e) {
            log(`CUSTOM-EDITOR: error: ${e.message}\n${e.stack}`);
            webviewPanel.webview.html = `<html><body style="color:#c00;padding:2em;font-family:monospace">
                <h2>ArrayView failed to open</h2><pre>${e.message}</pre>
                <p>Check Output → "ArrayView" for details.</p></body></html>`;
        }
    }
}

let version = 'unknown';
let isProcessingSignal = false;
let lastHandledRequestId = null;
let lastHandledUrl = null;
let lastHandledAt = 0;

// Track open webview panels by URL so we can reveal instead of re-creating.
const _openPanels = new Map(); // url -> vscode.WebviewPanel

// ---------------------------------------------------------------------------
// SSH relay: accept array data from remote machines over TCP
// ---------------------------------------------------------------------------
const RELAY_PORT = parseInt(process.env.ARRAYVIEW_RELAY_PORT || '17789', 10);
const RELAY_MAGIC = Buffer.from('AVRELAY1');  // 8-byte handshake
const RELAY_MAX_SIZE = 2 * 1024 * 1024 * 1024;  // 2 GB
let relayServer = null;
const relayTempFiles = new Set();

function startRelayServer() {
    relayServer = net.createServer((socket) => {
        const chunks = [];
        let totalLen = 0;
        let aborted = false;

        socket.on('data', (chunk) => {
            totalLen += chunk.length;
            if (totalLen > RELAY_MAX_SIZE + 4096) {
                aborted = true;
                try { socket.end('{"error":"payload too large"}\n'); } catch (_) {}
                socket.destroy();
                return;
            }
            chunks.push(chunk);
        });

        socket.on('end', async () => {
            if (aborted) return;
            try {
                const buf = Buffer.concat(chunks);

                // Validate magic
                if (buf.length < 12 || !buf.subarray(0, 8).equals(RELAY_MAGIC)) {
                    socket.end('{"error":"invalid magic"}\n');
                    return;
                }

                // Parse header
                const headerLen = buf.readUInt32LE(8);
                if (buf.length < 12 + headerLen) {
                    socket.end('{"error":"incomplete header"}\n');
                    return;
                }
                let header;
                try {
                    header = JSON.parse(buf.subarray(12, 12 + headerLen).toString('utf-8'));
                } catch (e) {
                    socket.end('{"error":"malformed header JSON"}\n');
                    return;
                }

                // Extract .npy data
                const npyData = buf.subarray(12 + headerLen);
                if (npyData.length === 0) {
                    socket.end('{"error":"no array data"}\n');
                    return;
                }

                // Save to temp file
                const tmpName = 'arrayview-relay-' + crypto.randomBytes(8).toString('hex') + '.npy';
                const tmpFile = path.join(os.tmpdir(), tmpName);
                fs.writeFileSync(tmpFile, npyData);
                relayTempFiles.add(tmpFile);

                const title = header.title || header.name || 'Relayed Array';
                log(`RELAY: received ${npyData.length} bytes from ${socket.remoteAddress}, saved ${tmpName}`);

                // Open in direct webview (pythonPath=null → PythonBridge tries python3/python)
                try {
                    await openDirectWebview(tmpFile, `ArrayView: ${title}`, null, null);
                    socket.end('{"ok":true}\n');
                } catch (e) {
                    log(`RELAY: openDirectWebview failed: ${e.message}`);
                    socket.end(JSON.stringify({ error: e.message }) + '\n');
                }
            } catch (e) {
                log(`RELAY: error processing connection: ${e.message}`);
                try { socket.end(JSON.stringify({ error: e.message }) + '\n'); } catch (_) {}
            }
        });

        socket.on('error', (err) => {
            log(`RELAY: socket error: ${err.message}`);
        });
    });

    relayServer.on('error', (err) => {
        if (err.code === 'EADDRINUSE') {
            log(`RELAY: port ${RELAY_PORT} already in use, relay disabled`);
            relayServer = null;
        } else {
            log(`RELAY: server error: ${err.message}`);
        }
    });

    relayServer.listen(RELAY_PORT, '0.0.0.0', () => {
        log(`RELAY: listening on 0.0.0.0:${RELAY_PORT}`);
    });
}

function log(message) {
    const line = `[${new Date().toISOString()}] ${message}\n`;
    try { fs.appendFileSync(LOG_FILE, line); } catch (_) {}
    console.log(`[arrayview-opener] ${message}`);
}

function isExpiredSignal(data) {
    const sentAtMs = Number(data.sentAtMs || 0);
    const maxAgeMs = Number(data.maxAgeMs || 15000);
    if (!sentAtMs || maxAgeMs <= 0) return false;
    const ageMs = Date.now() - sentAtMs;
    if (ageMs <= maxAgeMs) return false;
    log(`SIGNAL: expired ageMs=${ageMs} maxAgeMs=${maxAgeMs}`);
    return true;
}

function isProcessAlive(pid) {
    try { process.kill(pid, 0); return true; } catch { return false; }
}

function cleanupStaleFiles() {
    // Remove stale .claimed-* and .tmp files left behind by crashes.
    // Also remove window-*.json registration files for dead processes.
    try {
        const files = fs.readdirSync(SIGNAL_DIR);
        for (const f of files) {
            if (f.startsWith('open-request-') && (f.includes('.claimed-') || f.endsWith('.tmp'))) {
                try {
                    fs.unlinkSync(path.join(SIGNAL_DIR, f));
                    log(`CLEANUP: removed stale file ${f}`);
                } catch (_) {}
            }
            if (f.startsWith('window-') && f.endsWith('.json')) {
                try {
                    const data = JSON.parse(fs.readFileSync(path.join(SIGNAL_DIR, f), 'utf8'));
                    if (data.pid && !isProcessAlive(data.pid)) {
                        fs.unlinkSync(path.join(SIGNAL_DIR, f));
                        log(`CLEANUP: removed stale registration ${f} (pid ${data.pid} dead)`);
                    }
                } catch (_) {}
            }
        }
    } catch (_) {}
}

// Shared-fallback files: any window may claim these, so we must verify hookTag.
const SHARED_FALLBACK_BASENAMES = new Set([
    path.basename(SIGNAL_FILE),
    'open-request-v0800.json',
    'open-request-v0400.json',
]);

async function tryOpenSignalFile() {
    // If we are currently showing a URL, leave any pending signal files on disk.
    // The 1-second polling loop will pick them up once we are done.  This avoids
    // in-memory queues that can be lost when the extension host reloads.
    if (isProcessingSignal) {
        log(`SKIP: isProcessingSignal=true`);
        return;
    }

    // Check targeted file first (matches our window's IPC hook or PID), then primary,
    // then compat signal files for older/published arrayview Python versions.
    const candidates = [];
    if (TARGETED_SIGNAL_FILE) candidates.push(TARGETED_SIGNAL_FILE);
    candidates.push(
        SIGNAL_FILE,
        path.join(SIGNAL_DIR, 'open-request-v0800.json'),
        path.join(SIGNAL_DIR, 'open-request-v0400.json'),
    );

    // Multi-window race mitigation: if this window is not focused, add a small delay
    // before claiming shared files. This gives the focused window a chance to claim first.
    const isFocused = vscode.window.state.focused;
    const isOwnTargetedFile = (f) => TARGETED_SIGNAL_FILE && f === TARGETED_SIGNAL_FILE;

    for (const signalFile of candidates) {
        // If not our targeted file and window not focused, delay briefly
        if (!isOwnTargetedFile(signalFile) && !isFocused) {
            await new Promise(resolve => setTimeout(resolve, 100));
        }

        const claimedFile = signalFile + '.claimed-' + process.pid;
        let raw;
        try {
            fs.renameSync(signalFile, claimedFile);
        } catch {
            continue;  // file doesn't exist or claimed by another window
        }
        try {
            raw = fs.readFileSync(claimedFile, 'utf8');
        } catch (e) {
            log(`ERROR: read claimed file failed: ${e.message}`);
            try { fs.unlinkSync(claimedFile); } catch (_) {}
            continue;
        }

        let data;
        try { data = JSON.parse(raw); } catch (err) {
            log(`SIGNAL: invalid JSON: ${err.message}`);
            try { fs.unlinkSync(claimedFile); } catch (_) {}
            continue;
        }

        // --- Multi-window guard ---
        // If a shared fallback file carries a hookTag that doesn't match ours,
        // it was written by Python for a different VS Code window.  Forward it
        // to that window's targeted file so the correct extension instance picks
        // it up, then skip processing here.
        const isSharedFallback = SHARED_FALLBACK_BASENAMES.has(path.basename(signalFile));
        if (isSharedFallback && data.hookTag && OWN_HOOK_TAG && data.hookTag !== OWN_HOOK_TAG) {
            log(`SIGNAL: hookTag mismatch (ours=${OWN_HOOK_TAG} signal=${data.hookTag}), forwarding to correct window`);
            const targetedFile = path.join(SIGNAL_DIR, `open-request-ipc-${data.hookTag}.json`);
            const tmp = targetedFile + '.tmp';
            try {
                fs.writeFileSync(tmp, JSON.stringify(data));
                fs.renameSync(tmp, targetedFile);
                log(`SIGNAL: forwarded to ${path.basename(targetedFile)}`);
            } catch (_) {}
            try { fs.unlinkSync(claimedFile); } catch (_) {}
            continue;
        }

        // --- Broadcast guard ---
        // If this signal is marked as broadcast (Python couldn't determine which window
        // to target), only process it if this window is currently focused. This ensures
        // only the active window opens the viewer when multiple windows are open.
        if (data.broadcast === true && !isFocused) {
            log(`SIGNAL: broadcast signal skipped (window not focused)`);
            try { fs.unlinkSync(claimedFile); } catch (_) {}
            continue;
        }

        try { fs.unlinkSync(claimedFile); } catch (_) {}

        if (isExpiredSignal(data)) continue;

        log(`DISPATCH: file=${path.basename(signalFile)} mode=${data.mode} hasFilepath=${!!data.filepath} hasShm=${!!data.shm} hasUrl=${!!data.url} keys=${Object.keys(data).join(',')}`);
        try {
            await processSignalData(data);
        } catch (error) {
            log(`ERROR: ${error.message}`);
        }
        return;  // processed one signal, done for this tick
    }
}

// Open or reveal a VS Code WebviewPanel for the given URL.
// Using createWebviewPanel instead of simpleBrowser.show lets us set a custom
// tab title (e.g. "ArrayView: sample.npy").  The webview just wraps the
// arrayview server URL in a full-page iframe.
function openInWebviewPanel(url, title) {
    const label = title || 'ArrayView';

    // Reveal existing panel for this URL if still open.
    const existing = _openPanels.get(url);
    if (existing) {
        try {
            existing.reveal(undefined, false);
            log(`PANEL: revealed existing panel for ${url}`);
            return;
        } catch (_) {
            _openPanels.delete(url);
        }
    }

    const viewColumn = vscode.window.activeTextEditor
        ? vscode.ViewColumn.Beside
        : vscode.ViewColumn.Active;

    const panel = vscode.window.createWebviewPanel(
        'arrayview.preview',
        label,
        { viewColumn, preserveFocus: false },
        {
            enableScripts: true,
            enableForms: true,
            retainContextWhenHidden: true,
        }
    );

    // Use a nonce so the CSP allows exactly our inline script and nothing else.
    // The iframe src is set via JS (not as an HTML attribute) to avoid any
    // attribute-encoding issues and to match VS Code's own Simple Browser pattern.
    const nonce = crypto.randomBytes(16).toString('hex');
    const jsonUrl = JSON.stringify(url); // safe JS string literal embedding

    panel.webview.html = `<!DOCTYPE html>
<html>
<head>
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; frame-src *; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';">
<style>
  html, body { margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; }
  iframe { position: fixed; top: 0; left: 0; width: 100%; height: 100%; border: none; }
</style>
</head>
<body>
<iframe id="f" sandbox="allow-scripts allow-forms allow-same-origin allow-downloads"></iframe>
<script nonce="${nonce}">document.getElementById('f').src = ${jsonUrl};</script>
</body>
</html>`;

    _openPanels.set(url, panel);
    panel.onDidDispose(() => _openPanels.delete(url));
    log(`PANEL: created "${label}" for ${url}`);
}

async function processSignalData(data) {
    isProcessingSignal = true;
    log(`LOCK: isProcessingSignal=true`);
    try {
        log(`SIGNAL-DATA: mode=${data.mode} filepath=${data.filepath || '(none)'} shm=${data.shm ? JSON.stringify(data.shm) : '(none)'} url=${data.url || '(none)'}`);
        // Direct webview mode: bypass iframe and use embedded viewer
        if (data.mode === 'direct' && (data.filepath || data.shm)) {
            // Clean up any compat signal duplicates to prevent double-processing
            for (const compat of ['open-request-v0800.json', 'open-request-v0400.json']) {
                try { fs.unlinkSync(path.join(SIGNAL_DIR, compat)); } catch (_) {}
            }
            const shmParams = data.shm ? { ...data.shm, arrayName: data.arrayName } : null;
            // Forward extra CLI args generically — Python builds this list from
            // its argparse namespace so new flags work without extension changes.
            const extraArgs = Array.isArray(data.extraArgs) ? data.extraArgs : [];
            await openDirectWebview(data.filepath, data.title, data.pythonPath, shmParams, extraArgs);
            return;
        }

        const url = data.url;
        if (!url) { log('SIGNAL: missing url'); return; }

        const requestId = data.requestId || null;
        const now = Date.now();
        if (requestId && requestId === lastHandledRequestId) {
            log(`SIGNAL: duplicate requestId ignored: ${requestId}`);
            return;
        }
        if (!requestId && url === lastHandledUrl && now - lastHandledAt < 5000) {
            log(`SIGNAL: duplicate url ignored within debounce window`);
            return;
        }

        log(`SIGNAL: requestId=${requestId || 'none'} url=${url} title=${data.title || '(none)'}`);
        lastHandledRequestId = requestId;
        lastHandledUrl = url;
        lastHandledAt = now;

        let openUrl = url;
        if (vscode.env.remoteName) {
            // Remote / tunnel: asExternalUri converts the localhost URL to the
            // public devtunnel URL (e.g. https://HOST-8000.euw.devtunnels.ms/).
            // VS Code strips query strings during this conversion, so we extract
            // ?sid=... from the original URL and re-append it manually.
            let port = 8000;
            try { port = parseInt(new URL(url).port, 10) || 8000; } catch (_) {}
            let origQuery = '';
            try { origQuery = new URL(url).search; } catch (_) {}

            try {
                const baseUri = vscode.Uri.parse(`http://localhost:${port}/`);
                log(`REMOTE: asExternalUri(http://localhost:${port}/)...`);
                const externalUri = await Promise.race([
                    vscode.env.asExternalUri(baseUri),
                    new Promise((_, reject) => setTimeout(() => reject(new Error('asExternalUri timeout after 8s')), 8000)),
                ]);
                const externalBase = externalUri.toString().replace(/\/$/, '');
                log(`REMOTE: → ${externalBase}`);
                openUrl = externalBase + '/' + origQuery;
                log(`REMOTE: final URL = ${openUrl}`);
            } catch (err) {
                log(`REMOTE: asExternalUri failed (${err.message}), using localhost`);
                openUrl = url;
            }
        }

        log(`openInWebviewPanel(${openUrl})`);
        openInWebviewPanel(openUrl, data.title);
        log('openInWebviewPanel done');
    } catch (error) {
        log(`ERROR: ${error.message}`);
    } finally {
        isProcessingSignal = false;
        log(`UNLOCK: isProcessingSignal=false`);
        // Signal files for subsequent arrays remain on disk; the 1-second poll
        // will pick them up now that isProcessingSignal is false again.
    }
}

function activate(context) {
    version = context.extension.packageJSON.version;
    log(`=== ACTIVATE v${version} ===`);
    log(`remoteName=${vscode.env.remoteName} appHost=${vscode.env.appHost}`);
    log(`ipcHook=${OWN_IPC_HOOK || 'NOT_SET'} hookTag=${OWN_HOOK_TAG || 'none'}`);
    if (TARGETED_SIGNAL_FILE) {
        log(`targetedFile=${path.basename(TARGETED_SIGNAL_FILE)}`);
    } else {
        log(`targetedFile=none (will use shared fallback only)`);
    }

    try { fs.mkdirSync(SIGNAL_DIR, { recursive: true }); } catch (_) {}

    // Inject ARRAYVIEW_WINDOW_ID into all terminals opened in this window.
    // Python reads this env var to know which targeted signal file to write,
    // solving multi-window targeting in tunnels where IPC hooks and PID
    // ancestry are shared across windows.
    const windowId = OWN_HOOK_TAG || String(process.pid);
    try {
        const envCollection = context.environmentVariableCollection;
        envCollection.replace('ARRAYVIEW_WINDOW_ID', windowId);
        log(`ENV: set ARRAYVIEW_WINDOW_ID=${windowId} in terminal env`);
    } catch (e) {
        log(`ENV: failed to set ARRAYVIEW_WINDOW_ID: ${e.message}`);
    }
    const regFile = path.join(SIGNAL_DIR, `window-${windowId}.json`);
    try {
        fs.writeFileSync(regFile, JSON.stringify({
            hookTag: OWN_HOOK_TAG || '',
            pid: process.pid,
            ppids: EXT_PPIDS,   // ancestor PIDs for multi-window matching by Python
            ts: Date.now(),
            fallbackId: !OWN_HOOK_TAG  // true if using PID fallback
        }));
        log(`REGISTER: wrote ${path.basename(regFile)} (${OWN_HOOK_TAG ? 'hookTag' : 'PID fallback'})`);
        context.subscriptions.push({ dispose: () => {
            try { fs.unlinkSync(regFile); } catch (_) {}
            log(`REGISTER: deleted ${path.basename(regFile)}`);
        }});
    } catch (e) {
        log(`REGISTER: failed to write: ${e.message}`);
    }

    cleanupStaleFiles();

    // Clean up stale registrations from previous tunnel sessions.
    // When the extension host restarts (e.g. tunnel reconnect), the old
    // process may still be alive but is no longer connected to a VS Code
    // client.  Detect by finding registrations with the same first ppid
    // (same server/tunnel) but an older timestamp.
    if (EXT_PPIDS.length >= 1) {
        try {
            const now = Date.now();
            for (const f of fs.readdirSync(SIGNAL_DIR)) {
                if (!f.startsWith('window-') || !f.endsWith('.json')) continue;
                const wid = f.slice(7, -5);
                if (wid === windowId) continue;
                try {
                    const data = JSON.parse(fs.readFileSync(path.join(SIGNAL_DIR, f), 'utf8'));
                    const sameTunnel = data.ppids && data.ppids.length >= 1 &&
                        data.ppids[0] === EXT_PPIDS[0];
                    if (sameTunnel && data.ts && data.ts < now - 5000) {
                        fs.unlinkSync(path.join(SIGNAL_DIR, f));
                        log(`CLEANUP: removed stale same-tunnel registration ${f} (ts=${data.ts})`);
                        // Also remove any stale signal files targeting that window
                        const prefix = data.fallbackId ? 'pid' : 'ipc';
                        try { fs.unlinkSync(path.join(SIGNAL_DIR, `open-request-${prefix}-${wid}.json`)); } catch (_) {}
                    }
                } catch (_) {}
            }
        } catch (_) {}
    }

    void tryOpenSignalFile();

    const interval = setInterval(() => void tryOpenSignalFile(), 1000);
    context.subscriptions.push({ dispose: () => clearInterval(interval) });

    try {
        const ownBasename = TARGETED_SIGNAL_FILE ? path.basename(TARGETED_SIGNAL_FILE) : null;
        const watcher = fs.watch(SIGNAL_DIR, (eventType, filename) => {
            if (!filename || filename.includes('.claimed-') || filename.endsWith('.tmp')) return;
            const isOwn = ownBasename && filename === ownBasename;
            const isFallback = filename === path.basename(SIGNAL_FILE) ||
                               filename === 'open-request-v0800.json' ||
                               filename === 'open-request-v0400.json';
            if (isOwn || isFallback) {
                log(`WATCH: event=${eventType} file=${filename}`);
                setTimeout(() => void tryOpenSignalFile(), 100);
            }
        });
        context.subscriptions.push({ dispose: () => watcher.close() });
        log(`WATCH: fs.watch active on ${SIGNAL_DIR}`);
    } catch (err) {
        log(`WATCH: fs.watch failed (polling still active): ${err.message}`);
    }

    // Register arrayview.openFile command for direct webview mode
    const openFileCmd = vscode.commands.registerCommand('arrayview.openFile', async (uri) => {
        let filePath;
        if (uri && uri.fsPath) {
            filePath = uri.fsPath;
        } else {
            const selected = await vscode.window.showOpenDialog({
                canSelectFiles: true,
                canSelectMany: false,
                filters: {
                    'Array files': ['npy', 'npz', 'nii', 'gz', 'h5', 'hdf5', 'zarr', 'mat', 'tif', 'tiff', 'png', 'jpg', 'jpeg'],
                },
            });
            if (!selected || !selected.length) return;
            filePath = selected[0].fsPath;
        }

        try {
            await openDirectWebview(filePath);
        } catch (e) {
            vscode.window.showErrorMessage(`ArrayView: ${e.message}`);
        }
    });
    context.subscriptions.push(openFileCmd);

    // Register custom editor provider for "Open With..." support
    const editorProvider = vscode.window.registerCustomEditorProvider(
        ArrayViewEditorProvider.viewType,
        new ArrayViewEditorProvider(),
        { webviewOptions: { retainContextWhenHidden: true } }
    );
    context.subscriptions.push(editorProvider);

    // Start SSH relay listener for zero-config remote array viewing
    startRelayServer();
    context.subscriptions.push({ dispose: () => {
        if (relayServer) { relayServer.close(); relayServer = null; }
    }});

    log('=== ACTIVATE DONE ===');
}

function deactivate() {
    if (relayServer) { relayServer.close(); relayServer = null; }
    // Clean up any remaining relay temp files
    for (const f of relayTempFiles) {
        try { fs.unlinkSync(f); } catch (_) {}
    }
    relayTempFiles.clear();
    log(`deactivate v${version}`);
}

module.exports = { activate, deactivate };
