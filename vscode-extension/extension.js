const vscode = require('vscode');
const fs = require('fs');
const path = require('path');
const os = require('os');
const crypto = require('crypto');
const http = require('http');
const https = require('https');
const { spawn, spawnSync } = require('child_process');
const {
    collectReleaseSidsFromUrl,
    pingUrlFromViewerUrl,
    sessionMetadataUrlFromViewerUrl,
    releaseUrlForSid,
    isVersionAtLeast,
    isLoopbackUrl,
    shouldDeferBroadcast,
    shouldRemoveSameTunnelRegistration,
    validatedAckPath,
    ackPayload,
    isTerminalAck,
    sameClaimOwner,
    claimJournalDisposition,
    isArrayViewStatus,
} = require('./lifecycle_helpers');

const SIGNAL_DIR = path.join(os.homedir(), '.arrayview');
const SIGNAL_FILE = path.join(SIGNAL_DIR, 'open-request-v0900.json');  // fallback
const LOG_FILE = path.join(SIGNAL_DIR, 'extension.log');
const EXTENSION_INSTANCE_ID = crypto.randomBytes(16).toString('hex');

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
    // Parent-process walk only works on Unix (uses ps).
    // On Windows, VSCODE_IPC_HOOK_CLI is already available directly in
    // the extension host process, and the extension injects
    // ARRAYVIEW_WINDOW_ID for terminal-to-window routing.
    if (process.platform === 'win32') return '';
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
// Declared as `let` so activate() can update it to the stable windowId (which may
// differ from process.pid when the env-collection ID is reused across restarts).
let TARGETED_SIGNAL_FILE = OWN_HOOK_TAG
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
        let ppid = 0;
        if (process.platform === 'win32') {
            try {
                const res = spawnSync('powershell', [
                    '-NoProfile', '-Command',
                    `(Get-CimInstance -ClassName Win32_Process -Filter "ProcessId=${p}").ParentProcessId`
                ], { encoding: 'utf8', timeout: 3000 });
                ppid = parseInt((res.stdout || '').trim(), 10);
            } catch (_) { break; }
        } else {
            const res = spawnSync('ps', ['-p', String(p), '-o', 'ppid='],
                { encoding: 'utf8', timeout: 1000 });
            ppid = parseInt((res.stdout || '').trim(), 10);
        }
        if (!ppid || ppid <= 1) break;
        result.push(ppid);
        p = ppid;
    }
    return result;
}
const EXT_PPIDS = _getAncestorPids(process.pid, 8);

let version = 'unknown';
let isProcessingSignal = false;
let logWindowId = '';
let lastHandledRequestId = null;
let lastHandledUrl = null;
let lastHandledAt = 0;
const _activeClaimedFiles = new Set();

// Track open webview panels by stable request identity. The externally resolved
// tunnel URL may change while one request is being recovered.
const _openPanels = new Map(); // request key (or URL for legacy calls) -> panel
const _readyPanels = new WeakSet();
const _publicTunnelUrls = new Map(); // port -> last externally reachable base URL
const TUNNEL_ROUTE_CACHE_FILE = path.join(SIGNAL_DIR, 'tunnel-routes.json');

function _cachedTunnelBases(port) {
    const candidates = [];
    const addCandidate = value => {
        if (typeof value !== 'string' || isLoopbackUrl(value)) return;
        try {
            const parsed = new URL(value);
            if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') return;
            const normalized = value.replace(/\/$/, '');
            if (!candidates.includes(normalized)) candidates.push(normalized);
        } catch (_) {}
    };
    const inMemory = _publicTunnelUrls.get(port);
    addCandidate(inMemory);
    try {
        const cache = JSON.parse(fs.readFileSync(TUNNEL_ROUTE_CACHE_FILE, 'utf8'));
        // Prefer this window's route, but retain verified routes across window
        // reloads.  VS Code's desktop tunnel resolver can return localhost
        // even when the provider still exposes the same public port route.
        addCandidate(cache[`${logWindowId}:${port}`]);
        for (const [key, value] of Object.entries(cache)) {
            if (key.endsWith(`:${port}`)) addCandidate(value);
        }
    } catch (_) {}
    return candidates;
}

async function _verifiedCachedTunnelBase(
    port,
    expectedServerId,
    ensureActive = () => {}
) {
    for (const candidate of _cachedTunnelBases(port)) {
        ensureActive();
        log(`REMOTE: checking cached route ${candidate}`);
        if (await arrayViewStatusOk(`${candidate}/ping`, expectedServerId)) {
            ensureActive();
            _rememberTunnelBase(port, candidate);
            log(`REMOTE: cached route ready for localhost:${port}`);
            return candidate;
        }
        log(`REMOTE: cached route stale for localhost:${port}`);
    }
    return null;
}

function _rememberTunnelBase(port, externalBase) {
    if (!externalBase || isLoopbackUrl(externalBase)) return;
    const normalized = externalBase.replace(/\/$/, '');
    _publicTunnelUrls.set(port, normalized);
    let cache = {};
    try {
        const parsed = JSON.parse(fs.readFileSync(TUNNEL_ROUTE_CACHE_FILE, 'utf8'));
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) cache = parsed;
    } catch (_) {}
    cache[`${logWindowId}:${port}`] = normalized;
    const tmp = `${TUNNEL_ROUTE_CACHE_FILE}.tmp-${process.pid}-${crypto.randomBytes(4).toString('hex')}`;
    try {
        fs.writeFileSync(tmp, JSON.stringify(cache));
        fs.renameSync(tmp, TUNNEL_ROUTE_CACHE_FILE);
    } catch (error) {
        try { fs.unlinkSync(tmp); } catch (_) {}
        log(`REMOTE: failed to cache tunnel route: ${error.message}`);
    }
}

function _withTimeout(promise, timeoutMs, label) {
    let timer;
    return Promise.race([
        promise,
        new Promise((_, reject) => {
            timer = setTimeout(() => reject(new Error(`${label} timeout after ${timeoutMs}ms`)), timeoutMs);
        }),
    ]).finally(() => clearTimeout(timer));
}

function _asExternalUriAttempt(baseUri) {
    // A timed-out VS Code resolver cannot be cancelled. Keep attempts
    // request-local and side-effect free so a hung promise cannot poison all
    // future launches for the same port.
    return Promise.resolve().then(() => vscode.env.asExternalUri(baseUri));
}

async function _boundedCommand(command, args, timeoutMs = 3000) {
    try {
        return await _withTimeout(
            vscode.commands.executeCommand(command, ...(args || [])),
            timeoutMs,
            command
        );
    } catch (error) {
        log(`REMOTE: ${command} unavailable: ${error.message}`);
        return null;
    }
}

function _tunnelItem(port) {
    return {
        // Match VS Code's stripped TunnelItem shape.  The privacy action
        // forwards this source back to the tunnel provider after closing the
        // old route, so its enum values must be the real workbench values.
        tunnelType: 'Forwarded',
        remoteHost: 'localhost',
        remotePort: port,
        localPort: port,
        name: 'ArrayView',
        source: { source: 0, description: 'User Forwarded' },
    };
}

function _publicBaseFromTunnelResult(result, expectedPort) {
    if (typeof result === 'string') {
        try {
            const parsed = new URL(result);
            if (
                (parsed.protocol === 'http:' || parsed.protocol === 'https:')
                && !isLoopbackUrl(result)
            ) {
                return result.replace(/\/$/, '');
            }
        } catch (_) {}
        return null;
    }
    if (!result || typeof result !== 'object') return null;
    const remotePort = Number(
        result.tunnelRemotePort ?? result.remotePort ?? 0
    );
    const remoteHost = String(
        result.tunnelRemoteHost ?? result.remoteHost ?? 'localhost'
    ).toLowerCase();
    if (remotePort !== Number(expectedPort) || result.privacy !== 'public') {
        return null;
    }
    if (!['localhost', '127.0.0.1', '::1'].includes(remoteHost)) return null;
    const candidates = [
        result.localAddress,
        result.tunnelLocalAddress,
        result.localUri,
    ];
    for (const candidate of candidates) {
        if (!candidate) continue;
        let value = typeof candidate === 'string'
            ? candidate
            : (typeof candidate.toString === 'function' ? candidate.toString() : '');
        if (value && !/^[a-z][a-z0-9+.-]*:/i.test(value)) {
            value = `${result.protocol || 'http'}://${value}`;
        }
        try {
            const parsed = new URL(value);
            if (
                (parsed.protocol === 'http:' || parsed.protocol === 'https:')
                && !isLoopbackUrl(parsed.toString())
            ) {
                return parsed.toString().replace(/\/$/, '');
            }
        } catch (_) {}
    }
    return null;
}

// Pending placeholder tabs from resolveCustomEditor, keyed by filePath.
// When a signal file arrives, we navigate the placeholder instead of
// creating a second panel, avoiding a visible flicker.
const _pendingPlaceholders = new Map(); // filePath -> { panel, basename }

function log(message) {
    const prefix = logWindowId ? `[${logWindowId.slice(0, 8)}] ` : '';
    const line = `[${new Date().toISOString()}] ${prefix}${message}\n`;
    try { fs.appendFileSync(LOG_FILE, line); } catch (_) {}
    console.log(`[arrayview-opener] ${prefix}${message}`);
}

function _claimOwner() {
    return {
        pid: process.pid,
        windowId: logWindowId,
        extensionInstanceId: EXTENSION_INSTANCE_ID,
        claimToken: crypto.randomBytes(16).toString('hex'),
    };
}

function _evidenceForClaimOwner(owner) {
    if (!owner || !owner.windowId) return { pidAlive: false, registration: null };
    const pidAlive = Boolean(owner.pid && isProcessAlive(owner.pid));
    try {
        const registration = JSON.parse(fs.readFileSync(
            path.join(SIGNAL_DIR, `window-${owner.windowId}.json`),
            'utf8'
        ));
        return {
            pidAlive,
            registration: {
                pid: registration.pid,
                windowId: owner.windowId,
                extensionInstanceId: registration.extensionInstanceId,
            },
        };
    } catch (_) {
        return { pidAlive, registration: null };
    }
}

function _atomicWriteJson(filePath, payload) {
    const tmpPath = `${filePath}.tmp-${process.pid}-${crypto.randomBytes(4).toString('hex')}`;
    try {
        fs.writeFileSync(tmpPath, JSON.stringify(payload));
        fs.renameSync(tmpPath, filePath);
    } catch (error) {
        try { fs.unlinkSync(tmpPath); } catch (_) {}
        throw error;
    }
}

function _writeClaimAck(ackPath, data, owner) {
    _atomicWriteJson(
        ackPath,
        ackPayload('claimed', data, logWindowId, null, version, owner)
    );
    return true;
}

function _acquireAckLock(lockPath, owner) {
    const tryAcquire = () => {
        let descriptor;
        try {
            descriptor = fs.openSync(lockPath, 'wx');
            fs.writeFileSync(descriptor, JSON.stringify(owner));
            fs.closeSync(descriptor);
            descriptor = null;
            return true;
        } catch (error) {
            if (descriptor !== undefined && descriptor !== null) {
                try { fs.closeSync(descriptor); } catch (_) {}
            }
            if (error.code !== 'EEXIST') throw error;
            return false;
        }
    };

    if (tryAcquire()) return true;
    let stale = false;
    try {
        const lockOwner = JSON.parse(fs.readFileSync(lockPath, 'utf8'));
        stale = claimJournalDisposition(
            { state: 'claimed', claimOwner: lockOwner },
            _evidenceForClaimOwner(lockOwner)
        ) === 'takeover';
    } catch (_) {
        try {
            stale = Date.now() - fs.statSync(lockPath).mtimeMs > 10000;
        } catch (__) {
            stale = false;
        }
    }
    if (!stale) return false;
    try { fs.unlinkSync(lockPath); } catch (_) { return false; }
    return tryAcquire();
}

function _releaseAckLock(lockPath, owner) {
    try {
        const lockOwner = JSON.parse(fs.readFileSync(lockPath, 'utf8'));
        if (sameClaimOwner(lockOwner, owner)) fs.unlinkSync(lockPath);
    } catch (_) {}
}

function writeProtocolAck(data, state, message) {
    if (data?.protocolVersion !== 1 || !data.requestId || !data.ackPath) return false;
    const ackPath = validatedAckPath(data.ackPath, data.requestId, os.homedir());
    if (!ackPath) {
        log(`ACK: rejected invalid path for requestId=${data.requestId}`);
        return false;
    }

    const owner = data.__claimOwner || null;
    const lockOwner = owner || _claimOwner();
    const lockPath = `${ackPath}.lock`;
    let acquired = false;
    try {
        acquired = _acquireAckLock(lockPath, lockOwner);
        if (!acquired) {
            log(`ACK: lock busy state=${state} requestId=${data.requestId}`);
            return false;
        }

        let existing = null;
        try { existing = JSON.parse(fs.readFileSync(ackPath, 'utf8')); } catch (_) {}
        if (isTerminalAck(existing)) {
            log(`ACK: preserving terminal state=${existing.state} requestId=${data.requestId}`);
            return true;
        }
        if (owner && !sameClaimOwner(owner, existing?.claimOwner)) {
            log(`ACK: fenced stale owner state=${state} requestId=${data.requestId}`);
            return false;
        }
        if (!owner && existing?.claimOwner) {
            log(`ACK: unowned write rejected state=${state} requestId=${data.requestId}`);
            return false;
        }

        _atomicWriteJson(
            ackPath,
            ackPayload(state, data, logWindowId, message, version, owner)
        );
        log(`ACK: state=${state} requestId=${data.requestId}`);
        return true;
    } catch (error) {
        log(`ACK: write failed state=${state}: ${error.message}`);
        return false;
    } finally {
        if (acquired) {
            _releaseAckLock(lockPath, lockOwner);
        }
    }
}

function _ownsProtocolClaim(data) {
    if (data?.protocolVersion !== 1) return true;
    const owner = data.__claimOwner;
    const existing = _ackForProtocolRequest(data);
    return sameClaimOwner(owner, existing?.claimOwner) && !isTerminalAck(existing);
}

function claimProtocolRequest(data) {
    if (data?.protocolVersion !== 1) return 'acquired';
    if (!data.requestId || !data.ackPath) return 'retry';
    const ackPath = validatedAckPath(data.ackPath, data.requestId, os.homedir());
    if (!ackPath) {
        log(`ACK: rejected invalid claim path for requestId=${data.requestId}`);
        return 'retry';
    }
    const owner = _claimOwner();
    const lockPath = `${ackPath}.lock`;
    let acquired = false;
    try {
        acquired = _acquireAckLock(lockPath, owner);
        if (!acquired) {
            log(`ACK: claim lock busy requestId=${data.requestId}`);
            return 'retry';
        }

        const ackExists = fs.existsSync(ackPath);
        let existing = null;
        try { existing = JSON.parse(fs.readFileSync(ackPath, 'utf8')); } catch (_) {}
        if (!ackExists) {
            _writeClaimAck(ackPath, data, owner);
            data.__claimOwner = owner;
            log(`ACK: state=claimed requestId=${data.requestId}`);
            return 'acquired';
        }

        const disposition = claimJournalDisposition(
            existing,
            _evidenceForClaimOwner(existing?.claimOwner)
        );
        if (disposition === 'takeover') {
            _writeClaimAck(ackPath, data, owner);
            data.__claimOwner = owner;
            log(`ACK: took over stale requestId=${data.requestId}`);
            return 'acquired';
        }
        log(`ACK: request disposition=${disposition} state=${existing?.state || 'unknown'} requestId=${data.requestId}`);
        return disposition === 'terminal' || disposition === 'active'
            ? 'duplicate'
            : 'retry';
    } catch (error) {
        log(`ACK: claim failed requestId=${data.requestId}: ${error.message}`);
        return 'retry';
    } finally {
        if (acquired) _releaseAckLock(lockPath, owner);
    }
}

function _shellCommand(command, args) {
    return [command, ...args].map((part) => {
        if (/^[A-Za-z0-9_./:=+-]+$/.test(part)) return part;
        return `'${String(part).replace(/'/g, `'\\''`)}'`;
    }).join(' ');
}

function _arrayviewLaunchCandidates(filePath) {
    const candidates = [];
    const owningFolder = vscode.workspace.getWorkspaceFolder(vscode.Uri.file(filePath));
    const folders = owningFolder ? [owningFolder] : (vscode.workspace.workspaceFolders || []);
    for (const folder of folders) {
        const isWin = process.platform === 'win32';
        const venvPy = isWin
            ? path.join(folder.uri.fsPath, '.venv', 'Scripts', 'python.exe')
            : path.join(folder.uri.fsPath, '.venv', 'bin', 'python');
        if (fs.existsSync(venvPy)) {
            candidates.push({ command: venvPy, argsPrefix: ['-m', 'arrayview'] });
            break;
        }
    }
    candidates.push({ command: 'uv', argsPrefix: ['run', '--python', '3.12', '--with', 'arrayview', 'python', '-m', 'arrayview'] });
    candidates.push({ command: 'python3', argsPrefix: ['-m', 'arrayview'] });
    return candidates;
}

function launchArrayViewFile(filePath, title) {
    const argsSuffix = [filePath, '--window', 'vscode'];
    if (title) argsSuffix.push('--name', title);

    return new Promise((resolve, reject) => {
        const candidates = _arrayviewLaunchCandidates(filePath);
        const owningFolder = vscode.workspace.getWorkspaceFolder(vscode.Uri.file(filePath));

        const tryNext = () => {
            const candidate = candidates.shift();
            if (!candidate) {
                reject(new Error('Python with arrayview not found. Install with: uv pip install -e . or pip install arrayview'));
                return;
            }

            const args = [...candidate.argsPrefix, ...argsSuffix];
            log(`PYTHON: launching ${_shellCommand(candidate.command, args)}`);
            let child;
            try {
                child = spawn(candidate.command, args, {
                    cwd: owningFolder?.uri.fsPath || path.dirname(filePath),
                    detached: true,
                    stdio: ['ignore', 'pipe', 'pipe'],
                    env: {
                        ...process.env,
                        TERM_PROGRAM: 'vscode',
                        ARRAYVIEW_WINDOW_ID: logWindowId || '',
                        ARRAYVIEW_HANDOFF_PATH: filePath,
                    },
                });
            } catch (error) {
                log(`PYTHON: launch failed for ${candidate.command}: ${error.message}`);
                tryNext();
                return;
            }

            let settled = false;
            const settleOk = () => {
                if (settled) return;
                settled = true;
                child.unref();
                resolve();
            };
            const settleRetry = (message) => {
                if (settled) return;
                settled = true;
                log(`PYTHON: ${message}`);
                tryNext();
            };

            child.stdout.on('data', (chunk) => {
                const text = chunk.toString().trim();
                if (text) log(`PYTHON: ${text}`);
                if (text.includes('http://localhost:') || text.includes('ArrayView')) {
                    settleOk();
                }
            });
            child.stderr.on('data', (chunk) => {
                const text = chunk.toString().trim();
                if (text) log(`PYTHON: ${text}`);
            });
            child.on('error', (error) => {
                settleRetry(`${candidate.command} failed: ${error.message}`);
            });
            child.on('exit', (code) => {
                if (settled) return;
                if (code === 0 || code === null) {
                    settleOk();
                } else {
                    settleRetry(`${candidate.command} exited with code ${code}`);
                }
            });

            setTimeout(settleOk, 1200);
        };

        tryNext();
    });
}

function isArrayViewCustomEditorTab(tab, uri = null) {
    const input = tab && tab.input;
    if (!input || input.viewType !== ArrayViewEditorProvider.viewType) {
        return false;
    }
    return !uri || input.uri.toString() === uri.toString();
}

function keepActiveArrayViewPreview(reason, uri = null) {
    const tab = vscode.window.tabGroups.activeTabGroup.activeTab;
    if (!isArrayViewCustomEditorTab(tab, uri)) {
        return false;
    }
    if (tab.isPreview === false) {
        return true;
    }
    void vscode.commands.executeCommand('workbench.action.keepEditor')
        .then(
            () => log(`CUSTOM-EDITOR: kept preview tab (${reason})`),
            (e) => log(`CUSTOM-EDITOR: keepEditor failed (${reason}): ${e.message}`)
        );
    return true;
}

function scheduleKeepArrayViewEditor(uri, reason) {
    for (const delay of [0, 50, 200, 750]) {
        setTimeout(() => keepActiveArrayViewPreview(`${reason}+${delay}ms`, uri), delay);
    }
}

async function closeActiveArrayViewCustomEditor(uri, reason) {
    const tab = vscode.window.tabGroups.activeTabGroup.activeTab;
    if (!isArrayViewCustomEditorTab(tab, uri)) {
        return false;
    }
    try {
        await vscode.commands.executeCommand('workbench.action.closeActiveEditor');
        log(`CUSTOM-EDITOR: closed placeholder (${reason})`);
        return true;
    } catch (e) {
        log(`CUSTOM-EDITOR: close placeholder failed (${reason}): ${e.message}`);
        return false;
    }
}

class ArrayViewEditorProvider {
    static viewType = 'arrayview.arrayEditor';

    openCustomDocument(uri, _openContext, _token) {
        return { uri, dispose: () => {} };
    }

    async resolveCustomEditor(document, webviewPanel, _token) {
        const filePath = document.uri.fsPath;
        const title = path.basename(filePath);
        log(`CUSTOM-EDITOR: resolveCustomEditor for ${filePath}`);
        // This custom editor is a handoff placeholder.  We keep it open and
        // navigate its webview when the signal-file URL arrives — no flicker.
        webviewPanel.webview.options = { enableScripts: true };
        webviewPanel.webview.html = `<html><body style="background:#1e1e1e;color:#ccc;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;font-family:ui-monospace,monospace">
            <div>Opening ${title} in ArrayView...</div></body></html>`;
        const placeholderKey = path.resolve(filePath);
        const placeholder = { panel: webviewPanel, basename: title, filePath: placeholderKey };
        _pendingPlaceholders.set(placeholderKey, placeholder);
        webviewPanel.onDidDispose(() => {
            if (_pendingPlaceholders.get(placeholderKey) === placeholder) {
                _pendingPlaceholders.delete(placeholderKey);
            }
            log(`CUSTOM-EDITOR: placeholder disposed for ${title}`);
        });
        // Large files may legitimately spend minutes loading before the URL is
        // ready. Keep the placeholder correlated for the whole launch budget.
        setTimeout(() => {
            if (_pendingPlaceholders.get(placeholderKey) === placeholder) {
                _pendingPlaceholders.delete(placeholderKey);
                try {
                    webviewPanel.webview.html = `<html><body style="color:#c00;padding:2em;font-family:monospace;background:#1e1e1e">
                        <h2>ArrayView failed to start</h2>
                        <p>The Python server did not respond. Check ~/.arrayview/extension.log for details.</p></body></html>`;
                } catch (_) { /* panel already disposed */ }
            }
        }, 190000);
        try {
            await launchArrayViewFile(filePath, title);
            log(`CUSTOM-EDITOR: launched network viewer for ${filePath}`);
        } catch (e) {
            if (_pendingPlaceholders.get(placeholderKey) === placeholder) {
                _pendingPlaceholders.delete(placeholderKey);
            }
            log(`CUSTOM-EDITOR: error: ${e.message}\n${e.stack || ''}`);
            webviewPanel.webview.html = `<html><body style="color:#c00;padding:2em;font-family:monospace">
                <h2>ArrayView failed to open</h2><pre>${e.message}</pre>
                <p>Check ~/.arrayview/extension.log for details.</p></body></html>`;
        }
    }
}

function arrayViewStatusOk(url, expectedServerId = null, timeoutMs = 1500) {
    return new Promise((resolve) => {
        let parsed;
        try {
            parsed = new URL(url);
        } catch (_) {
            resolve(false);
            return;
        }
        const lib = parsed.protocol === 'https:' ? https : http;
        let settled = false;
        const done = (ok) => {
            if (settled) return;
            settled = true;
            resolve(ok);
        };
        const req = lib.get(parsed, { timeout: timeoutMs }, (res) => {
            if (res.statusCode !== 200) {
                res.resume();
                done(false);
                return;
            }
            let body = '';
            res.setEncoding('utf8');
            res.on('data', chunk => {
                if (body.length < 65536) body += chunk;
            });
            res.on('end', () => {
                try {
                    const payload = JSON.parse(body);
                    done(isArrayViewStatus(payload, expectedServerId));
                } catch (_) {
                    done(false);
                }
            });
        });
        req.on('timeout', () => {
            req.destroy();
            done(false);
        });
        req.on('error', () => done(false));
    });
}

function httpStatus2xx(url, timeoutMs = 3000) {
    return new Promise((resolve) => {
        let parsed;
        try {
            parsed = new URL(url);
        } catch (_) {
            resolve(false);
            return;
        }
        const lib = parsed.protocol === 'https:' ? https : http;
        let settled = false;
        const done = (ok) => {
            if (settled) return;
            settled = true;
            resolve(ok);
        };
        const req = lib.get(parsed, { timeout: timeoutMs }, (res) => {
            res.resume();
            done((res.statusCode || 0) >= 200 && (res.statusCode || 0) < 300);
        });
        req.on('timeout', () => {
            req.destroy();
            done(false);
        });
        req.on('error', () => done(false));
    });
}

function httpJson(url, timeoutMs = 1500) {
    return new Promise((resolve) => {
        let parsed;
        try {
            parsed = new URL(url);
        } catch (_) {
            resolve(null);
            return;
        }
        const lib = parsed.protocol === 'https:' ? https : http;
        let settled = false;
        const done = (value) => {
            if (settled) return;
            settled = true;
            resolve(value);
        };
        const req = lib.get(parsed, { timeout: timeoutMs }, (res) => {
            if (res.statusCode !== 200) {
                res.resume();
                done(null);
                return;
            }
            let body = '';
            res.setEncoding('utf8');
            res.on('data', chunk => {
                if (body.length < 65536) body += chunk;
            });
            res.on('end', () => {
                try { done(JSON.parse(body)); } catch (_) { done(null); }
            });
        });
        req.on('timeout', () => {
            req.destroy();
            done(null);
        });
        req.on('error', () => done(null));
    });
}

function httpPostJson(url, payload, timeoutMs = 1500) {
    return new Promise((resolve) => {
        let parsed;
        try {
            parsed = new URL(url);
        } catch (_) {
            resolve(null);
            return;
        }
        const body = Buffer.from(JSON.stringify(payload));
        const lib = parsed.protocol === 'https:' ? https : http;
        let settled = false;
        const done = (value) => {
            if (settled) return;
            settled = true;
            resolve(value);
        };
        const req = lib.request(parsed, {
            method: 'POST',
            timeout: timeoutMs,
            headers: {
                'Content-Type': 'application/json',
                'Content-Length': String(body.length),
            },
        }, (res) => {
            if ((res.statusCode || 0) < 200 || (res.statusCode || 0) >= 300) {
                res.resume();
                done(null);
                return;
            }
            let responseBody = '';
            res.setEncoding('utf8');
            res.on('data', chunk => {
                if (responseBody.length < 65536) responseBody += chunk;
            });
            res.on('end', () => {
                try { done(JSON.parse(responseBody)); } catch (_) { done(null); }
            });
        });
        req.on('timeout', () => {
            req.destroy();
            done(null);
        });
        req.on('error', () => done(null));
        req.end(body);
    });
}

async function waitForHttpStatus2xx(url, timeoutMs = 150000, pollMs = 500) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        if (await httpStatus2xx(url)) return true;
        const remaining = deadline - Date.now();
        if (remaining > 0) {
            await new Promise(resolve => setTimeout(resolve, Math.min(pollMs, remaining)));
        }
    }
    return false;
}

function httpPostOk(url, timeoutMs = 1500, headers = {}) {
    return new Promise((resolve) => {
        let parsed;
        try {
            parsed = new URL(url);
        } catch (_) {
            resolve(false);
            return;
        }
        const lib = parsed.protocol === 'https:' ? https : http;
        let settled = false;
        const done = (ok) => {
            if (settled) return;
            settled = true;
            resolve(ok);
        };
        const req = lib.request(
            parsed,
            { method: 'POST', timeout: timeoutMs, headers },
            (res) => {
            res.resume();
            done((res.statusCode || 0) >= 200 && (res.statusCode || 0) < 300);
            }
        );
        req.on('timeout', () => {
            req.destroy();
            done(false);
        });
        req.on('error', () => done(false));
        req.end();
    });
}

function releaseUrlSession(url, backendUrl = null, serverId = null) {
    const sids = collectReleaseSidsFromUrl(url);
    if (!sids.length) return;
    for (const sid of sids) {
        const releaseUrl = releaseUrlForSid(url, backendUrl, sid);
        if (!releaseUrl) continue;
        const headers = serverId
            ? { 'X-ArrayView-Expected-Server-ID': serverId }
            : {};
        void httpPostOk(releaseUrl, 1500, headers).then((ok) => {
            log(`PANEL: release sid=${sid.slice(0, 8)} ok=${ok}`);
        });
    }
}

function isExpiredSignal(data) {
    const sentAtMs = Number(data.sentAtMs || 0);
    const maxAgeMs = Number(data.maxAgeMs || 15000);
    if (!sentAtMs || maxAgeMs <= 0) return false;
    const ageMs = Date.now() - sentAtMs;
    if (ageMs < maxAgeMs) return false;
    log(`SIGNAL: expired ageMs=${ageMs} maxAgeMs=${maxAgeMs}`);
    return true;
}

function _remainingSignalMs(data) {
    const sentAtMs = Number(data?.sentAtMs || 0);
    const maxAgeMs = Number(data?.maxAgeMs || 0);
    if (!sentAtMs || maxAgeMs <= 0) return null;
    return Math.max(0, sentAtMs + maxAgeMs - Date.now());
}

function isProcessAlive(pid) {
    try { process.kill(pid, 0); return true; } catch { return false; }
}

function _removeRegistrationIfOwned(regFile, owner) {
    try {
        const current = JSON.parse(fs.readFileSync(regFile, 'utf8'));
        const matches = current.pid === owner.pid
            && current.windowId === owner.windowId
            && current.extensionInstanceId === owner.extensionInstanceId;
        if (!matches) return false;
        fs.unlinkSync(regFile);
        return true;
    } catch (_) {
        return false;
    }
}

function _ackForProtocolRequest(data) {
    if (data?.protocolVersion !== 1 || !data.requestId || !data.ackPath) return null;
    const ackPath = validatedAckPath(data.ackPath, data.requestId, os.homedir());
    if (!ackPath) return null;
    try {
        return JSON.parse(fs.readFileSync(ackPath, 'utf8'));
    } catch (_) {
        return null;
    }
}

function _recoveryQueuePath(filename, data, ack) {
    const claimedWindowId = ack?.claimOwner?.windowId || null;
    if (claimedWindowId && claimedWindowId !== logWindowId) return null;
    if (data?.broadcast === true && claimedWindowId && TARGETED_SIGNAL_FILE) {
        const base = TARGETED_SIGNAL_FILE.replace(/\.json$/, '');
        return `${base}.request-${data.requestId}.json`;
    }
    const original = filename.replace(/\.claimed-\d+$/, '');
    return path.join(SIGNAL_DIR, original);
}

function _restoreClaimedFile(fullPath, filename, data, ack = null) {
    const queuePath = _recoveryQueuePath(filename, data, ack);
    if (!queuePath) {
        log(`CLEANUP: retained claim for window=${ack?.claimOwner?.windowId || 'unknown'} ${filename}`);
        return false;
    }
    if (!fs.existsSync(queuePath)) {
        fs.renameSync(fullPath, queuePath);
        log(`CLEANUP: restored interrupted claim ${filename} -> ${path.basename(queuePath)}`);
    } else {
        fs.unlinkSync(fullPath);
        log(`CLEANUP: removed duplicate claim ${filename} (queue copy exists)`);
    }
    return true;
}

function _deleteTerminalClaimedFile(claimedFile, data) {
    if (data?.protocolVersion !== 1) {
        try { fs.unlinkSync(claimedFile); } catch (_) {}
        return true;
    }
    if (!isTerminalAck(_ackForProtocolRequest(data))) return false;
    try {
        fs.unlinkSync(claimedFile);
        log(`JOURNAL: removed terminal claim ${path.basename(claimedFile)} requestId=${data.requestId}`);
    } catch (_) {}
    return true;
}

function _requeueOwnedClaim(claimedFile, signalFile, data) {
    if (data?.protocolVersion !== 1) {
        try { fs.unlinkSync(claimedFile); } catch (_) {}
        return true;
    }
    const ackPath = validatedAckPath(data.ackPath, data.requestId, os.homedir());
    const owner = data.__claimOwner;
    if (!ackPath || !owner) return false;
    const lockPath = `${ackPath}.lock`;
    let acquired = false;
    try {
        acquired = _acquireAckLock(lockPath, owner);
        if (!acquired) return false;
        const existing = _ackForProtocolRequest(data);
        if (isTerminalAck(existing)) return _deleteTerminalClaimedFile(claimedFile, data);
        if (!sameClaimOwner(owner, existing?.claimOwner)) return false;
        fs.unlinkSync(ackPath);
        return _restoreClaimedFile(
            claimedFile,
            path.basename(signalFile),
            data,
            existing
        );
    } catch (error) {
        log(`JOURNAL: requeue failed requestId=${data.requestId}: ${error.message}`);
        return false;
    } finally {
        if (acquired) {
            _releaseAckLock(lockPath, owner);
        }
    }
}

function _scheduleClaimedRecovery(claimedFile, signalFile, data, attempts = 3) {
    let remaining = attempts;
    const retry = () => {
        if (!fs.existsSync(claimedFile)) return;
        if (_deleteTerminalClaimedFile(claimedFile, data)) return;
        if (_requeueOwnedClaim(claimedFile, signalFile, data)) return;
        remaining -= 1;
        if (remaining > 0) setTimeout(retry, 1000);
        else log(`JOURNAL: recovery deferred to scanner requestId=${data.requestId || 'none'}`);
    };
    setTimeout(retry, 250);
}

function _expireProtocolRequest(data, existingAck) {
    if (data?.protocolVersion !== 1 || !isExpiredSignal(data)) return false;
    const ackPath = validatedAckPath(data.ackPath, data.requestId, os.homedir());
    if (!ackPath) return false;
    const fenceOwner = _claimOwner();
    const lockPath = `${ackPath}.lock`;
    let acquired = false;
    try {
        acquired = _acquireAckLock(lockPath, fenceOwner);
        if (!acquired) return false;
        const latest = _ackForProtocolRequest(data);
        if (isTerminalAck(latest)) return true;
        const payload = ackPayload(
            'failed',
            data,
            latest?.windowId || existingAck?.windowId || data.windowId || logWindowId,
            'Signal expired during extension-host recovery',
            version,
            fenceOwner
        );
        _atomicWriteJson(ackPath, payload);
        log(`ACK: fenced expired requestId=${data.requestId}`);
        return true;
    } catch (error) {
        log(`ACK: expiry fencing failed requestId=${data.requestId}: ${error.message}`);
        return false;
    } finally {
        if (acquired) {
            _releaseAckLock(lockPath, fenceOwner);
        }
    }
}

function cleanupStaleFiles() {
    // Recover or remove stale .claimed-* files left behind by crashes, and
    // remove incomplete .tmp files. Also remove window-*.json registration
    // files for dead processes.
    //
    // A .claimed-* file is produced by tryOpenSignalFile renaming a signal
    // file just before processing it. If the extension host died mid-process,
    // that signal was never shown and the file sits here forever. Rather than
    // deleting it (which loses the user's open request), restore non-expired
    // ones to their original basename so the 1s poll re-claims and re-shows
    // them on the next activate — only delete expired ones.
    try {
        const files = fs.readdirSync(SIGNAL_DIR);
        for (const f of files) {
            if (f.startsWith('open-request-') && f.endsWith('.tmp')) {
                try { fs.unlinkSync(path.join(SIGNAL_DIR, f)); log(`CLEANUP: removed stale tmp ${f}`); } catch (_) {}
                continue;
            }
            if (f.startsWith('open-request-') && f.includes('.claimed-')) {
                const fullPath = path.join(SIGNAL_DIR, f);
                if (_activeClaimedFiles.has(fullPath)) {
                    log(`CLEANUP: retained in-flight claim ${f}`);
                    continue;
                }
                try {
                    const data = JSON.parse(fs.readFileSync(fullPath, 'utf8'));
                    const ack = _ackForProtocolRequest(data);
                    if (isTerminalAck(ack)) {
                        fs.unlinkSync(fullPath);
                        log(`CLEANUP: removed terminal claim ${f}`);
                        continue;
                    }

                    if (isExpiredSignal(data)) {
                        if (_expireProtocolRequest(data, ack)) {
                            _deleteTerminalClaimedFile(fullPath, data);
                            log(`CLEANUP: failed expired interrupted claim ${f}`);
                        } else {
                            log(`CLEANUP: retained expired claim pending safe fencing ${f}`);
                        }
                        continue;
                    }

                    let disposition = 'unknown';
                    if (ack?.claimOwner) {
                        disposition = claimJournalDisposition(
                            ack,
                            _evidenceForClaimOwner(ack.claimOwner)
                        );
                    } else {
                        const suffixPid = Number((f.match(/\.claimed-(\d+)$/) || [])[1] || 0);
                        const activeLegacyOwner = suffixPid > 0
                            && suffixPid !== process.pid
                            && isProcessAlive(suffixPid);
                        disposition = activeLegacyOwner ? 'active' : 'takeover';
                    }
                    if (disposition === 'active' || disposition === 'unknown') {
                        log(`CLEANUP: retained ${disposition} claim ${f}`);
                        continue;
                    }

                    if (!ack && data?.ackPath) {
                        const corruptAckPath = validatedAckPath(
                            data.ackPath, data.requestId, os.homedir()
                        );
                        if (corruptAckPath && fs.existsSync(corruptAckPath)) {
                            try { fs.unlinkSync(corruptAckPath); } catch (_) {}
                        }
                    }
                    _restoreClaimedFile(fullPath, f, data, ack);
                } catch (_) {
                    try { fs.unlinkSync(fullPath); log(`CLEANUP: removed unparseable claimed ${f}`); } catch (__) {}
                }
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

function _targetedSignalPath(hookTag, data) {
    const base = path.join(SIGNAL_DIR, `open-request-ipc-${hookTag}.json`);
    if (data?.protocolVersion === 1 && data.requestId) {
        return base.replace(/\.json$/, `.request-${data.requestId}.json`);
    }
    return base;
}

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
    if (TARGETED_SIGNAL_FILE) {
        const base = path.basename(TARGETED_SIGNAL_FILE, '.json');
        try {
            const queued = fs.readdirSync(SIGNAL_DIR)
                .filter(name => name.startsWith(`${base}.request-`) && name.endsWith('.json'))
                .sort((a, b) => {
                    try {
                        return fs.statSync(path.join(SIGNAL_DIR, a)).mtimeMs -
                               fs.statSync(path.join(SIGNAL_DIR, b)).mtimeMs;
                    } catch (_) { return a.localeCompare(b); }
                });
            candidates.push(...queued.map(name => path.join(SIGNAL_DIR, name)));
        } catch (_) {}
        candidates.push(TARGETED_SIGNAL_FILE);
    }
    try {
        const sharedBase = path.basename(SIGNAL_FILE, '.json');
        const sharedQueued = fs.readdirSync(SIGNAL_DIR)
            .filter(name => name.startsWith(`${sharedBase}.request-`) && name.endsWith('.json'))
            .sort((a, b) => {
                try {
                    return fs.statSync(path.join(SIGNAL_DIR, a)).mtimeMs -
                           fs.statSync(path.join(SIGNAL_DIR, b)).mtimeMs;
                } catch (_) { return a.localeCompare(b); }
            });
        candidates.push(...sharedQueued.map(name => path.join(SIGNAL_DIR, name)));
    } catch (_) {}
    // Drain queued compatibility copies too. New Python versions write these
    // so an older opener can still claim the request; a current opener must
    // remove the duplicate after the primary v0900 request completes.
    try {
        for (const compatBase of ['open-request-v0800', 'open-request-v0400']) {
            const queued = fs.readdirSync(SIGNAL_DIR)
                .filter(name => name.startsWith(`${compatBase}.request-`) && name.endsWith('.json'))
                .sort((a, b) => {
                    try {
                        return fs.statSync(path.join(SIGNAL_DIR, a)).mtimeMs -
                               fs.statSync(path.join(SIGNAL_DIR, b)).mtimeMs;
                    } catch (_) { return a.localeCompare(b); }
                });
            candidates.push(...queued.map(name => path.join(SIGNAL_DIR, name)));
        }
    } catch (_) {}
    candidates.push(
        SIGNAL_FILE,
        path.join(SIGNAL_DIR, 'open-request-v0800.json'),
        path.join(SIGNAL_DIR, 'open-request-v0400.json'),
    );

    // Multi-window race mitigation: if this window is not focused, add a small delay
    // before claiming shared files. This gives the focused window a chance to claim first.
    const isFocused = vscode.window.state.focused;
    const isOwnTargetedFile = (f) => TARGETED_SIGNAL_FILE && (
        f === TARGETED_SIGNAL_FILE ||
        path.basename(f).startsWith(`${path.basename(TARGETED_SIGNAL_FILE, '.json')}.request-`)
    );

    for (const signalFile of candidates) {
        // An untargeted broadcast belongs to whichever VS Code window is
        // focused. Leave it untouched here so an unfocused extension host
        // cannot win the filesystem race and discard the request.
        if (!isOwnTargetedFile(signalFile) && !isFocused) {
            try {
                const pending = JSON.parse(fs.readFileSync(signalFile, 'utf8'));
                if (shouldDeferBroadcast(false, isFocused, pending)) continue;
            } catch (_) {}
        }

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

        // Tunnel launches belong to the workspace extension host. A local
        // desktop host can share ~/.arrayview, but cannot resolve the remote
        // port and would otherwise open a known-bad localhost tab.
        if (data.remoteOnly === true && !vscode.env.remoteName) {
            log(`SIGNAL: remote-only signal deferred by local extension host`);
            try {
                fs.renameSync(claimedFile, signalFile);
            } catch (_) {
                try { fs.unlinkSync(claimedFile); } catch (_) {}
            }
            continue;
        }

        if (data.requiredExtensionVersion && !isVersionAtLeast(version, data.requiredExtensionVersion)) {
            const message = `Stale ArrayView opener v${version}; v${data.requiredExtensionVersion} is required. Reload this VS Code window.`;
            log(`SIGNAL: ${message}`);
            writeProtocolAck(data, 'failed', message);
            try { fs.unlinkSync(claimedFile); } catch (_) {}
            continue;
        }

        // --- Multi-window guard ---
        // If a shared fallback file carries a hookTag that doesn't match ours,
        // it was written by Python for a different VS Code window.  Forward it
        // to that window's targeted file so the correct extension instance picks
        // it up, then skip processing here.
        const signalBasename = path.basename(signalFile);
        const isSharedFallback = SHARED_FALLBACK_BASENAMES.has(signalBasename) ||
            signalBasename.startsWith(`${path.basename(SIGNAL_FILE, '.json')}.request-`) ||
            signalBasename.startsWith('open-request-v0800.request-') ||
            signalBasename.startsWith('open-request-v0400.request-');
        if (isSharedFallback && data.hookTag && OWN_HOOK_TAG && data.hookTag !== OWN_HOOK_TAG) {
            log(`SIGNAL: hookTag mismatch (ours=${OWN_HOOK_TAG} signal=${data.hookTag}), forwarding to correct window`);
            const targetedFile = _targetedSignalPath(data.hookTag, data);
            const tmp = `${targetedFile}.tmp-${process.pid}-${crypto.randomBytes(4).toString('hex')}`;
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
        if (data.broadcast === true && !isOwnTargetedFile(signalFile) && !isFocused) {
            log(`SIGNAL: broadcast signal skipped (window not focused)`);
            try { fs.unlinkSync(claimedFile); } catch (_) {}
            continue;
        }

        if (isExpiredSignal(data)) {
            if (data.protocolVersion !== 1) {
                try { fs.unlinkSync(claimedFile); } catch (_) {}
            } else if (_expireProtocolRequest(data, _ackForProtocolRequest(data))) {
                _deleteTerminalClaimedFile(claimedFile, data);
            } else {
                log(`JOURNAL: retained expired pre-claim request ${path.basename(claimedFile)}`);
            }
            continue;
        }

        const claimResult = claimProtocolRequest(data);
        if (claimResult !== 'acquired') {
            if (claimResult === 'duplicate') {
                try { fs.unlinkSync(claimedFile); } catch (_) {}
            } else if (!_restoreClaimedFile(
                claimedFile,
                path.basename(signalFile),
                data,
                _ackForProtocolRequest(data)
            )) {
                log(`JOURNAL: retained indeterminate claim ${path.basename(claimedFile)}`);
            }
            continue;
        }

        // A broadcast request is written to both the current and compatibility
        // queue names. Once one window claims it, remove the sibling copies so
        // another focused tunnel window cannot open the same SID later.
        if (data.requestId) {
            for (const prefix of ['open-request-v0900', 'open-request-v0800', 'open-request-v0400']) {
                const duplicate = path.join(SIGNAL_DIR, `${prefix}.request-${data.requestId}.json`);
                if (duplicate === signalFile) continue;
                try {
                    fs.unlinkSync(duplicate);
                    log(`SIGNAL: removed compatibility copy ${path.basename(duplicate)}`);
                } catch (_) {}
            }
        }

        if (isExpiredSignal(data)) {
            writeProtocolAck(data, 'failed', 'Signal expired before processing');
            _deleteTerminalClaimedFile(claimedFile, data);
            continue;
        }

        log(`DISPATCH: file=${path.basename(signalFile)} mode=${data.mode} hasUrl=${!!data.url} keys=${Object.keys(data).join(',')}`);
        _activeClaimedFiles.add(claimedFile);
        try {
            await processSignalData(data);
        } catch (error) {
            log(`ERROR: ${error.message}`);
            writeProtocolAck(data, 'failed', error.message);
        } finally {
            _activeClaimedFiles.delete(claimedFile);
        }
        if (!_deleteTerminalClaimedFile(claimedFile, data)) {
            log(`JOURNAL: requeueing non-terminal claim ${path.basename(claimedFile)} requestId=${data.requestId || 'none'}`);
            if (!_requeueOwnedClaim(claimedFile, signalFile, data)) {
                _scheduleClaimedRecovery(claimedFile, signalFile, data);
            }
        }
        return;  // processed one signal, done for this tick
    }
}

// Open or reveal a VS Code WebviewPanel for the given server URL.
// The panel is only a URL wrapper: ArrayView data and controls still flow
// through the FastAPI/WebSocket backend, never direct Python/webview IPC.
function _viewerPanelHtml(url, warmupUrl = null, warmupTimeoutMs = 12000) {
    const nonce = crypto.randomBytes(16).toString('hex');
    const jsonUrl = JSON.stringify(url);
    const jsonWarmupUrl = JSON.stringify(warmupUrl);
    const jsonWarmupTimeoutMs = JSON.stringify(warmupTimeoutMs);
    return `<!DOCTYPE html>
<html>
<head>
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; connect-src http: https:; frame-src *; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';">
<style>
  html, body { margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; }
  iframe { position: fixed; top: 0; left: 0; width: 100%; height: 100%; border: none; }
  #backend-error {
    box-sizing: border-box;
    display: none;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    padding: 32px;
    background: #101010;
    color: #e6e6e6;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  }
  #backend-error.visible { display: flex; }
  #backend-error .box { max-width: 720px; line-height: 1.5; }
  #backend-error h2 { margin: 0 0 12px; font-size: 18px; font-weight: 600; }
  #backend-error p { margin: 8px 0; color: #bdbdbd; }
  #backend-error code { color: #f5c842; word-break: break-all; }
</style>
</head>
<body>
<iframe id="f" allow="clipboard-read; clipboard-write; fullscreen"></iframe>
<div id="backend-error">
  <div class="box">
    <h2>ArrayView backend is not responding</h2>
    <p>The VS Code tab opened, but the local ArrayView server for this view is unavailable.</p>
    <p>Close this tab and run the command again. If it keeps happening, check that the terminal command is still running or that another process did not take the port.</p>
    <p><code id="backend-url"></code></p>
  </div>
</div>
<script nonce="${nonce}">
const vscodeApi = acquireVsCodeApi();
const arrayviewUrl = ${jsonUrl};
const warmupUrl = ${jsonWarmupUrl};
const warmupTimeoutMs = ${jsonWarmupTimeoutMs};
const frame = document.getElementById('f');
vscodeApi.postMessage({ type: 'panel-phase', phase: 'wrapper-started' });
let viewerReady = false;
let viewerLoaded = false;
let reloadTimer = null;
let reloadCount = 0;
const MAX_RELOADS = 12;
const RELOAD_DELAY_MS = 1500;
function showBackendError() {
    if (viewerReady) return;
    if (reloadTimer) { clearTimeout(reloadTimer); reloadTimer = null; }
    document.getElementById('backend-url').textContent = arrayviewUrl;
    document.getElementById('backend-error').classList.add('visible');
    frame.style.display = 'none';
}
function scheduleReload() {
    if (viewerReady || viewerLoaded) return;
    if (reloadTimer) { clearTimeout(reloadTimer); }
    reloadTimer = setTimeout(() => {
        reloadTimer = null;
        if (viewerReady || viewerLoaded) return;
        if (reloadCount >= MAX_RELOADS) { showBackendError(); return; }
        reloadCount++;
        console.log('[arrayview-opener] iframe reload ' + reloadCount + ' (viewer not ready)');
        const sep = arrayviewUrl.includes('?') ? '&' : '?';
        frame.src = arrayviewUrl + sep + '_avretry=' + reloadCount;
    }, RELOAD_DELAY_MS);
}
window.addEventListener('message', (event) => {
    const msg = event && event.data;
    if (msg && msg.type === 'backend-error') {
        console.log('[arrayview-opener] viewer reported backend-error');
        showBackendError();
        return;
    }
    if (!msg || msg.source !== 'arrayview-viewer') return;
    vscodeApi.postMessage({ type: 'viewer-phase', phase: msg.phase || 'unknown' });
    if (msg.phase === 'script-loaded') {
        viewerLoaded = true;
        if (reloadTimer) { clearTimeout(reloadTimer); reloadTimer = null; }
        console.log('[arrayview-opener] viewer script loaded; waiting for first frame');
        return;
    }
    if (msg.phase === 'frame-rendered') {
        if (!viewerReady) {
            viewerLoaded = true;
            viewerReady = true;
            if (reloadTimer) { clearTimeout(reloadTimer); reloadTimer = null; }
            console.log('[arrayview-opener] viewer phase ' + msg.phase);
            vscodeApi.postMessage({ type: 'viewer-ready', phase: msg.phase });
        }
    }
});
frame.addEventListener('load', () => {
    console.log('[arrayview-opener] iframe loaded ' + arrayviewUrl);
    vscodeApi.postMessage({ type: 'panel-phase', phase: 'iframe-loaded' });
    scheduleReload();
});
frame.addEventListener('error', () => console.log('[arrayview-opener] iframe error ' + arrayviewUrl));
async function warmTransportAndOpen() {
    if (warmupUrl) {
        vscodeApi.postMessage({ type: 'panel-phase', phase: 'transport-warmup-started' });
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), warmupTimeoutMs);
        try {
            await fetch(warmupUrl, {
                mode: 'no-cors',
                cache: 'no-store',
                signal: controller.signal,
            });
            vscodeApi.postMessage({ type: 'panel-phase', phase: 'transport-warmup-complete' });
        } catch (error) {
            vscodeApi.postMessage({
                type: 'panel-phase',
                phase: 'transport-warmup-failed',
                message: String(error && error.message ? error.message : error),
            });
        } finally {
            clearTimeout(timeout);
        }
    }
    frame.src = arrayviewUrl;
}
void warmTransportAndOpen();
</script>
</body>
</html>`;
}

function waitForViewerReady(panel, timeoutMs = 25000) {
    return new Promise((resolve) => {
        let settled = false;
        let messageSubscription = null;
        let disposeSubscription = null;
        let timer = null;
        const finish = (error = null) => {
            if (settled) return;
            settled = true;
            if (timer) clearTimeout(timer);
            if (messageSubscription) messageSubscription.dispose();
            if (disposeSubscription) disposeSubscription.dispose();
            resolve(error);
        };
        messageSubscription = panel.webview.onDidReceiveMessage((message) => {
            if (message?.type === 'panel-phase' || message?.type === 'viewer-phase') {
                log(`PANEL: ${message.type} ${message.phase || 'unknown'}`);
            }
            if (message?.type === 'viewer-ready' && message.phase === 'frame-rendered') {
                finish();
            }
        });
        disposeSubscription = panel.onDidDispose(() => {
            finish(new Error('Viewer panel closed before its first frame rendered'));
        });
        timer = setTimeout(() => {
            finish(new Error('Viewer did not render a frame before timeout'));
        }, timeoutMs);
    });
}

function _integratedBrowserLaunchUrl(
    url,
    requestId,
    serverId,
    windowId,
    token
) {
    try {
        const parsed = new URL(url);
        parsed.searchParams.set('_av_integrated_browser', '1');
        parsed.searchParams.set('_av_launch_request_id', requestId);
        parsed.searchParams.set('_av_launch_server_id', serverId);
        parsed.searchParams.set('_av_launch_window_id', windowId);
        parsed.searchParams.set('_av_launch_token', token);
        return parsed.toString();
    } catch (_) {
        return null;
    }
}

async function waitForBackendViewerReady(
    backendUrl,
    sid,
    requestId,
    serverId,
    windowId,
    token,
    timeoutMs,
    ensureActive = () => {},
    retryPreScriptNavigation = null,
    preScriptTimeoutMs = 10000
) {
    let statusUrl;
    try {
        statusUrl = `${new URL(backendUrl).origin}/viewer-phase/${encodeURIComponent(sid)}/${encodeURIComponent(requestId)}`;
    } catch (_) {
        return new Error('Unable to derive viewer phase journal URL');
    }
    const required = ['script-loaded', 'ws-open', 'metadata-loaded', 'frame-rendered'];
    const logged = new Set();
    const deadline = Date.now() + timeoutMs;
    const preScriptDeadline = Date.now() + Math.min(
        timeoutMs,
        Math.max(1, preScriptTimeoutMs)
    );
    let activeToken = token;
    let scriptLoaded = false;
    let navigationAttempt = 0;
    const firstNavigationRetryDelayMs = Math.min(
        1500,
        Math.max(50, Math.floor(preScriptTimeoutMs * 0.25))
    );
    const laterNavigationRetryDelayMs = Math.min(
        3000,
        Math.max(50, Math.floor(preScriptTimeoutMs * 0.35))
    );
    let nextNavigationRetryAt = Date.now() + firstNavigationRetryDelayMs;
    const maxNavigationRetries = 2;
    while (Date.now() < deadline) {
        ensureActive();
        const activeDeadline = scriptLoaded
            ? deadline
            : Math.min(deadline, preScriptDeadline);
        const payload = await httpJson(
            `${statusUrl}?token=${encodeURIComponent(activeToken)}`,
            Math.max(1, Math.min(1500, activeDeadline - Date.now()))
        );
        if (
            payload
            && payload.sid === sid
            && payload.request_id === requestId
            && payload.server_id === serverId
            && payload.window_id === windowId
            && payload.token === activeToken
            && Array.isArray(payload.phases)
            && Array.isArray(payload.viewer_instance_ids)
        ) {
            scriptLoaded = payload.phases.includes('script-loaded');
            for (const phase of payload.phases) {
                if (!logged.has(phase)) {
                    logged.add(phase);
                    log(`PANEL: viewer-phase ${phase} (backend journal)`);
                }
            }
            if (payload.phases.includes('frame-rendered')) {
                if (payload.viewer_instance_ids.length !== 1) {
                    return new Error(
                        `Integrated browser opened ${payload.viewer_instance_ids.length} viewer instances for one request`
                    );
                }
                let previous = -1;
                for (const phase of required) {
                    const index = payload.phases.indexOf(phase);
                    if (index <= previous) {
                        return new Error(
                            `Viewer phase journal reached first frame out of order: ${payload.phases.join(' -> ')}`
                        );
                    }
                    previous = index;
                }
                return null;
            }
        }
        if (!scriptLoaded && Date.now() >= preScriptDeadline) {
            return new Error(
                'Integrated browser did not start the viewer script before recovery timeout'
            );
        }
        if (
            !scriptLoaded
            && retryPreScriptNavigation
            && navigationAttempt < maxNavigationRetries
            && Date.now() >= nextNavigationRetryAt
            && deadline - Date.now() > 500
        ) {
            navigationAttempt += 1;
            ensureActive();
            let replacementToken = null;
            try {
                replacementToken = await retryPreScriptNavigation(
                    navigationAttempt,
                    deadline
                );
            } catch (error) {
                log(`PANEL: pre-script navigation retry failed: ${error.message || error}`);
            }
            ensureActive();
            if (replacementToken) activeToken = replacementToken;
            nextNavigationRetryAt = Date.now() + laterNavigationRetryDelayMs;
        }
        const remaining = (
            scriptLoaded ? deadline : Math.min(deadline, preScriptDeadline)
        ) - Date.now();
        if (remaining > 0) {
            await new Promise(resolve => setTimeout(resolve, Math.min(100, remaining)));
        }
    }
    return new Error('Integrated browser did not render a frame before timeout');
}

async function openInIntegratedBrowser(
    url,
    backendUrl,
    requestId,
    serverId,
    windowId,
    viewerTimeoutMs,
    ensureActive = () => {},
    preScriptTimeoutMs = 10000
) {
    const viewerDeadline = Date.now() + viewerTimeoutMs;
    ensureActive();
    const remoteProxyEnabled = vscode.workspace
        .getConfiguration('workbench.browser')
        .get('enableRemoteProxy', false);
    log(`PANEL: integrated browser remoteProxy=${remoteProxyEnabled}`);
    const sid = collectReleaseSidsFromUrl(backendUrl)[0] || null;
    if (!sid || !requestId || !serverId || !windowId) {
        throw new Error('Integrated browser launch is missing correlated viewer identity');
    }
    // A replay must navigate the existing request tab but prove readiness from
    // the newly navigated document.  The reuse filter deliberately excludes
    // this fresh token; the backend journal is reset before navigation.
    // With remote proxy enabled the browser resolves localhost in the remote
    // workspace and must use the backend URL.  Otherwise it runs on the client
    // and must use the client-forwarded asExternalUri URL.
    const browserUrl = remoteProxyEnabled ? backendUrl : url;
    const journalUrl = `${new URL(backendUrl).origin}/viewer-phase/${encodeURIComponent(sid)}/${encodeURIComponent(requestId)}`;
    const reuseUrlFilter = `?_av_launch_request_id=${encodeURIComponent(requestId)}`;
    const prepareNavigation = async (navigationAttempt = 0, deadline = null) => {
        ensureActive();
        const token = crypto.randomBytes(16).toString('hex');
        let launchUrl = _integratedBrowserLaunchUrl(
            browserUrl,
            requestId,
            serverId,
            windowId,
            token
        );
        if (!launchUrl) throw new Error('Unable to build integrated browser launch URL');
        if (navigationAttempt > 0) {
            const parsed = new URL(launchUrl);
            parsed.searchParams.set('_av_navigation_attempt', String(navigationAttempt));
            launchUrl = parsed.toString();
        }
        const prepared = await httpPostJson(
            journalUrl,
            {
                phase: 'launch-prepared',
                server_id: serverId,
                window_id: windowId,
                token,
            },
            Math.max(1, Math.min(1500, viewerDeadline - Date.now()))
        );
        if (
            !prepared
            || prepared.request_id !== requestId
            || prepared.server_id !== serverId
            || prepared.window_id !== windowId
            || prepared.token !== token
        ) {
            throw new Error('Unable to prepare correlated viewer readiness journal');
        }
        const commandPromise = vscode.commands.executeCommand('workbench.action.browser.open', {
            url: launchUrl,
            // Each invocation gets a new browser tab in the preferred group.
            // openToSide=true creates and locks a new editor group per launch,
            // eventually leaving VS Code unable to load another browser page.
            openToSide: false,
            // Retry/replay of this exact request reuses its one tab, while every
            // distinct ArrayView invocation opens a fresh browser tab.
            reuseUrlFilter,
        });
        if (navigationAttempt > 0 && deadline !== null) {
            try {
                await _withTimeout(
                    commandPromise,
                    Math.max(1, Math.min(3000, deadline - Date.now())),
                    'integrated browser pre-script navigation'
                );
            } catch (error) {
                // The command may already have dispatched before its promise
                // stalls. Keep polling this fresh token, then use the one
                // remaining bounded retry if no script reports readiness.
                log(`PANEL: pre-script navigation command unavailable: ${error.message || error}`);
            }
        } else {
            await commandPromise;
        }
        return token;
    };
    let commandAttempted = false;
    let token;
    try {
        commandAttempted = true;
        token = await prepareNavigation();
        ensureActive();
    } catch (error) {
        if (commandAttempted) releaseUrlSession(url, backendUrl, serverId);
        throw error;
    }
    log(`PANEL: browser-command-completed transport=integrated-browser`);
    log(`PANEL: integrated browser opened ${browserUrl}`);
    return {
        viewerReady: waitForBackendViewerReady(
            backendUrl,
            sid,
            requestId,
            serverId,
            windowId,
            token,
            Math.max(1, viewerDeadline - Date.now()),
            ensureActive,
            async (navigationAttempt, deadline) => {
                const remaining = deadline - Date.now();
                if (remaining <= 0) return null;
                if (navigationAttempt === 1) {
                    log(`PANEL: retrying pre-script navigation attempt=${navigationAttempt}`);
                    return prepareNavigation(navigationAttempt, deadline);
                }
                log(`PANEL: hard-reloading exact request tab after pre-script stall`);
                await _withTimeout(
                    vscode.commands.executeCommand('workbench.action.browser.open', {
                        reuseUrlFilter,
                    }),
                    Math.max(1, Math.min(3000, remaining)),
                    'integrated browser exact-tab reveal'
                );
                ensureActive();
                await _withTimeout(
                    vscode.commands.executeCommand('workbench.action.browser.hardReload'),
                    Math.max(1, Math.min(3000, deadline - Date.now())),
                    'integrated browser hard reload'
                );
                return null;
            },
            preScriptTimeoutMs
        ),
    };
}

async function integratedBrowserCommandAvailable(timeoutMs = 1500) {
    try {
        const commands = await _withTimeout(
            vscode.commands.getCommands(true),
            timeoutMs,
            'integrated browser command discovery'
        );
        return commands.includes('workbench.action.browser.open');
    } catch (error) {
        // Command enumeration can block on an unrelated extension host. The
        // built-in command itself remains safe to attempt and will reject
        // clearly if this VS Code version truly does not provide it.
        log(`PANEL: command discovery unavailable; attempting integrated browser directly: ${error.message || error}`);
        return true;
    }
}

function _backendPortMapping(displayUrl, backendUrl) {
    try {
        const display = new URL(displayUrl);
        const parsed = new URL(backendUrl);
        if (!['localhost', '127.0.0.1'].includes(display.hostname.toLowerCase())) return null;
        if (parsed.hostname.toLowerCase() !== 'localhost') return null;
        if (display.protocol !== 'http:' && display.protocol !== 'https:') return null;
        if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') return null;
        let displayPort;
        if (display.port) {
            displayPort = Number(display.port);
        } else if (display.protocol === 'http:') {
            displayPort = 80;
        } else {
            displayPort = 443;
        }
        let backendPort;
        if (parsed.port) {
            backendPort = Number(parsed.port);
        } else if (parsed.protocol === 'http:') {
            backendPort = 80;
        } else if (parsed.protocol === 'https:') {
            backendPort = 443;
        } else {
            return null;
        }
        if (!Number.isInteger(displayPort) || displayPort < 1 || displayPort > 65535) return null;
        if (!Number.isInteger(backendPort) || backendPort < 1 || backendPort > 65535) return null;
        // The wrapper warms this exact client-visible port before navigating
        // its nested iframe.  Keeping the real display port is essential:
        // VS Code does not remap WebSocket ports, and ArrayView's viewer uses
        // the document origin for both HTTP and WebSocket traffic.
        return [{ webviewPort: displayPort, extensionHostPort: backendPort }];
    } catch (_) {
        return null;
    }
}

function _replaceWebviewPortMapping(webview, portMapping) {
    const options = { ...(webview.options || {}) };
    delete options.portMapping;
    if (portMapping) options.portMapping = portMapping;
    webview.options = options;
}

async function openInWebviewPanel(
    url,
    title,
    floating = false,
    backendUrl = null,
    requestKey = null,
    serverId = null,
    viewerTimeoutMs = 25000
) {
    const label = title || 'ArrayView';
    const panelKey = requestKey || url;
    const portMapping = _backendPortMapping(url, backendUrl);
    const warmupTimeoutMs = Math.max(0, Math.min(12000, viewerTimeoutMs - 1000));
    const warmupUrl = warmupTimeoutMs > 0 ? pingUrlFromViewerUrl(url) : null;

    // Reveal/reconcile the existing logical panel for this request. A replay
    // may resolve the same backend SID through a new external tunnel URL.
    const existing = _openPanels.get(panelKey);
    if (existing) {
        try {
            if (existing.__arrayviewUrl !== url) {
                _readyPanels.delete(existing);
                const viewerReady = waitForViewerReady(existing, viewerTimeoutMs).then((error) => {
                    if (!error) _readyPanels.add(existing);
                    return error;
                });
                existing.__arrayviewUrl = url;
                _replaceWebviewPortMapping(existing.webview, portMapping);
                existing.webview.html = _viewerPanelHtml(
                    url,
                    warmupUrl,
                    warmupTimeoutMs
                );
                existing.title = label;
                existing.reveal(undefined, false);
                log(`PANEL: reconciled existing request panel to ${url}`);
                return viewerReady;
            }
            existing.reveal(undefined, false);
            log(`PANEL: revealed existing panel for ${url}`);
            return _readyPanels.has(existing)
                ? Promise.resolve(null)
                : waitForViewerReady(existing, viewerTimeoutMs);
        } catch (_) {
            _openPanels.delete(panelKey);
        }
    }

    const viewColumn = vscode.window.activeTextEditor
        ? vscode.ViewColumn.Beside
        : vscode.ViewColumn.Active;

    const webviewOptions = {
        enableScripts: true,
        enableForms: true,
        retainContextWhenHidden: true,
    };
    if (portMapping) webviewOptions.portMapping = portMapping;

    const panel = vscode.window.createWebviewPanel(
        'arrayview.preview',
        label,
        { viewColumn, preserveFocus: false },
        webviewOptions
    );

    const viewerReady = waitForViewerReady(panel, viewerTimeoutMs).then((error) => {
        if (!error) _readyPanels.add(panel);
        return error;
    });
    panel.webview.html = _viewerPanelHtml(url, warmupUrl, warmupTimeoutMs);
    panel.__arrayviewUrl = url;

    _openPanels.set(panelKey, panel);
    const pingUrl = pingUrlFromViewerUrl(portMapping ? backendUrl : url);
    let panelDisposed = false;
    panel.onDidDispose(() => {
        panelDisposed = true;
        if (_openPanels.get(panelKey) === panel) {
            _openPanels.delete(panelKey);
            releaseUrlSession(url, backendUrl, serverId);
        } else {
            log(`PANEL: ignored disposal from superseded request panel ${panelKey}`);
        }
    });
    log(`PANEL: created "${label}" for ${url}`);

    if (pingUrl) {
        setTimeout(async () => {
            for (let attempt = 0; attempt <= 10 && !panelDisposed; attempt++) {
                if (await arrayViewStatusOk(pingUrl)) return;
                await new Promise(resolve => setTimeout(resolve, 1500));
            }
            if (!panelDisposed) {
                await panel.webview.postMessage({ type: 'backend-error', url });
            }
        }, 3500);
    }

    const cfg = vscode.workspace.getConfiguration('arrayview');
    if ((floating || cfg.get('openInFloatingWindow')) && vscode.env.uiKind !== vscode.UIKind.Web) {
        panel.reveal();
        try {
            await vscode.commands.executeCommand('workbench.action.moveEditorToNewWindow');
        } catch (e) {
            log(`FLOAT: moveEditorToNewWindow failed: ${e}`);
        }
    }
    return viewerReady;
}

/**
 * Ensure a forwarded port has public visibility so the devtunnel URL is
 * accessible from the VS Code client.  VS Code auto-forwards ports as
 * private by default; the devtunnel URL only works if the port is public.
 *
 * VS Code's remote.portsAttributes schema does not support privacy, so do
 * not persist a no-op entry for every dynamically selected ArrayView port.
 * Change the live forward through `remote.tunnel.privacypublic`.  This
 * closes the existing private tunnel and re-forwards with public visibility.
 * The command is registered lazily by the Ports view when the provider
 * supports privacy changes (devtunnels do).
 */
async function ensurePortPublic(
    port,
    externalBase,
    expectedServerId = null,
    ensureActive = () => {}
) {
    ensureActive();
    const hasExternalRoute = !isLoopbackUrl(externalBase);
    const publicPingUrl = `${externalBase}/ping`;
    if (hasExternalRoute && await arrayViewStatusOk(publicPingUrl, expectedServerId)) {
        _rememberTunnelBase(port, externalBase);
        log(`PORT: verified public route for ${externalBase}`);
        return true;
    }
    _publicTunnelUrls.delete(port);

    // Change privacy of the already-forwarded port.
    // The privacy command (remote.tunnel.privacypublic) is lazily
    // registered by VS Code's Forwarded Ports view.  In a pure tunnel
    // session (no Remote-SSH), it may not be loaded yet.  Try focusing
    // the forwarded ports view first to trigger lazy loading, then retry.
    const tunnelItem = _tunnelItem(port);

    let privacyDone = false;
    let promotedExternalBase = null;
    try {
        ensureActive();
        const result = await _boundedCommand(
            'remote.tunnel.privacypublic', [tunnelItem]
        );
        if (result && typeof result === 'object') {
            log(`PORT: privacy result ${JSON.stringify({
                remotePort: result.tunnelRemotePort ?? result.remotePort ?? null,
                remoteHost: result.tunnelRemoteHost ?? result.remoteHost ?? null,
                localAddress: result.localAddress ?? result.tunnelLocalAddress ?? null,
                privacy: result.privacy ?? null,
                protocol: result.protocol ?? null,
            })}`);
            promotedExternalBase = _publicBaseFromTunnelResult(result, port);
        } else {
            const detail = typeof result === 'string'
                ? JSON.stringify(result.slice(0, 500))
                : (result === null ? 'null' : typeof result);
            log(`PORT: privacy command returned ${detail}`);
            promotedExternalBase = _publicBaseFromTunnelResult(result, port);
        }
        if (promotedExternalBase) {
            privacyDone = true;
            log(`PORT: changed privacy to public via command`);
            log(`PORT: privacy command returned ${promotedExternalBase}`);
        }
    } catch (e) {
        log(`PORT: privacy command failed: ${e.message || e}`);
    }
    ensureActive();

    if (!privacyDone) {
        // Retry: force-load forwarded ports view, then retry the command
        log(`PORT: privacy not found — loading forwarded ports view...`);
        try {
            ensureActive();
            await _boundedCommand('~remote.forwardedPorts.focus', [], 2000);
            await new Promise(r => setTimeout(r, 500));
        } catch (_) {}
        ensureActive();

        // Check if the command is now registered
        const cmds = await _withTimeout(
            vscode.commands.getCommands(true),
            3000,
            'get tunnel commands'
        );
        if (cmds.includes('remote.tunnel.privacypublic')) {
            try {
                ensureActive();
                log(`PORT: privacy command found after view load — retrying`);

                // Do not call asExternalUri again here.  It starts another
                // forward for the same port; VS Code suppresses the privacy
                // action's replacement forward while that factory operation
                // is still in progress, making the command resolve undefined.
                const result = await _boundedCommand(
                    'remote.tunnel.privacypublic', [tunnelItem]
                );
                if (result && typeof result === 'object') {
                    log(`PORT: privacy retry result ${JSON.stringify({
                        remotePort: result.tunnelRemotePort ?? result.remotePort ?? null,
                        remoteHost: result.tunnelRemoteHost ?? result.remoteHost ?? null,
                        localAddress: result.localAddress ?? result.tunnelLocalAddress ?? null,
                        privacy: result.privacy ?? null,
                        protocol: result.protocol ?? null,
                    })}`);
                    promotedExternalBase = _publicBaseFromTunnelResult(result, port);
                } else {
                    const detail = typeof result === 'string'
                        ? JSON.stringify(result.slice(0, 500))
                        : (result === null ? 'null' : typeof result);
                    log(`PORT: privacy retry returned ${detail}`);
                    promotedExternalBase = _publicBaseFromTunnelResult(result, port);
                }
                if (promotedExternalBase) {
                    privacyDone = true;
                    log(`PORT: changed privacy to public via command (retry)`);
                    log(`PORT: privacy command returned ${promotedExternalBase}`);
                }
            } catch (e2) {
                log(`PORT: privacy retry failed: ${e2.message || e2}`);
            }
        } else {
            log(`PORT: privacypublic still not available after view load`);
        }
        ensureActive();
    }
    if (!privacyDone) return false;

    if (promotedExternalBase) {
        const promotedPingUrl = `${promotedExternalBase}/ping`;
        const deadline = Date.now() + 20000;
        while (Date.now() < deadline) {
            ensureActive();
            if (await arrayViewStatusOk(promotedPingUrl, expectedServerId)) {
                _rememberTunnelBase(port, promotedExternalBase);
                log(`PORT: public route ready for ${promotedExternalBase}`);
                return promotedExternalBase;
            }
            await new Promise(resolve => setTimeout(resolve, 500));
        }
        log(`PORT: returned public route did not become ready for ${promotedExternalBase}`);
        return false;
    }

    // A tunnel resolver can return localhost until the forward has been
    // promoted.  In that case the local /ping says nothing about client
    // reachability: promotion succeeded, but the caller must resolve again to
    // obtain and verify the new non-loopback route.
    if (!hasExternalRoute) {
        log(`PORT: promoted localhost:${port}; waiting for external URI retry`);
        return true;
    }

    const deadline = Date.now() + 20000;
    while (Date.now() < deadline) {
        ensureActive();
        if (await arrayViewStatusOk(publicPingUrl, expectedServerId)) {
            _rememberTunnelBase(port, externalBase);
            log(`PORT: public route ready for ${externalBase}`);
            return true;
        }
        await new Promise(resolve => setTimeout(resolve, 500));
    }
    log(`PORT: public route did not become ready for ${externalBase}`);
    return false;
}

async function resolveRemoteViewerUrl(
    url,
    expectedServerId = null,
    ensureActive = () => {}
) {
    ensureActive();
    let port = 8000;
    try { port = parseInt(new URL(url).port, 10) || 8000; } catch (_) {}
    let origQuery = '';
    try { origQuery = new URL(url).search; } catch (_) {}
    const desktopTunnelRemoteProxy = (
        vscode.env.remoteName === 'tunnel'
        && vscode.env.appHost === 'desktop'
        && vscode.workspace
            .getConfiguration('workbench.browser')
            .get('enableRemoteProxy', false)
    );
    if (desktopTunnelRemoteProxy) {
        const backendPingUrl = pingUrlFromViewerUrl(url);
        if (
            backendPingUrl
            && await arrayViewStatusOk(backendPingUrl, expectedServerId)
        ) {
            ensureActive();
            log(`REMOTE: desktop integrated-browser proxy uses backend URL directly`);
            return url;
        }
        log(`REMOTE: desktop integrated-browser proxy cannot reach expected backend`);
        return null;
    }
    const baseUri = vscode.Uri.parse(`http://localhost:${port}/`);
    const cachedBase = vscode.env.remoteName === 'tunnel'
        ? await _verifiedCachedTunnelBase(port, expectedServerId, ensureActive)
        : null;
    if (cachedBase) {
        return cachedBase + '/' + origQuery;
    }
    let tunnelPromotionAttempted = false;
    const attempts = [
        { timeoutMs: 6000, pauseMs: 0 },
        { timeoutMs: 10000, pauseMs: 500 },
        { timeoutMs: 10000, pauseMs: 1500 },
        { timeoutMs: 10000, pauseMs: 3000 },
        { timeoutMs: 10000, pauseMs: 5000 },
        { timeoutMs: 10000, pauseMs: 8000 },
    ];
    for (let i = 0; i < attempts.length; i++) {
        ensureActive();
        const attempt = attempts[i];
        if (attempt.pauseMs) {
            await new Promise(resolve => setTimeout(resolve, attempt.pauseMs));
        }
        try {
            ensureActive();
            log(`REMOTE: asExternalUri(http://localhost:${port}/) attempt=${i + 1}`);
            const externalUri = await _withTimeout(
                _asExternalUriAttempt(baseUri),
                attempt.timeoutMs,
                'asExternalUri'
            );
            ensureActive();
            const externalBase = externalUri.toString().replace(/\/$/, '');
            log(`REMOTE: → ${externalBase}`);

            if (vscode.env.remoteName === 'tunnel' && isLoopbackUrl(externalBase)) {
                if (vscode.env.appHost === 'desktop' && desktopTunnelRemoteProxy) {
                    ensureActive();
                    if (!await arrayViewStatusOk(
                        `${externalBase}/ping`, expectedServerId
                    )) {
                        throw new Error(
                            'desktop tunnel loopback does not reach the expected backend'
                        );
                    }
                    ensureActive();
                    const finalUrl = externalBase + '/' + origQuery;
                    log(`REMOTE: desktop tunnel remote-proxy URL = ${finalUrl}`);
                    return finalUrl;
                }
                if (!tunnelPromotionAttempted) {
                    tunnelPromotionAttempted = true;
                    // Let auto-forwarders in all connected windows finish
                    // materializing their default/private route.  Privacy
                    // promotion must be the final forwarding operation.
                    await new Promise(resolve => setTimeout(resolve, 1500));
                    const promoted = await ensurePortPublic(
                        port, externalBase, expectedServerId, ensureActive
                    );
                    if (typeof promoted === 'string') {
                        ensureActive();
                        const finalUrl = promoted + '/' + origQuery;
                        log(`REMOTE: promotion returned final URL = ${finalUrl}`);
                        return finalUrl;
                    } else if (promoted) {
                        log(`REMOTE: tunnel route promoted without a returned public URL`);
                    } else {
                        log(`REMOTE: tunnel promotion did not return a verified public URL`);
                    }
                    // Promotion may activate a provider route without exposing
                    // its address through the command API.  Re-check routes
                    // learned by prior incarnations of this window, accepting
                    // one only when /ping identifies this exact backend.
                    const recoveredBase = await _verifiedCachedTunnelBase(
                        port, expectedServerId, ensureActive
                    );
                    if (recoveredBase) {
                        const finalUrl = recoveredBase + '/' + origQuery;
                        log(`REMOTE: recovered verified cached URL = ${finalUrl}`);
                        return finalUrl;
                    }
                    // Never call asExternalUri after the final privacy action:
                    // that recreates the route as private.  A failed verified
                    // promotion is terminal for this request.
                    return null;
                } else {
                    log(`REMOTE: tunnel route still loopback after promotion; waiting before retry`);
                }
                continue;
            }

            if (vscode.env.remoteName === 'tunnel') {
                const publicReady = await ensurePortPublic(
                    port, externalBase, expectedServerId, ensureActive
                );
                if (!publicReady) {
                    throw new Error('public tunnel route is not ready');
                }
                _rememberTunnelBase(port, externalBase);
            }

            const finalUrl = externalBase + '/' + origQuery;
            log(`REMOTE: final URL = ${finalUrl}`);
            return finalUrl;
        } catch (err) {
            log(`REMOTE: asExternalUri attempt ${i + 1} failed: ${err.message}`);
        }
    }

    if (vscode.env.remoteName === 'tunnel' && tunnelPromotionAttempted) {
        log(`REMOTE: tunnel route did not converge within bounded retry window`);
    }
    return null;
}

async function processSignalData(data) {
    isProcessingSignal = true;
    log(`LOCK: isProcessingSignal=true`);
    // Hard safety net: if any await inside the body hangs (e.g. VS Code's
    // createWebviewPanel / openInWebviewPanel never resolves when the
    // extension host is degraded), the finally below would never run and
    // isProcessingSignal would stick true — every subsequent signal would
    // be skipped at tryOpenSignalFile's guard until the user reloaded the
    // window. Racing the whole body against a timeout guarantees the lock
    // always releases so the 1s poll picks up queued signals again. The
    // The cancellation flag prevents a timed-out body from opening a panel or
    // overwriting the terminal failure ACK after the queue lock is released.
    const remainingSignalMs = _remainingSignalMs(data);
    const signalHardTimeoutMs = remainingSignalMs === null
        ? 185000
        : Math.max(1000, remainingSignalMs + 1000);
    const operation = { cancelled: false };
    let hardTimer = null;
    try {
        await Promise.race([
            _processSignalDataBody(data, operation),
            new Promise((_, reject) =>
                hardTimer = setTimeout(() => {
                    operation.cancelled = true;
                    reject(new Error(`processSignalData hard timeout after ${signalHardTimeoutMs}ms`));
                }, signalHardTimeoutMs)
            ),
        ]);
    } catch (error) {
        log(`ERROR: ${error.message}`);
        writeProtocolAck(data, 'failed', error.message);
    } finally {
        if (hardTimer) clearTimeout(hardTimer);
        isProcessingSignal = false;
        log(`UNLOCK: isProcessingSignal=false`);
        // Signal files for subsequent arrays remain on disk; the 1-second poll
        // will pick them up now that isProcessingSignal is false again.
    }
}

async function _processSignalDataBody(data, operation = { cancelled: false }) {
    const ensureActive = () => {
        if (operation.cancelled) {
            throw new Error('Signal processing was cancelled before panel open');
        }
        if (isExpiredSignal(data)) {
            operation.cancelled = true;
            _expireProtocolRequest(data, _ackForProtocolRequest(data));
            throw new Error('Signal expired before display side effect');
        }
        if (!_ownsProtocolClaim(data)) {
            throw new Error('Signal claim ownership was lost before display side effect');
        }
    };
    const advanceAck = (state, message = null) => {
        ensureActive();
        if (!writeProtocolAck(data, state, message)) {
            throw new Error(`Failed to persist ${state} launch progress`);
        }
    };
    log(`SIGNAL-DATA: mode=${data.mode} url=${data.url || '(none)'}`);
    const url = data.url;
    if (!url) {
        log('SIGNAL: missing url');
        writeProtocolAck(data, 'failed', 'Signal is missing url');
        return;
    }

    const localMetadataUrl = sessionMetadataUrlFromViewerUrl(url);
    // A newly spawned daemon creates the session asynchronously.  During
    // loading /metadata/<sid> is temporarily 404; treating that as expired
    // races large remote-file loads and loses the viewer before its panel can
    // open.
    const remainingMs = _remainingSignalMs(data);
    const metadataWaitMs = remainingMs === null
        ? 150000
        : Math.max(1, Math.min(150000, remainingMs));
    if (
        localMetadataUrl
        && !await waitForHttpStatus2xx(localMetadataUrl, metadataWaitMs)
    ) {
        ensureActive();
        throw new Error('Viewer session expired before a panel could be opened; retrying the command will create a fresh session');
    }
    ensureActive();

    const requestId = data.requestId || null;
    const panelKey = requestId ? `request:${requestId}` : null;
    const now = Date.now();
    if (data.protocolVersion !== 1 && requestId && requestId === lastHandledRequestId) {
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
        // Remote / tunnel: asExternalUri forwards the port and returns the
        // devtunnel URL (e.g. https://HOST-8000.euw.devtunnels.ms/).
        // VS Code strips query strings during this conversion, so we extract
        // ?sid=... from the original URL and re-append it manually.
        //
        // The forward is created as Private by default.  A Private devtunnel
        // redirects to Microsoft/GitHub auth, which the Simple Browser iframe
        // cannot complete (CSP frame-ancestors:none) — producing a blank
        // tab.  We flip the forward to Public after asExternalUri creates it.
        //
        // Timing: remote.tunnel.privacypublic only works after the forward
        // exists. Resolve a real external URI before opening the panel; a
        // localhost fallback inside a tunnel webview points at the wrong side
        // of the connection and renders as a blank tab.
        const remoteUrl = await resolveRemoteViewerUrl(
            url,
            data.serverId || null,
            ensureActive
        );
        ensureActive();
        if (!remoteUrl) {
            log('REMOTE: failed to resolve external URI; leaving signal retry to reopen later');
            writeProtocolAck(data, 'failed', 'Failed to resolve remote viewer URL');
            return;
        }
        openUrl = remoteUrl;
    }
    ensureActive();
    advanceAck('port_resolved');
    const remainingViewerMs = _remainingSignalMs(data);
    const viewerTimeoutMs = remainingViewerMs === null
        ? 25000
        : Math.max(1, remainingViewerMs);

    // Check for a pending placeholder (resolveCustomEditor handoff).
    // If one matches this signal, navigate the existing placeholder tab
    // instead of creating a second panel — eliminates the flicker.
    const desktopTunnel = (
        vscode.env.remoteName === 'tunnel'
        && vscode.env.appHost === 'desktop'
    );
    let useIntegratedBrowser = false;
    if (desktopTunnel) {
        useIntegratedBrowser = await integratedBrowserCommandAvailable();
        if (!useIntegratedBrowser) {
            log('PANEL: integrated browser unavailable; retaining tunnel webview fallback');
        }
    }
    let handedOff = false;
    let viewerReady;
    let integratedBrowserOpened = false;
    for (const [filePath, placeholder] of useIntegratedBrowser ? [] : _pendingPlaceholders) {
        const exactHandoff = data.handoffPath
            && path.resolve(data.handoffPath) === placeholder.filePath;
        const legacyTitleMatch = !data.handoffPath
            && data.title
            && data.title.includes(placeholder.basename);
        if (exactHandoff || legacyTitleMatch) {
            _pendingPlaceholders.delete(filePath);
            try {
                ensureActive();
                viewerReady = waitForViewerReady(placeholder.panel, viewerTimeoutMs);
                const handoffPortMapping = _backendPortMapping(openUrl, data.url);
                _replaceWebviewPortMapping(
                    placeholder.panel.webview,
                    handoffPortMapping
                );
                const handoffWarmupTimeoutMs = Math.max(
                    0,
                    Math.min(12000, viewerTimeoutMs - 1000)
                );
                placeholder.panel.webview.html = _viewerPanelHtml(
                    openUrl,
                    handoffWarmupTimeoutMs > 0
                        ? pingUrlFromViewerUrl(openUrl)
                        : null,
                    handoffWarmupTimeoutMs
                );
                placeholder.panel.__arrayviewUrl = openUrl;
                placeholder.panel.title = data.title || placeholder.title;
                if (panelKey) _openPanels.set(panelKey, placeholder.panel);
                placeholder.panel.onDidDispose(() => {
                    if (
                        !panelKey
                        || _openPanels.get(panelKey) === placeholder.panel
                    ) {
                        if (panelKey) _openPanels.delete(panelKey);
                        releaseUrlSession(openUrl, data.url, data.serverId || null);
                    } else {
                        log(`HANDOFF: ignored disposal from superseded panel ${panelKey}`);
                    }
                });
                log(`HANDOFF: navigated placeholder for ${placeholder.basename} to ${openUrl}`);
                handedOff = true;
            } catch (_) {
                log(`HANDOFF: placeholder panel disposed for ${placeholder.basename}`);
            }
            break;
        }
    }
    if (handedOff) {
        advanceAck('panel_opened');
    } else {
        ensureActive();
        if (useIntegratedBrowser) {
            log(`openInIntegratedBrowser(${openUrl})`);
            const opened = await openInIntegratedBrowser(
                openUrl,
                data.url,
                requestId,
                data.serverId || null,
                data.windowId || logWindowId,
                viewerTimeoutMs,
                ensureActive
            );
            viewerReady = opened.viewerReady;
            integratedBrowserOpened = true;
            log('openInIntegratedBrowser done');
        } else {
            log(`openInWebviewPanel(${openUrl})`);
            viewerReady = openInWebviewPanel(
                openUrl,
                data.title,
                !!data.floating,
                data.url,
                panelKey,
                data.serverId || null,
                viewerTimeoutMs
            );
            log('openInWebviewPanel done');
        }
        advanceAck('panel_opened');
    }

    try {
        const pingUrl = pingUrlFromViewerUrl(openUrl);
        if (!pingUrl) throw new Error('Unable to derive backend ping URL');
        const metadataUrl = sessionMetadataUrlFromViewerUrl(openUrl);
        if (!metadataUrl) throw new Error('Unable to derive viewer session URL');
        for (let attempt = 0; attempt < 10; attempt++) {
            const serverReady = await arrayViewStatusOk(pingUrl, data.serverId || null);
            const sessionReady = serverReady && await httpStatus2xx(metadataUrl);
            if (sessionReady) {
                const viewerError = await viewerReady;
                if (viewerError) throw viewerError;
                ensureActive();
                advanceAck('visibility_verified');
                advanceAck('backend_ready');
                return;
            }
            await new Promise(resolve => setTimeout(resolve, 250));
        }
        throw new Error('Viewer session did not become ready after panel opened');
    } catch (error) {
        if (integratedBrowserOpened) {
            releaseUrlSession(openUrl, data.url, data.serverId || null);
        }
        throw error;
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
    // --- Determine stable window ID ---
    // Priority: 1) IPC hookTag (stable by nature), 2) previously persisted
    // ARRAYVIEW_WINDOW_ID (survives extension host restarts because VS Code
    // persists EnvironmentVariableCollection per-window), 3) current PID (fallback).
    let windowId;
    const envCollection = context.environmentVariableCollection;
    if (OWN_HOOK_TAG) {
        // hookTag is already stable (same IPC socket path → same SHA256 hash)
        windowId = OWN_HOOK_TAG;
    } else {
        // macOS local: reuse the previous window ID stored in the persistent env
        // collection so terminals that already have ARRAYVIEW_WINDOW_ID set
        // continue to target the correct registration after an extension restart.
        let previousId = null;
        try {
            const entry = envCollection.get('ARRAYVIEW_WINDOW_ID');
            if (entry && entry.value) previousId = entry.value;
        } catch (_) {}

        if (previousId && previousId !== String(process.pid)) {
            // Make sure no OTHER currently-alive window already owns this ID.
            const regPath = path.join(SIGNAL_DIR, `window-${previousId}.json`);
            let otherOwns = false;
            try {
                if (fs.existsSync(regPath)) {
                    const regData = JSON.parse(fs.readFileSync(regPath, 'utf8'));
                    if (regData.pid && regData.pid !== process.pid && isProcessAlive(regData.pid)) {
                        otherOwns = true;
                    }
                }
            } catch (_) {}
            windowId = otherOwns ? String(process.pid) : previousId;
            if (!otherOwns) {
                log(`ENV: reusing previous ARRAYVIEW_WINDOW_ID=${windowId} (stable across restart)`);
            } else {
                log(`ENV: previous ID ${previousId} owned by another window, using pid=${windowId}`);
            }
        } else {
            windowId = previousId || String(process.pid);
            log(`ENV: first activation or PID unchanged, using pid=${windowId}`);
        }
    }
    logWindowId = windowId;

    // Update TARGETED_SIGNAL_FILE to match the stable windowId determined above.
    // The module-level initializer used process.pid (available at load time), but
    // windowId may be a previously-persisted ID that differs from process.pid.
    // Python writes to open-request-pid-{windowId}.json, so the watcher must
    // watch the same filename.
    if (!OWN_HOOK_TAG) {
        TARGETED_SIGNAL_FILE = path.join(SIGNAL_DIR, `open-request-pid-${windowId}.json`);
        log(`targetedFile updated to ${path.basename(TARGETED_SIGNAL_FILE)}`);
    }

    try {
        envCollection.replace('ARRAYVIEW_WINDOW_ID', windowId);
        log(`ENV: set ARRAYVIEW_WINDOW_ID=${windowId}`);
    } catch (e) {
        log(`ENV: failed to set ARRAYVIEW_WINDOW_ID: ${e.message}`);
    }
    const regFile = path.join(SIGNAL_DIR, `window-${windowId}.json`);
    try {
        _atomicWriteJson(regFile, {
            hookTag: OWN_HOOK_TAG || '',
            pid: process.pid,
            windowId,
            ppids: EXT_PPIDS,   // ancestor PIDs for multi-window matching by Python
            ts: Date.now(),
            fallbackId: !OWN_HOOK_TAG,  // true if using PID fallback
            remoteName: vscode.env.remoteName || null,
            extensionVersion: version,
            extensionInstanceId: EXTENSION_INSTANCE_ID,
            signalQueueVersion: 1
        });
        log(`REGISTER: wrote ${path.basename(regFile)} (${OWN_HOOK_TAG ? 'hookTag' : 'PID fallback'})`);
        context.subscriptions.push({ dispose: () => {
            const removed = _removeRegistrationIfOwned(regFile, {
                pid: process.pid,
                windowId,
                extensionInstanceId: EXTENSION_INSTANCE_ID,
            });
            log(`REGISTER: dispose ${removed ? 'deleted' : 'preserved replacement'} ${path.basename(regFile)}`);
        }});
    } catch (e) {
        log(`REGISTER: failed to write: ${e.message}`);
    }

    cleanupStaleFiles();

    // Clean up stale registrations from previous tunnel sessions.
    // Do not delete live same-tunnel registrations just because they are older:
    // multiple VS Code windows in one tunnel can share the same first parent,
    // and removing those registrations makes Python target the wrong window.
    if (EXT_PPIDS.length >= 1) {
        try {
            for (const f of fs.readdirSync(SIGNAL_DIR)) {
                if (!f.startsWith('window-') || !f.endsWith('.json')) continue;
                const wid = f.slice(7, -5);
                if (wid === windowId) continue;
                try {
                    const data = JSON.parse(fs.readFileSync(path.join(SIGNAL_DIR, f), 'utf8'));
                    if (shouldRemoveSameTunnelRegistration(
                        windowId,
                        EXT_PPIDS,
                        wid,
                        data,
                        data.pid ? isProcessAlive(data.pid) : false
                    )) {
                        fs.unlinkSync(path.join(SIGNAL_DIR, f));
                        log(`CLEANUP: removed dead same-tunnel registration ${f} (pid=${data.pid})`);
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

    const recoveryInterval = setInterval(() => cleanupStaleFiles(), 5000);
    context.subscriptions.push({ dispose: () => clearInterval(recoveryInterval) });

    try {
        const ownBasename = TARGETED_SIGNAL_FILE ? path.basename(TARGETED_SIGNAL_FILE) : null;
        const watcher = fs.watch(SIGNAL_DIR, (eventType, filename) => {
            if (!filename || filename.includes('.claimed-') || filename.endsWith('.tmp')) return;
            const ownQueuePrefix = ownBasename ? `${ownBasename.slice(0, -5)}.request-` : null;
            const isOwn = ownBasename && (
                filename === ownBasename ||
                (filename.startsWith(ownQueuePrefix) && filename.endsWith('.json'))
            );
            const isFallback = filename === path.basename(SIGNAL_FILE) ||
                               filename.startsWith(`${path.basename(SIGNAL_FILE, '.json')}.request-`) ||
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

    const openFileCmd = vscode.commands.registerCommand('arrayview.openFile', async (uri) => {
        let filePath;
        if (uri && uri.fsPath) {
            filePath = uri.fsPath;
        } else {
            const selected = await vscode.window.showOpenDialog({
                canSelectFiles: true,
                canSelectMany: false,
                filters: {
                    'Array files': ['npy', 'npz', 'nii', 'gz', 'h5', 'hdf5', 'zarr', 'mat', 'tif', 'tiff', 'pt', 'pth'],
                },
            });
            if (!selected || !selected.length) return;
            filePath = selected[0].fsPath;
        }

        try {
            await launchArrayViewFile(filePath, path.basename(filePath));
        } catch (e) {
            log(`COMMAND: openFile failed: ${e.message}`);
            vscode.window.showErrorMessage(`ArrayView: ${e.message}`);
        }
    });
    context.subscriptions.push(openFileCmd);

    const editorProvider = vscode.window.registerCustomEditorProvider(
        ArrayViewEditorProvider.viewType,
        new ArrayViewEditorProvider(),
        {
            webviewOptions: { retainContextWhenHidden: true },
            supportsMultipleEditorsPerDocument: true,
        }
    );
    context.subscriptions.push(editorProvider);

    if (vscode.window.tabGroups && vscode.window.tabGroups.onDidChangeTabs) {
        context.subscriptions.push(vscode.window.tabGroups.onDidChangeTabs(() => {
            keepActiveArrayViewPreview('tab-change');
        }));
        keepActiveArrayViewPreview('activate');
    }

    log('=== ACTIVATE DONE ===');

    // Log available tunnel/port commands for debugging privacy flip issues.
    vscode.commands.getCommands(true).then(cmds => {
        const relevant = cmds.filter(c =>
            c.includes('tunnel') || c.includes('port') ||
            c.includes('forward') || c.includes('privacy') ||
            c.includes('preview')
        );
        log(`AVAILABLE CMD: ${JSON.stringify(relevant)}`);
    }).catch(() => {});
}

function deactivate() {
    log(`deactivate v${version}`);
}

module.exports = {
    activate,
    deactivate,
    __test: {
        _withTimeout,
        _asExternalUriAttempt,
        resolveRemoteViewerUrl,
        claimProtocolRequest,
        writeProtocolAck,
        cleanupStaleFiles,
        _deleteTerminalClaimedFile,
        _requeueOwnedClaim,
        _ownsProtocolClaim,
        _expireProtocolRequest,
        _acquireAckLock,
        _releaseAckLock,
        _removeRegistrationIfOwned,
        _targetedSignalPath,
        _processSignalDataBody,
        _remainingSignalMs,
        tryOpenSignalFile,
        _viewerPanelHtml,
        _publicBaseFromTunnelResult,
        _integratedBrowserLaunchUrl,
        integratedBrowserCommandAvailable,
        waitForBackendViewerReady,
        openInIntegratedBrowser,
        openInWebviewPanel,
        _openPanels,
        extensionInstanceId: EXTENSION_INSTANCE_ID,
        signalDir: SIGNAL_DIR,
        setWindowId(windowId) { logWindowId = windowId; },
        setTargetedSignalFile(filePath) { TARGETED_SIGNAL_FILE = filePath; },
    },
};
