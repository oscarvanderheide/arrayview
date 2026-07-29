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

// How many previous window ids a registration advertises.  Only terminals still
// alive from those incarnations can use them, so a handful covers every real
// case and keeps one long-lived window from accumulating ids indefinitely.
const MAX_SUPERSEDED_WINDOW_IDS = 8;

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
// Identifies which request currently holds the queue. The lock is released as
// soon as a request's panel is up, while its readiness wait continues, so a
// later request can be holding the queue by the time an earlier one finishes.
// Without an owner check that earlier request's `finally` would release a lock
// belonging to someone else, letting two requests into the critical section at
// once.
let signalQueueOwner = null;
// Upper bound on waiting for a viewer's first frame once the backend has
// published its URL. Keeps a failed launch from holding the request queue.
const VIEWER_READY_TIMEOUT_MS = 45000;
// Minimum remaining signal lifetime worth opening a panel for. Real requests
// carry 190–240s, so this only rejects ones already at their deadline.
const PANEL_MIN_REMAINING_MS = 1000;
let logWindowId = '';
let lastHandledRequestId = null;
let lastHandledUrl = null;
let lastHandledAt = 0;
const _activeClaimedFiles = new Set();

// Track open webview panels by stable request identity. The externally resolved
// tunnel URL may change while one request is being recovered.
const _openPanels = new Map(); // request key (or URL for legacy calls) -> panel
const _readyPanels = new WeakSet();
// Browser commands act on VS Code's active integrated-browser editor. Initial
// opens, retries, and reveal+reload recovery therefore need a very small
// command-only critical section. Readiness remains request-scoped and runs
// concurrently after each command returns.
let _integratedBrowserCommandTail = Promise.resolve();

// Hedging, not escalation. Measured against the live devtunnel relay (30 fresh
// connections, `/ping`, backend healthy on loopback throughout): answers arrive
// within 640 ms, or they never arrive at all. There is no slow tail to wait for
// — roughly one connection in five is accepted by the relay and then
// black-holed, returning no response, no reset and no error.
//
// That distribution is what the old escalating ladder (1.5 s / 4 s / 8 s) got
// wrong. Widening the budget cannot help a request that is already lost, so a
// stalled probe cost a flat 13.6 s and taught us nothing. Stalls are largely
// independent per connection, so the effective move is a *second connection*
// while the first is still hanging: in the observed failures a hedge opened
// 1.5 s later answered in ~200 ms. Attempts therefore overlap, and the first
// real verdict wins.
let RELAY_PROBE_HEDGE = { attempts: 3, staggerMs: 200, attemptTimeoutMs: 1000 };

// Backoff for VS Code's own resolver. Kept beside the probe budget above so a
// test can shrink both without waiting out the real schedule.
let EXTERNAL_URI_ATTEMPTS = [
    { timeoutMs: 6000, pauseMs: 0 },
    { timeoutMs: 10000, pauseMs: 500 },
    { timeoutMs: 10000, pauseMs: 1500 },
    { timeoutMs: 10000, pauseMs: 3000 },
    { timeoutMs: 10000, pauseMs: 5000 },
    { timeoutMs: 10000, pauseMs: 8000 },
];

function _setRetryTiming({
    relayProbeHedge,
    externalUriAttempts,
    readinessProbeTimeoutsMs,
} = {}) {
    if (relayProbeHedge) {
        RELAY_PROBE_HEDGE = { ...RELAY_PROBE_HEDGE, ...relayProbeHedge };
    }
    if (externalUriAttempts) EXTERNAL_URI_ATTEMPTS = externalUriAttempts;
    if (readinessProbeTimeoutsMs) {
        READINESS_PROBE_TIMEOUTS_MS = readinessProbeTimeoutsMs;
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

async function _runIntegratedBrowserCommand(operation) {
    const previous = _integratedBrowserCommandTail;
    let release;
    _integratedBrowserCommandTail = new Promise(resolve => { release = resolve; });
    await previous;
    try {
        return await operation();
    } finally {
        release();
    }
}

function _asExternalUriAttempt(baseUri) {
    // A timed-out VS Code resolver cannot be cancelled. Keep attempts
    // request-local and side-effect free so a hung promise cannot poison all
    // future launches for the same port.
    return Promise.resolve().then(() => vscode.env.asExternalUri(baseUri));
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

function _reportExtensionVersionSkew(ownWindowId, ownVersion) {
    // A window keeps running whatever extension build it loaded at activation,
    // so an upgrade leaves older hosts live until each window reloads. Those
    // hosts share this signal directory and race over the same claims, which
    // reads as intermittent failure rather than as a stale build.
    const peers = [];
    try {
        for (const f of fs.readdirSync(SIGNAL_DIR)) {
            if (!f.startsWith('window-') || !f.endsWith('.json')) continue;
            if (f === `window-${ownWindowId}.json`) continue;
            let data;
            try {
                data = JSON.parse(fs.readFileSync(path.join(SIGNAL_DIR, f), 'utf8'));
            } catch (_) { continue; }
            if (!data || !data.extensionVersion || data.extensionVersion === ownVersion) continue;
            if (!data.pid || !isProcessAlive(data.pid)) continue;
            peers.push(`${data.extensionVersion} (window ${data.windowId || '?'}, pid ${data.pid})`);
        }
    } catch (_) { return; }
    if (!peers.length) return;

    log(`SKEW: this window runs v${ownVersion}; live peers on ${peers.join(', ')}`);
    const detail = `This window runs ArrayView opener v${ownVersion}, but other open windows still run ${peers.join(', ')}. Mixed versions share one signal directory and can drop each other's requests. Reload the other windows.`;
    vscode.window.showWarningMessage(detail, 'Reload Other Windows').then(choice => {
        if (choice) vscode.commands.executeCommand('workbench.action.reloadWindow');
    }, () => {});
}

function _arrayviewPackageSpec() {
    try {
        const configured = vscode.workspace.getConfiguration('arrayview').get('packageSpec');
        if (typeof configured === 'string' && configured.trim()) return configured.trim();
    } catch (_) {}
    return 'arrayview';
}

// `uv tool install arrayview` leaves a stable, already-built venv with its
// bytecode compiled. Launching from it skips the work `uv run --with` redoes
// whenever a new arrayview is published: a measured 10.4 s to resolve and
// install 110 packages, then ~6 s more compiling .pyc on first import.
function _uvToolArrayviewBin() {
    const exe = process.platform === 'win32' ? 'arrayview.exe' : 'arrayview';
    const dirs = [];
    if (process.env.UV_TOOL_BIN_DIR) dirs.push(process.env.UV_TOOL_BIN_DIR);
    if (process.env.XDG_BIN_HOME) dirs.push(process.env.XDG_BIN_HOME);
    dirs.push(path.join(os.homedir(), '.local', 'bin'));
    if (process.platform === 'win32' && process.env.APPDATA) {
        dirs.push(path.join(process.env.APPDATA, 'uv', 'tools', 'bin'));
    }
    for (const dir of dirs) {
        const candidate = path.join(dir, exe);
        try {
            if (fs.existsSync(candidate)) return candidate;
        } catch (_) {}
    }
    return null;
}

function _arrayviewPackageArgs() {
    const spec = _arrayviewPackageSpec();
    if (spec !== 'arrayview' && path.isAbsolute(spec)) {
        // A local checkout runs live, so edits apply without a reinstall. The
        // surrounding uv invocation is unchanged, which keeps resolve timing
        // comparable to the released path.
        log(`PYTHON: using local checkout ${spec}`);
        return ['--with-editable', spec];
    }
    return ['--with', spec];
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
    // Skipped when packageSpec points at a checkout: the tool environment holds
    // the released code and would silently shadow the working tree under test.
    if (_arrayviewPackageSpec() === 'arrayview') {
        const toolBin = _uvToolArrayviewBin();
        if (toolBin) candidates.push({ command: toolBin, argsPrefix: [] });
    }
    candidates.push({ command: 'uv', argsPrefix: ['run', '--directory', os.tmpdir(), '--no-project', '--python', '3.12', ..._arrayviewPackageArgs(), 'python', '-m', 'arrayview'] });
    candidates.push({ command: 'python3', argsPrefix: ['-m', 'arrayview'] });
    return candidates;
}

async function _fastLoadViaDaemon(filePath, title) {
    const port = 8000;
    const pingUrl = `http://localhost:${port}/ping`;
    let serverId = null;
    try {
        const pingPayload = await httpJson(pingUrl, 1000);
        if (!pingPayload || pingPayload.service !== 'arrayview' || !pingPayload.instance_id) {
            log(`FASTLOAD: no daemon on port ${port}`);
            return false;
        }
        serverId = pingPayload.instance_id;
    } catch (_) {
        return false;
    }
    const sid = crypto.randomBytes(16).toString('hex');
    const loadPayload = { filepath: filePath, name: title || path.basename(filePath), requested_sid: sid, background: true, release_on_disconnect: true };
    if (serverId) loadPayload.expected_server_id = serverId;
    let loadResult = null;
    try {
        loadResult = await httpPostJson(`http://localhost:${port}/load`, loadPayload, 5000);
    } catch (_) { /* fall through */ }
    if (!loadResult || !loadResult.sid) {
        log(`FASTLOAD: load failed for ${filePath}`);
        return false;
    }
    const resolvedSid = loadResult.sid;
    const url = `http://localhost:${port}/?sid=${encodeURIComponent(resolvedSid)}`;
    const requestId = crypto.randomBytes(16).toString('hex');
    const ackPath = path.join(SIGNAL_DIR, `open-ack-v0100-${requestId}.json`);
    const signalPayload = {
        action: 'open-preview',
        url, title: title || `ArrayView: ${path.basename(filePath)}`,
        maxAgeMs: 240000, protocolVersion: 1, requestId, ackPath,
        requiredExtensionVersion: version,
        remoteOnly: true,
        windowId: logWindowId,
        serverId,
        handoffPath: filePath || undefined,
        sentAtMs: Date.now(),
    };
    const signalWritten = _writeSignalFile(signalPayload, logWindowId);
    if (!signalWritten) return false;

    let ackOk = false;
    let sawClaimed = false;
    const ackDeadline = Date.now() + 12000;
    while (Date.now() < ackDeadline) {
        await new Promise(r => setTimeout(r, 150));
        let ack = null;
        try { ack = JSON.parse(fs.readFileSync(ackPath, 'utf8')); } catch (_) {}
        if (!ack || ack.requestId !== requestId) continue;
        if (ack.state === 'claimed') sawClaimed = true;
        if (ack.state === 'panel_opened' || ack.state === 'backend_ready' || ack.state === 'visibility_verified') {
            log(`FASTLOAD: signal ${ack.state} for ${path.basename(filePath)}`);
            ackOk = true;
            break;
        }
        if (ack.state === 'failed') {
            log(`FASTLOAD: signal failed (${ack.message || 'unknown'}), falling back to Python`);
            break;
        }
    }
    // A claim means a window owns this request and is still working on it —
    // typically waiting on a slow load. Spawning Python anyway would load the
    // same file a second time and queue a duplicate request behind the first.
    if (!ackOk && sawClaimed) {
        log(`FASTLOAD: request still claimed after ${12000}ms, leaving it with the owning window`);
        return true;
    }
    if (!ackOk) {
        try { httpPostJson(`http://localhost:${port}/release/${resolvedSid}`, { server_id: serverId }, 2000); } catch (_) {}
        return false;
    }
    return true;
}

function _writeSignalFile(payload, windowId) {
    try { fs.mkdirSync(SIGNAL_DIR, { recursive: true }); } catch (_) {}
    const requestId = payload.requestId;
    if (!requestId) return false;
    const prefix = `open-request-ipc-${windowId}`;
    const filename = `${prefix}.request-${requestId}.json`;
    const signalFile = path.join(SIGNAL_DIR, filename);
    const tmpFile = signalFile + '.tmp';
    try {
        fs.writeFileSync(tmpFile, JSON.stringify(payload));
        fs.renameSync(tmpFile, signalFile);
        return true;
    } catch (e) {
        log(`FASTLOAD: signal write error: ${e.message}`);
        try { fs.unlinkSync(tmpFile); } catch (_) {}
        return false;
    }
}

function launchArrayViewFile(filePath, title) {
    if (title === undefined) title = path.basename(filePath);
    return (async () => {
        const fast = await _fastLoadViaDaemon(filePath, title);
        if (fast) return;
        await _spawnPythonForFile(filePath, title);
    })();
}

function _spawnPythonForFile(filePath, title) {
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

function _escapeHtml(value) {
    return String(value == null ? '' : value)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

// A request can fail long after launchArrayViewFile resolved — the load runs in
// the backend and the signal is processed asynchronously. Without this the
// placeholder tab keeps saying "Opening …" and the reason is only ever written
// to the log, which is the failure mode R10 exists to prevent.
function _reportFailureToPlaceholder(data, message) {
    const handoff = data && data.handoffPath;
    if (!handoff) return false;
    let key;
    try { key = path.resolve(handoff); } catch (_) { return false; }
    const placeholder = _pendingPlaceholders.get(key);
    if (!placeholder) return false;
    _pendingPlaceholders.delete(key);
    try {
        placeholder.panel.webview.html = `<html><body style="background:#1e1e1e;color:#ccc;padding:2em;font-family:ui-monospace,monospace;line-height:1.5">
            <h2 style="color:#f14c4c;margin-top:0">ArrayView could not open ${_escapeHtml(placeholder.basename)}</h2>
            <pre style="white-space:pre-wrap;color:#e8e8e8;background:#252526;padding:1em;border-radius:4px">${_escapeHtml(message)}</pre>
            <p style="color:#888">Full details: <code>~/.arrayview/extension.log</code></p></body></html>`;
        log(`CUSTOM-EDITOR: reported failure in placeholder for ${placeholder.basename}`);
        return true;
    } catch (_) {
        return false;  // panel already disposed
    }
}

// Registers *webviewPanel* as the tab that the eventual signal-file URL should
// navigate, keyed by the resolved source path. Both entry points need this:
// resolveCustomEditor for a single array file, and the folder command, which
// has no custom editor because VS Code never opens a directory in an editor.
// Without a placeholder the folder command would show nothing at all while the
// backend walks the tree — the exact case that hurts most, since enumerating a
// DICOM folder on a network mount is the slowest thing ArrayView does.
function _registerHandoffPlaceholder(webviewPanel, sourcePath, title, logTag) {
    webviewPanel.webview.options = { enableScripts: true };
    webviewPanel.webview.html = `<html><body style="background:#1e1e1e;color:#ccc;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;font-family:ui-monospace,monospace">
            <div>Opening ${_escapeHtml(title)} in ArrayView...</div></body></html>`;
    const placeholderKey = path.resolve(sourcePath);
    const placeholder = { panel: webviewPanel, basename: title, filePath: placeholderKey };
    _pendingPlaceholders.set(placeholderKey, placeholder);
    webviewPanel.onDidDispose(() => {
        if (_pendingPlaceholders.get(placeholderKey) === placeholder) {
            _pendingPlaceholders.delete(placeholderKey);
        }
        log(`${logTag}: placeholder disposed for ${title}`);
    });
    // Large inputs may legitimately spend minutes loading before the URL is
    // ready. Keep the placeholder correlated for the whole launch budget.
    const timer = setTimeout(() => {
        if (_pendingPlaceholders.get(placeholderKey) === placeholder) {
            _pendingPlaceholders.delete(placeholderKey);
            try {
                webviewPanel.webview.html = `<html><body style="color:#c00;padding:2em;font-family:monospace;background:#1e1e1e">
                        <h2>ArrayView failed to start</h2>
                        <p>The Python server did not respond. Check ~/.arrayview/extension.log for details.</p></body></html>`;
            } catch (_) { /* panel already disposed */ }
        }
    }, 190000);
    return {
        placeholderKey,
        placeholder,
        forget() {
            clearTimeout(timer);
            if (_pendingPlaceholders.get(placeholderKey) === placeholder) {
                _pendingPlaceholders.delete(placeholderKey);
            }
        },
        reportError(error) {
            this.forget();
            log(`${logTag}: error: ${error.message}\n${error.stack || ''}`);
            try {
                webviewPanel.webview.html = `<html><body style="color:#c00;padding:2em;font-family:monospace;background:#1e1e1e">
                <h2>ArrayView failed to open</h2><pre>${_escapeHtml(error.message)}</pre>
                <p>Check ~/.arrayview/extension.log for details.</p></body></html>`;
            } catch (_) { /* panel already disposed */ }
        },
    };
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
        const handoff = _registerHandoffPlaceholder(
            webviewPanel, filePath, title, 'CUSTOM-EDITOR'
        );
        try {
            await launchArrayViewFile(filePath, title);
            log(`CUSTOM-EDITOR: launched network viewer for ${filePath}`);
        } catch (e) {
            handoff.reportError(e);
        }
    }
}

// Opens a directory the same way the custom editor opens a file. VS Code has no
// custom editor for folders — `explorer/context` + `explorerResourceIsFolder` is
// the only way to reach a folder from the Explorer — so this command creates the
// placeholder tab itself and then uses the identical launch path. The Python
// side decides what the folder means (DICOM series, per-case folders, or a flat
// collection); the extension deliberately does not guess and does not pass
// --stack, because --stack is wrong for a DICOM folder.
async function openFolderInArrayView(folderPath) {
    const title = path.basename(folderPath.replace(/[\\/]+$/, '')) || folderPath;
    log(`FOLDER: opening ${folderPath}`);
    const panel = vscode.window.createWebviewPanel(
        'arrayview.folderPlaceholder',
        title,
        vscode.ViewColumn.Active,
        { enableScripts: true, retainContextWhenHidden: true }
    );
    const handoff = _registerHandoffPlaceholder(panel, folderPath, title, 'FOLDER');
    try {
        await launchArrayViewFile(folderPath, title);
        log(`FOLDER: launched network viewer for ${folderPath}`);
    } catch (e) {
        handoff.reportError(e);
        throw e;
    }
}

// Probe outcomes. A route that answers wrongly proves it is not our backend; a
// route that does not answer in time proves nothing at all. Collapsing those two
// into one boolean makes a slow network indistinguishable from a dead tunnel,
// which is how a healthy devtunnel route gets discarded mid-session.
const PROBE_OK = 'ok';
const PROBE_DEAD = 'dead';
const PROBE_UNKNOWN = 'unknown';
// The relay answered about itself: reachable, but not currently carrying our
// port. Deliberately distinct from PROBE_UNKNOWN — a stall is an absence of
// information, whereas this is information, and the two want opposite
// responses. A stall means "use the route, nothing says it is wrong"; a
// detached connector means "the route cannot carry traffic right now", and
// re-forwarding the port is the thing that fixes it.
const PROBE_RELAY_DOWN = 'relay-down';

// Socket failures that say nothing about whether the backend is still there.
// The relay behind a devtunnel URL stalls and resets under load, so these are
// evidence about the network, not about the route.
const TRANSIENT_PROBE_ERRORS = new Set([
    'ETIMEDOUT',
    'ECONNRESET',
    'EPIPE',
    'EAI_AGAIN',
    'ENETUNREACH',
    'EHOSTUNREACH',
]);

// Statuses a relay generates about *itself*. A devtunnel answers 502 within a
// few hundred ms whenever its local connector is not attached yet — the relay
// is reachable, the backend simply has not been reached through it. Read as a
// verdict on the backend this is a lie, and a fast one: it fails a launch
// sooner than a stall would. Only meaningful off loopback; a non-200 from
// localhost really is our own backend answering wrongly.
const RELAY_STATUS_CODES = new Set([502, 503, 504, 521, 522, 523, 524]);

function probeArrayViewStatus(url, expectedServerId = null, timeoutMs = 1500) {
    return new Promise((resolve) => {
        let parsed;
        try {
            parsed = new URL(url);
        } catch (_) {
            resolve(PROBE_DEAD);
            return;
        }
        const lib = parsed.protocol === 'https:' ? https : http;
        let settled = false;
        const done = (outcome) => {
            if (settled) return;
            settled = true;
            resolve(outcome);
        };
        const viaRelay = !isLoopbackUrl(url);
        // `agent: false` gives every probe its own connection. Node 22's global
        // agent keeps sockets alive, and hedged attempts are only independent
        // if they are separate connections — pooling would let a second attempt
        // inherit the same wedged socket the first one is stuck on.
        const req = lib.get(parsed, { timeout: timeoutMs, agent: false }, (res) => {
            if (res.statusCode !== 200) {
                res.resume();
                done(viaRelay && RELAY_STATUS_CODES.has(res.statusCode)
                    ? PROBE_RELAY_DOWN
                    : PROBE_DEAD);
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
                    done(isArrayViewStatus(payload, expectedServerId)
                        ? PROBE_OK
                        : PROBE_DEAD);
                } catch (_) {
                    done(PROBE_DEAD);
                }
            });
            // A reset partway through the body leaves the verdict unknown.
            res.on('error', () => done(PROBE_UNKNOWN));
        });
        req.on('timeout', () => {
            req.destroy();
            done(PROBE_UNKNOWN);
        });
        req.on('error', (error) => done(
            TRANSIENT_PROBE_ERRORS.has(error && error.code)
                ? PROBE_UNKNOWN
                : PROBE_DEAD
        ));
    });
}

/**
 * Probe a relay URL with overlapping attempts and take the first real verdict.
 *
 * See RELAY_PROBE_HEDGE for the measurement this is built on. A stalled attempt
 * is not waited out: the next connection opens `staggerMs` later while the
 * previous one is still hanging, so a black-holed request costs the stagger
 * rather than its whole budget. PROBE_UNKNOWN is returned only when every
 * attempt has stalled, which needs all of them to be lost independently.
 */
function hedgedProbeStatus(url, expectedServerId = null, hedge = null) {
    const { attempts, staggerMs, attemptTimeoutMs } = hedge || RELAY_PROBE_HEDGE;
    return new Promise((resolve) => {
        let settled = false;
        let started = 0;
        let outstanding = 0;
        let timer = null;
        const finish = (outcome) => {
            if (settled) return;
            settled = true;
            if (timer) clearTimeout(timer);
            resolve(outcome);
        };
        const launch = () => {
            timer = null;
            started += 1;
            outstanding += 1;
            const attempt = started;
            probeArrayViewStatus(url, expectedServerId, attemptTimeoutMs)
                .then((outcome) => {
                    outstanding -= 1;
                    if (outcome !== PROBE_UNKNOWN) {
                        finish(outcome);
                        return;
                    }
                    log(`PROBE: attempt ${attempt} gave no verdict `
                        + `(budget=${attemptTimeoutMs}ms url=${url})`);
                    // Nothing left running and nothing left to start: every
                    // connection was lost, which is still not proof of death.
                    if (started >= attempts && outstanding === 0) {
                        finish(PROBE_UNKNOWN);
                    }
                });
            if (started < attempts) {
                timer = setTimeout(() => { if (!settled) launch(); }, staggerMs);
            }
        };
        launch();
    });
}

async function arrayViewStatusOk(url, expectedServerId = null, timeoutMs = 1500) {
    const outcome = await probeArrayViewStatus(url, expectedServerId, timeoutMs);
    return outcome === PROBE_OK;
}

// Loopback only. A loopback port has no black-hole mode — it answers, it
// refuses, or the backend has not bound it yet — so waiting longer is the
// right response there and sequential retry stays. Relay readiness goes
// through RELAY_PROBE_HEDGE instead, for the reason recorded above it.
let READINESS_PROBE_TIMEOUTS_MS = [1500, 2500];

/**
 * Probe until the answer means something. Retrying a stall is the whole point:
 * PROBE_UNKNOWN is a fact about the path, and only PROBE_DEAD — a well-formed
 * reply from a different backend, or a refused loopback port — is evidence
 * about the backend itself.
 */
async function probeUntilVerdict(
    url, expectedServerId, timeoutsMs, ensureActive = () => {}
) {
    let outcome = PROBE_UNKNOWN;
    for (let i = 0; i < timeoutsMs.length; i++) {
        ensureActive();
        outcome = await probeArrayViewStatus(url, expectedServerId, timeoutsMs[i]);
        if (outcome !== PROBE_UNKNOWN) return outcome;
        log(`READY: ping gave no verdict (attempt=${i + 1} `
            + `timeout=${timeoutsMs[i]}ms url=${url})`);
    }
    return outcome;
}

// Strict loopback identity verdict, deliberately narrower than
// probeArrayViewStatus. Abandoning a request is irreversible from the user's
// side, so it may only happen on positive proof that the port is owned by a
// *different* ArrayView backend: an HTTP 200 whose payload is a well-formed
// ArrayView status carrying someone else's instance_id. A refused connection,
// a timeout, a non-200 or an unparseable body all mean "not up yet" — most
// often a large array still loading before it binds the port — and must stay
// indistinguishable from success here. Note ECONNREFUSED is not in
// TRANSIENT_PROBE_ERRORS, so probeArrayViewStatus reports a still-starting
// backend as PROBE_DEAD; that verdict is safe for cache invalidation but must
// never be used to abandon a request.
const LOCAL_MINE = 'local-mine';
const LOCAL_FOREIGN = 'local-foreign';
const LOCAL_UNKNOWN = 'local-unknown';

function localBackendIdentity(port, expectedServerId, timeoutMs = 1500) {
    if (!expectedServerId) return Promise.resolve(LOCAL_UNKNOWN);
    return new Promise((resolve) => {
        let settled = false;
        const done = (verdict) => {
            if (settled) return;
            settled = true;
            resolve(verdict);
        };
        const req = http.get(
            `http://localhost:${port}/ping`,
            { timeout: timeoutMs },
            (res) => {
                if (res.statusCode !== 200) {
                    res.resume();
                    done(LOCAL_UNKNOWN);
                    return;
                }
                let body = '';
                res.setEncoding('utf8');
                res.on('data', (chunk) => {
                    if (body.length < 65536) body += chunk;
                });
                res.on('end', () => {
                    let payload = null;
                    try { payload = JSON.parse(body); } catch (_) {}
                    if (!payload || payload.service !== 'arrayview'
                        || !payload.instance_id) {
                        done(LOCAL_UNKNOWN);
                        return;
                    }
                    done(payload.instance_id === expectedServerId
                        ? LOCAL_MINE
                        : LOCAL_FOREIGN);
                });
                res.on('error', () => done(LOCAL_UNKNOWN));
            }
        );
        req.on('timeout', () => { req.destroy(); done(LOCAL_UNKNOWN); });
        req.on('error', () => done(LOCAL_UNKNOWN));
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

// 422 from /metadata/<sid> means the backend tried to load the file and failed.
// That is terminal: polling it again cannot change the answer, and the body
// carries the reason worth showing the user.
function httpSessionProbe(url, timeoutMs = 3000) {
    return new Promise((resolve) => {
        let parsed;
        try {
            parsed = new URL(url);
        } catch (_) {
            resolve({ ready: false, loadError: null });
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
            const status = res.statusCode || 0;
            if (status === 422) {
                let body = '';
                res.setEncoding('utf8');
                res.on('data', chunk => { if (body.length < 2000) body += chunk; });
                res.on('end', () => done({
                    ready: false,
                    loadError: body.trim() || 'the backend could not load this file',
                }));
                return;
            }
            res.resume();
            done({ ready: status >= 200 && status < 300, loadError: null });
        });
        req.on('timeout', () => { req.destroy(); done({ ready: false, loadError: null }); });
        req.on('error', () => done({ ready: false, loadError: null }));
    });
}

async function waitForSessionReady(url, timeoutMs = 150000, pollMs = 500) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        const probe = await httpSessionProbe(url);
        if (probe.ready) return { ready: true, loadError: null };
        if (probe.loadError) return { ready: false, loadError: probe.loadError };
        const remaining = deadline - Date.now();
        if (remaining > 0) {
            await new Promise(resolve => setTimeout(resolve, Math.min(pollMs, remaining)));
        }
    }
    return { ready: false, loadError: null };
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
function _viewerPanelHtml(url, hedgeNavigation = false) {
    const nonce = crypto.randomBytes(16).toString('hex');
    const jsonUrl = JSON.stringify(url);
    const jsonHedgeNavigation = JSON.stringify(Boolean(hedgeNavigation));
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
<div id="frames"></div>
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
const hedgeNavigation = ${jsonHedgeNavigation};
const framesRoot = document.getElementById('frames');
vscodeApi.postMessage({ type: 'panel-phase', phase: 'wrapper-started' });
let viewerReady = false;
let viewerLoaded = false;
let reloadTimer = null;
let wave = 0;
let candidateSerial = 0;
let winner = null;
let candidates = [];
let hedgeTimers = [];
const MAX_WAVES = 3;
const NAVIGATE_TIMEOUT_MS = hedgeNavigation ? 10000 : 8000;
const HEDGE_DELAYS_MS = hedgeNavigation ? [0, 700, 1400] : [0];
function cancelTimer(id) {
    if (id) clearTimeout(id);
}
function clearNavigationTimers() {
    hedgeTimers.forEach(cancelTimer);
    hedgeTimers = [];
    cancelTimer(reloadTimer);
    reloadTimer = null;
}
function retire(candidate) {
    if (!candidate || candidate === winner) return;
    try { candidate.frame.remove(); } catch (_) {}
}
function candidateForSource(source) {
    return candidates.find(candidate => candidate.frame.contentWindow === source) || null;
}
function showBackendError() {
    if (viewerReady) return;
    clearNavigationTimers();
    document.getElementById('backend-url').textContent = arrayviewUrl;
    document.getElementById('backend-error').classList.add('visible');
    candidates.forEach(retire);
}
function chooseWinner(candidate) {
    if (winner || !candidate) return winner === candidate;
    winner = candidate;
    viewerLoaded = true;
    clearNavigationTimers();
    candidate.frame.style.visibility = 'visible';
    candidates.forEach(retire);
    vscodeApi.postMessage({
        type: 'panel-phase',
        phase: 'navigation-winner',
        attempt: candidate.serial,
        wave,
    });
    return true;
}
function candidateUrl(serial) {
    if (!hedgeNavigation && serial === 1) return arrayviewUrl;
    try {
        const parsed = new URL(arrayviewUrl);
        if (hedgeNavigation) {
            parsed.searchParams.set('_av_relay_hedge', '1');
        }
        if (serial > 1) {
            parsed.searchParams.set('_av_nav_hedge', String(serial));
        }
        return parsed.toString();
    } catch (_) {
        const separator = arrayviewUrl.includes('?') ? '&' : '?';
        return arrayviewUrl + separator + '_av_nav_hedge=' + serial;
    }
}
function retryNavigation(phase) {
    if (winner || viewerReady) return;
    clearNavigationTimers();
    if (wave >= MAX_WAVES) {
        showBackendError();
        return;
    }
    vscodeApi.postMessage({ type: 'panel-phase', phase, wave });
    startWave();
}
function launchCandidate() {
    if (winner || viewerReady) return;
    const serial = ++candidateSerial;
    const frame = document.createElement('iframe');
    const candidate = { frame, serial };
    candidates.push(candidate);
    frame.allow = 'clipboard-read; clipboard-write; fullscreen';
    frame.style.visibility = serial === 1 ? 'visible' : 'hidden';
    frame.addEventListener('load', () => {
        vscodeApi.postMessage({
            type: 'panel-phase',
            phase: 'iframe-loaded',
            attempt: serial,
            wave,
        });
    });
    frame.addEventListener('error', () => {
        vscodeApi.postMessage({
            type: 'panel-phase',
            phase: 'iframe-error',
            attempt: serial,
            wave,
        });
    });
    // Set src while detached. Appending a src-less iframe first creates an
    // initial about:blank document whose load event can be mistaken for the
    // relay navigation completing.
    frame.src = candidateUrl(serial);
    vscodeApi.postMessage({
        type: 'panel-phase',
        phase: 'navigation-attempt',
        attempt: serial,
        wave,
    });
    framesRoot.appendChild(frame);
}
function startWave() {
    if (winner || viewerReady) return;
    wave++;
    candidates.forEach(retire);
    candidates = [];
    for (const delay of HEDGE_DELAYS_MS) {
        if (delay === 0) {
            launchCandidate();
        } else {
            hedgeTimers.push(setTimeout(launchCandidate, delay));
        }
    }
    reloadTimer = setTimeout(() => {
        reloadTimer = null;
        if (winner || viewerReady) return;
        retryNavigation('navigation-wave-timeout');
    }, NAVIGATE_TIMEOUT_MS);
}
window.addEventListener('message', (event) => {
    const msg = event && event.data;
    if (msg && msg.type === 'backend-error') {
        console.log('[arrayview-opener] viewer reported backend-error');
        showBackendError();
        return;
    }
    if (!msg || msg.source !== 'arrayview-viewer') return;
    const candidate = candidateForSource(event.source);
    if (!candidate) return;
    if (!winner && msg.phase === 'script-loaded') {
        chooseWinner(candidate);
    }
    if (winner !== candidate) return;
    vscodeApi.postMessage({ type: 'viewer-phase', phase: msg.phase || 'unknown' });
    if (msg.phase === 'script-loaded') {
        viewerLoaded = true;
        console.log('[arrayview-opener] viewer script loaded; waiting for first frame');
        return;
    }
    if (msg.phase === 'render-error') {
        // The backend cannot draw this array, so no reload will help and the
        // frame this panel is waiting for will never arrive. Retrying would
        // just repeat the failure until the request times out, holding the
        // signal queue and stalling every later click.
        viewerLoaded = true;
        console.log('[arrayview-opener] viewer reported render-error');
        vscodeApi.postMessage({
            type: 'viewer-failed',
            phase: 'render-error',
            message: msg.detail || 'This array could not be rendered',
        });
        return;
    }
    if (msg.phase === 'frame-rendered') {
        if (!viewerReady) {
            viewerLoaded = true;
            viewerReady = true;
            clearNavigationTimers();
            console.log('[arrayview-opener] viewer phase ' + msg.phase);
            vscodeApi.postMessage({ type: 'viewer-ready', phase: msg.phase });
        }
    }
});
startWave();
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
            // A verdict of "this cannot be drawn" is as terminal as a rendered
            // frame. Waiting out the full timeout instead would report a hang
            // and keep the queue locked while the answer is already known.
            if (message?.type === 'viewer-failed') {
                log(`PANEL: viewer failed — ${message.message}`);
                finish(new Error(message.message || 'The viewer could not render this array'));
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
    tabKey,
    navigationKey
) {
    try {
        const parsed = new URL(url);
        parsed.pathname = `/_av/${encodeURIComponent(tabKey)}/${encodeURIComponent(navigationKey)}`;
        parsed.search = '';
        parsed.hash = '';
        return parsed.toString();
    } catch (_) {
        return null;
    }
}

function _correlatedBrowserLaunchUrl(
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
        parsed.hash = `#av-${requestId}`;
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
    // Treated as an inactivity budget rather than a total one: every newly
    // observed phase extends it. The panel now opens before the array has
    // finished loading, so a legitimately slow launch would otherwise be
    // killed for being slow rather than for being stuck.
    let deadline = Date.now() + timeoutMs;
    const preScriptDeadline = Date.now() + Math.min(
        timeoutMs,
        Math.max(1, preScriptTimeoutMs)
    );
    let activeToken = token;
    let scriptLoaded = false;
    let unreachableCount = 0;
    const maxUnreachableAfterScript = 15;
    let navigationAttempt = 0;
    // Every recovery step is a full fresh navigation.  The relay drops a page
    // request outright rather than delivering it slowly, so the only useful
    // response is another independent fetch.  Escalating to a hard reload of
    // the already-blank tab recovered 0 of 5 observed stalls; a fresh
    // navigation recovered 2 of 7.  The budget buys repeats of the latter.
    const navigationRetryDelayMs = Math.min(
        1500,
        Math.max(50, Math.floor(preScriptTimeoutMs * 0.15))
    );
    let nextNavigationRetryAt = Date.now() + navigationRetryDelayMs;
    const maxNavigationRetries = 4;
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
                    deadline = Date.now() + timeoutMs;  // progress, not stuck
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
            unreachableCount = 0;
        } else if (scriptLoaded) {
            unreachableCount += 1;
            if (unreachableCount >= maxUnreachableAfterScript) {
                return new Error(
                    'Backend became unreachable after viewer script loaded'
                );
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
                    Math.min(deadline, preScriptDeadline)
                );
            } catch (error) {
                log(`PANEL: pre-script navigation retry failed: ${error.message || error}`);
            }
            ensureActive();
            if (replacementToken) activeToken = replacementToken;
            nextNavigationRetryAt = Date.now() + navigationRetryDelayMs;
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
    const tabKey = crypto.randomBytes(12).toString('base64url');
    const reuseUrlFilter = `/_av/${tabKey}/`;
    const prepareNavigation = async (navigationAttempt = 0, deadline = null) => {
        ensureActive();
        const token = crypto.randomBytes(16).toString('hex');
        const navigationKey = crypto.randomBytes(12).toString('base64url');
        const viewerQuery = new URL(backendUrl).search;
        // A hedge must fit inside the budget it is hedging.  `deadline` is the
        // caller's pre-script deadline; without clamping to it, every attempt
        // could spend a full POST timeout plus a full command timeout *past*
        // that budget, so raising the retry count multiplied the overshoot
        // instead of packing more attempts into it.
        const attemptDeadline = deadline !== null
            ? Math.min(deadline, viewerDeadline)
            : viewerDeadline;
        const prepared = await httpPostJson(
            journalUrl,
            {
                phase: 'launch-prepared',
                server_id: serverId,
                window_id: windowId,
                token,
                viewer_query: viewerQuery,
                tab_key: tabKey,
                navigation_key: navigationKey,
                navigation_attempt: navigationAttempt,
            },
            Math.max(1, Math.min(1500, attemptDeadline - Date.now()))
        );
        if (
            !prepared
            || prepared.request_id !== requestId
            || prepared.server_id !== serverId
            || prepared.window_id !== windowId
            || prepared.token !== token
            || prepared.tab_key !== tabKey
            || prepared.navigation_key !== navigationKey
            || prepared.navigation_attempt !== navigationAttempt
        ) {
            throw new Error('Unable to prepare correlated viewer readiness journal');
        }
        const launchUrl = _integratedBrowserLaunchUrl(
            browserUrl,
            tabKey,
            navigationKey
        );
        if (!launchUrl) throw new Error('Unable to build integrated browser launch URL');
        const commandPromise = _runIntegratedBrowserCommand(
            () => _withTimeout(
                vscode.commands.executeCommand('workbench.action.browser.open', {
                    url: launchUrl,
                    // Each invocation gets a new browser tab in the preferred group.
                    // openToSide=true creates and locks a new editor group per launch,
                    // eventually leaving VS Code unable to load another browser page.
                    openToSide: false,
                    // Retry/replay of this exact request reuses its one tab, while every
                    // distinct ArrayView invocation opens a fresh browser tab.
                    reuseUrlFilter,
                }),
                3000,
                'integrated browser open'
            )
        );
        if (navigationAttempt > 0 && deadline !== null) {
            try {
                await _withTimeout(
                    commandPromise,
                    Math.max(1, Math.min(3000, attemptDeadline - Date.now())),
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
                if (deadline - Date.now() <= 0) return null;
                log(`PANEL: retrying pre-script navigation attempt=${navigationAttempt}`);
                return prepareNavigation(navigationAttempt, deadline);
            },
            preScriptTimeoutMs
        ),
    };
}

async function openInExternalBrowser(
    backendUrl,
    requestId,
    serverId,
    windowId,
    viewerTimeoutMs,
    ensureActive = () => {}
) {
    const viewerDeadline = Date.now() + viewerTimeoutMs;
    const sid = collectReleaseSidsFromUrl(backendUrl)[0] || null;
    if (!sid || !requestId || !serverId || !windowId) {
        throw new Error('External browser launch is missing correlated viewer identity');
    }
    const pingUrl = pingUrlFromViewerUrl(backendUrl);
    const status = pingUrl ? await httpJson(pingUrl, 750) : null;
    if (
        !status
        || status.service !== 'arrayview'
        || status.instance_id !== serverId
    ) {
        throw new Error('Unable to verify the backend before opening the external browser');
    }

    const token = crypto.randomBytes(16).toString('hex');
    const launchUrl = _correlatedBrowserLaunchUrl(
        backendUrl,
        requestId,
        serverId,
        windowId,
        token
    );
    if (!launchUrl) throw new Error('Unable to build external browser launch URL');
    const journalUrl = `${new URL(backendUrl).origin}/viewer-phase/${encodeURIComponent(sid)}/${encodeURIComponent(requestId)}`;
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
        throw new Error('Unable to prepare correlated external-browser readiness journal');
    }

    ensureActive();
    // openExternal is the supported remote-extension path for a top-level
    // system browser. VS Code forwards remote localhost privately and the
    // top-level browser can complete tunnel authentication; no public-port
    // promotion or embedded iframe is involved.
    let opened;
    try {
        opened = await vscode.env.openExternal(vscode.Uri.parse(launchUrl));
    } catch (error) {
        releaseUrlSession(backendUrl, backendUrl, serverId);
        throw error;
    }
    if (opened === false) {
        releaseUrlSession(backendUrl, backendUrl, serverId);
        throw new Error('VS Code declined to open the external browser');
    }
    log(`PANEL: external browser opened ${backendUrl}`);
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
            null
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

async function reserveDirectIntegratedBrowser(backendUrl, expectedServerId = null) {
    const remoteProxyEnabled = vscode.workspace
        .getConfiguration('workbench.browser')
        .get('enableRemoteProxy', false);
    if (!remoteProxyEnabled) {
        log('PANEL: integrated browser direct proxy disabled');
        return false;
    }
    if (
        typeof expectedServerId !== 'string'
        || expectedServerId.length === 0
    ) {
        log('PANEL: direct-browser request has no backend identity');
        return false;
    }
    const pingUrl = pingUrlFromViewerUrl(backendUrl);
    const status = pingUrl ? await httpJson(pingUrl, 750) : null;
    if (
        !status
        || status.service !== 'arrayview'
        || status.instance_id !== expectedServerId
    ) {
        log('PANEL: unable to verify direct-browser ownership');
        return false;
    }
    if (!await integratedBrowserCommandAvailable()) {
        log('PANEL: integrated browser unavailable');
        return false;
    }
    return true;
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
    const hedgeNavigation = vscode.env.remoteName === 'tunnel'
        && !isLoopbackUrl(url);

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
                existing.webview.html = _viewerPanelHtml(url, hedgeNavigation);
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
    panel.webview.html = _viewerPanelHtml(url, hedgeNavigation);
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

async function resolveRemoteViewerUrl(
    url,
    expectedServerId = null,
    ensureActive = () => {}
) {
    ensureActive();
    if (vscode.env.remoteName === 'tunnel') {
        log('REMOTE: public tunnel URL resolution is disabled');
        return null;
    }
    let parsed;
    try {
        parsed = new URL(url);
    } catch (_) {
        return null;
    }
    const port = parseInt(parsed.port, 10) || 8000;
    if (await localBackendIdentity(port, expectedServerId) === LOCAL_FOREIGN) {
        log(`REMOTE: localhost:${port} is owned by a different backend`);
        return null;
    }
    const baseUri = vscode.Uri.parse(`${parsed.protocol}//${parsed.hostname}:${port}/`);
    for (let i = 0; i < EXTERNAL_URI_ATTEMPTS.length; i++) {
        ensureActive();
        const attempt = EXTERNAL_URI_ATTEMPTS[i];
        if (attempt.pauseMs) {
            await new Promise(resolve => setTimeout(resolve, attempt.pauseMs));
        }
        try {
            const externalUri = await _withTimeout(
                _asExternalUriAttempt(baseUri),
                attempt.timeoutMs,
                'asExternalUri'
            );
            ensureActive();
            const externalBase = externalUri.toString().replace(/\/$/, '');
            const finalUrl = externalBase + '/' + parsed.search;
            log(`REMOTE: final Remote SSH URL = ${finalUrl}`);
            return finalUrl;
        } catch (error) {
            log(`REMOTE: asExternalUri attempt ${i + 1} failed: ${error.message}`);
        }
    }
    return null;
}

async function processSignalData(data) {
    const queueTicket = Symbol('signal-queue');
    isProcessingSignal = true;
    signalQueueOwner = queueTicket;
    log(`LOCK: isProcessingSignal=true`);
    // The queue exists to serialise the parts that touch shared state:
    // resolving the route, claiming a pending placeholder, creating a panel.
    // Waiting for the viewer's first frame touches none of that — it is a
    // network wait that can legitimately run for tens of seconds while a large
    // array loads. Holding the queue across it made every later click wait for
    // an unrelated file, which is what "one bad array broke the next few"
    // actually was. Released once the panel is up; see the call site.
    const releaseQueue = (reason) => {
        if (signalQueueOwner !== queueTicket) return;
        signalQueueOwner = null;
        isProcessingSignal = false;
        log(`UNLOCK: isProcessingSignal=false (${reason})`);
    };
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
            _processSignalDataBody(data, operation, releaseQueue),
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
        const shown = _reportFailureToPlaceholder(data, error.message);
        if (!shown) {
            // No placeholder tab was waiting (terminal launch, or the user
            // closed it). Say it out loud rather than only in the log.
            try {
                vscode.window.showErrorMessage(`ArrayView: ${error.message}`);
            } catch (_) {}
        }
    } finally {
        if (hardTimer) clearTimeout(hardTimer);
        // Normally already released at panel_opened; this covers the paths that
        // fail before a panel ever opens. No-op if a later request now owns it.
        releaseQueue('request settled');
        // Signal files for subsequent arrays remain on disk; the 1-second poll
        // will pick them up now that isProcessingSignal is false again.
    }
}

async function _processSignalDataBody(
    data,
    operation = { cancelled: false },
    releaseQueue = () => {}
) {
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

    // The backend serves its port before a large array has finished loading, so
    // waiting for the session here left the tab blank for the whole load. The
    // panel is opened first and the viewer shows its own loading state while
    // /metadata/<sid> is still pending; readiness is awaited after the panel
    // exists. See the post-panel wait below.
    const remainingMs = _remainingSignalMs(data);
    const metadataWaitMs = remainingMs === null
        ? 150000
        : Math.max(1, Math.min(150000, remainingMs));
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
    const desktopTunnel = (
        vscode.env.remoteName === 'tunnel'
        && vscode.env.appHost === 'desktop'
    );
    const displaySurface = data.displaySurface || (
        vscode.env.remoteName === 'tunnel' ? 'integrated-browser' : null
    );
    let useIntegratedBrowser = false;
    const useExternalBrowser = displaySurface === 'external-browser';
    if (desktopTunnel && !useExternalBrowser) {
        useIntegratedBrowser = await reserveDirectIntegratedBrowser(
            url,
            data.serverId || null
        );
        if (useIntegratedBrowser) {
            log('REMOTE: desktop integrated-browser proxy uses backend URL directly');
        } else {
            const reason = 'Private integrated-browser routing is unavailable; '
                + 'enable workbench.browser.enableRemoteProxy and retry';
            log(`REMOTE: ${reason}`);
            writeProtocolAck(data, 'failed', reason);
            if (!_reportFailureToPlaceholder(data, reason)) {
                try { vscode.window.showErrorMessage(`ArrayView: ${reason}`); } catch (_) {}
            }
            return;
        }
    }
    if (
        vscode.env.remoteName === 'tunnel'
        && !desktopTunnel
        && !useExternalBrowser
    ) {
        const reason = 'Private integrated-browser routing is unavailable in browser-hosted VS Code; '
            + 'retry with --window browser';
        log(`REMOTE: ${reason}`);
        writeProtocolAck(data, 'failed', reason);
        if (!_reportFailureToPlaceholder(data, reason)) {
            try { vscode.window.showErrorMessage(`ArrayView: ${reason}`); } catch (_) {}
        }
        return;
    }
    if (vscode.env.remoteName && !useIntegratedBrowser && !useExternalBrowser) {
        // Preserve the existing Remote SSH webview route. Tunnel requests have
        // already selected a private browser surface or failed closed above,
        // so they can never reach the public-port resolver from signal
        // processing.
        const remoteUrl = await resolveRemoteViewerUrl(
            url,
            data.serverId || null,
            ensureActive
        );
        ensureActive();
        if (!remoteUrl) {
            log('REMOTE: failed to resolve external URI; leaving signal retry to reopen later');
            const reason = 'Failed to resolve remote viewer URL';
            writeProtocolAck(data, 'failed', reason);
            if (!_reportFailureToPlaceholder(data, reason)) {
                try { vscode.window.showErrorMessage(`ArrayView: ${reason}`); } catch (_) {}
            }
            return;
        }
        openUrl = remoteUrl;
    }
    ensureActive();
    advanceAck('port_resolved');
    const remainingViewerMs = _remainingSignalMs(data);
    // Bound this independently of the signal budget. Everything slow — reading
    // the array off its storage tier — already happened before the backend
    // published this URL, so first frame is a matter of seconds. Spending the
    // whole remaining signal lifetime here means one doomed launch holds the
    // queue lock for minutes and starves every request behind it.
    const viewerTimeoutMs = remainingViewerMs === null
        ? 25000
        : Math.max(1, Math.min(VIEWER_READY_TIMEOUT_MS, remainingViewerMs));

    // Opening a panel is a visible side effect, so refuse one for a request
    // with no usable life left — it would be abandoned before it could render.
    // The pre-panel session wait used to consume the remaining lifetime and
    // enforce this implicitly; with the panel now opening first, it has to be
    // an explicit check.
    if (
        remainingViewerMs !== null
        && remainingViewerMs < PANEL_MIN_REMAINING_MS
    ) {
        operation.cancelled = true;
        // Fence it properly if the deadline has actually passed; otherwise
        // record the terminal state directly, since this is a decision not to
        // display rather than an expiry the recovery path would notice.
        if (!_expireProtocolRequest(data, _ackForProtocolRequest(data))) {
            writeProtocolAck(
                data, 'failed', 'Signal expired before a panel could be opened'
            );
        }
        throw new Error('Signal expired before a panel could be opened');
    }

    // Check for a pending placeholder (resolveCustomEditor handoff).
    // If one matches this signal, navigate the existing placeholder tab
    // instead of creating a second panel — eliminates the flicker.
    let handedOff = false;
    let viewerReady;
    let integratedBrowserOpened = false;
    let externalBrowserOpened = false;
    let integratedBrowserPlaceholder = null;
    // Set when this request has reached a terminal state, so a panel disposal
    // arriving while the request is still running does not release the session
    // out from under it. See the handoff disposal handler below.
    const requestSettled = { done: false };
    let handoffPanelDisposed = false;
    if (useIntegratedBrowser || useExternalBrowser) {
        for (const [filePath, placeholder] of _pendingPlaceholders) {
            const exactHandoff = data.handoffPath
                && path.resolve(data.handoffPath) === placeholder.filePath;
            const legacyTitleMatch = !data.handoffPath
                && data.title
                && data.title.includes(placeholder.basename);
            if (exactHandoff || legacyTitleMatch) {
                integratedBrowserPlaceholder = { filePath, placeholder };
                break;
            }
        }
    }
    for (const [filePath, placeholder] of (
        useIntegratedBrowser || useExternalBrowser ? [] : _pendingPlaceholders
    )) {
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
                placeholder.panel.webview.html = _viewerPanelHtml(
                    openUrl,
                    vscode.env.remoteName === 'tunnel'
                        && !isLoopbackUrl(openUrl)
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
                        // VS Code reuses a single preview tab in the explorer,
                        // so clicking a second array disposes this placeholder
                        // while our own request is still driving it. Releasing
                        // the session here would kill the backend session that
                        // the in-flight readiness check is waiting on, which
                        // then surfaces as the misleading "Backend stopped
                        // answering before the viewer was ready". Record the
                        // disposal and let the terminal path do the release.
                        if (!requestSettled.done) {
                            handoffPanelDisposed = true;
                            log(`HANDOFF: panel disposed while request in flight; `
                                + `deferring session release to terminal path`);
                            return;
                        }
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
            if (integratedBrowserPlaceholder) {
                const { filePath, placeholder } = integratedBrowserPlaceholder;
                if (_pendingPlaceholders.get(filePath) === placeholder) {
                    _pendingPlaceholders.delete(filePath);
                }
                try {
                    placeholder.panel.dispose();
                    log(`CUSTOM-EDITOR: closed placeholder after integrated-browser handoff for ${placeholder.basename}`);
                } catch (error) {
                    log(`CUSTOM-EDITOR: placeholder already closed after integrated-browser handoff for ${placeholder.basename}: ${error.message}`);
                }
            }
        } else if (useExternalBrowser) {
            log(`openInExternalBrowser(${data.url})`);
            const opened = await openInExternalBrowser(
                data.url,
                requestId,
                data.serverId || null,
                data.windowId || logWindowId,
                viewerTimeoutMs,
                ensureActive
            );
            viewerReady = opened.viewerReady;
            externalBrowserOpened = true;
            log('openInExternalBrowser done');
            if (integratedBrowserPlaceholder) {
                const { filePath, placeholder } = integratedBrowserPlaceholder;
                if (_pendingPlaceholders.get(filePath) === placeholder) {
                    _pendingPlaceholders.delete(filePath);
                }
                try {
                    placeholder.panel.dispose();
                    log(`CUSTOM-EDITOR: closed placeholder after external-browser handoff for ${placeholder.basename}`);
                } catch (error) {
                    log(`CUSTOM-EDITOR: placeholder already closed after external-browser handoff for ${placeholder.basename}: ${error.message}`);
                }
            }
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
        try {
            advanceAck('panel_opened');
        } catch (error) {
            // The browser command is already a visible side effect. If the
            // exact request loses its claim before panel_opened is persisted,
            // do not leave the process-wide reservation stuck at `pending`
            // and do not leave an unowned backend session behind.
            if (integratedBrowserOpened || externalBrowserOpened) {
                releaseUrlSession(openUrl, data.url, data.serverId || null);
            }
            throw error;
        }
    }

    // Everything above claimed shared state: a pending placeholder, an entry
    // in _openPanels, or a browser-command slot. Everything below is this
    // request waiting on its own backend and display. Hand the queue on here so
    // the next launch proceeds while this array is still loading.
    releaseQueue('panel open; readiness continues off-queue');

    try {
        const pingUrl = pingUrlFromViewerUrl(openUrl);
        if (!pingUrl) throw new Error('Unable to derive backend ping URL');
        const metadataUrl = sessionMetadataUrlFromViewerUrl(openUrl);
        if (!metadataUrl) throw new Error('Unable to derive viewer session URL');
        // A disposal that already happened is the true cause of everything that
        // would fail below, so report it before probing the backend. Otherwise
        // a preview-tab replacement is diagnosed as a backend outage.
        if (handoffPanelDisposed) {
            throw new Error('Viewer panel closed before its first frame rendered');
        }
        // This gate exists to fail fast when the backend is genuinely gone. It
        // must not fail at all on a path that simply has not answered yet: the
        // panel is already up, and both gates below (session readiness, then
        // the viewer's own first frame) have their own budgets and produce far
        // more accurate messages. Abandoning here on a no-verdict reported a
        // cold devtunnel as a backend outage.
        const viaRelay = !isLoopbackUrl(pingUrl);
        const readiness = viaRelay
            ? await hedgedProbeStatus(pingUrl, data.serverId || null)
            : await probeUntilVerdict(
                pingUrl,
                data.serverId || null,
                READINESS_PROBE_TIMEOUTS_MS,
                ensureActive
            );
        ensureActive();
        if (readiness === PROBE_DEAD) {
            // Proof: a well-formed answer from someone else, or a refused
            // loopback port. Name which one, so a port taken over by a newer
            // launch is not diagnosed as a crash.
            throw new Error(data.serverId
                ? 'Another ArrayView backend now owns this port — '
                    + 'the session this tab was opened for is gone'
                : 'Backend stopped answering before the viewer was ready');
        }
        if (readiness === PROBE_UNKNOWN) {
            log('READY: ping never returned a verdict; continuing to session '
                + `readiness (relay=${viaRelay})`);
        }
        if (readiness === PROBE_RELAY_DOWN) {
            // Still not fatal — the panel is up and the gates below have their
            // own budgets — but name it, because "the relay is not carrying
            // the port" and "the probe was black-holed" look identical in a
            // log that records both as no-verdict.
            log('READY: relay answered but its connector is detached; '
                + 'continuing to session readiness');
        }
        // The array may still be loading; the panel is already up showing the
        // viewer's loading state. A load that failed answers 422 here with its
        // reason instead of never becoming ready.
        const sessionState = await waitForSessionReady(metadataUrl, metadataWaitMs);
        if (sessionState.loadError) {
            throw new Error(`ArrayView could not open this file: ${sessionState.loadError}`);
        }
        if (!sessionState.ready) {
            throw new Error('Viewer session did not become ready after panel opened');
        }
        const viewerError = await viewerReady;
        if (viewerError) throw viewerError;
        ensureActive();
        requestSettled.done = true;
        advanceAck('visibility_verified');
        advanceAck('backend_ready');
        return;
    } catch (error) {
        requestSettled.done = true;
        if (integratedBrowserOpened || externalBrowserOpened) {
            releaseUrlSession(openUrl, data.url, data.serverId || null);
        } else if (handedOff && handoffPanelDisposed) {
            // The disposal handler deferred this so it would not race the
            // readiness check above. The request is terminal now, so the
            // session that no longer has a panel must still be reclaimed.
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

    // Read what this window last injected, BEFORE the replace() below overwrites
    // it.  The collection is the only per-window store that survives a reload,
    // and after a reload it still holds the id the surviving terminals carry.
    const _readEnvCollection = (name) => {
        try {
            const entry = envCollection.get(name);
            return entry && entry.value ? entry.value : null;
        } catch (_) {
            return null;
        }
    };
    const previousId = _readEnvCollection('ARRAYVIEW_WINDOW_ID');
    const previousChain = (_readEnvCollection('ARRAYVIEW_WINDOW_CHAIN') || '')
        .split(',')
        .filter(Boolean);

    if (OWN_HOOK_TAG) {
        // hookTag is already stable (same IPC socket path → same SHA256 hash)
        windowId = OWN_HOOK_TAG;
    } else {
        // macOS local: reuse the previous window ID stored in the persistent env
        // collection so terminals that already have ARRAYVIEW_WINDOW_ID set
        // continue to target the correct registration after an extension restart.
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

    // Terminals opened before a reload still carry the id this window used then:
    // reloading rotates the IPC socket, and VS Code applies an environment
    // collection to NEW terminals only.  Publishing the ids we replaced lets
    // Python resolve those terminals to this window instead of failing the
    // exact-window check.  The chain is kept so a terminal that has survived
    // several reloads stays reachable, and bounded so it cannot grow forever.
    const supersedes = [];
    for (const candidate of [previousId, ...previousChain]) {
        if (candidate && candidate !== windowId && !supersedes.includes(candidate)) {
            supersedes.push(candidate);
        }
    }
    const supersededIds = supersedes.slice(0, MAX_SUPERSEDED_WINDOW_IDS);

    try {
        envCollection.replace('ARRAYVIEW_WINDOW_ID', windowId);
        envCollection.replace('ARRAYVIEW_WINDOW_CHAIN', supersededIds.join(','));
        log(`ENV: set ARRAYVIEW_WINDOW_ID=${windowId}`);
        if (supersededIds.length) log(`ENV: supersedes ${supersededIds.join(',')}`);
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
            supersedes: supersededIds,  // window ids whose terminals still point here
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

    _reportExtensionVersionSkew(windowId, version);

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

    const openFolderCmd = vscode.commands.registerCommand('arrayview.openFolder', async (uri) => {
        let folderPath;
        if (uri && uri.fsPath) {
            folderPath = uri.fsPath;
        } else {
            const selected = await vscode.window.showOpenDialog({
                canSelectFiles: false,
                canSelectFolders: true,
                canSelectMany: false,
                openLabel: 'Open in ArrayView',
            });
            if (!selected || !selected.length) return;
            folderPath = selected[0].fsPath;
        }

        try {
            await openFolderInArrayView(folderPath);
        } catch (e) {
            log(`COMMAND: openFolder failed: ${e.message}`);
            vscode.window.showErrorMessage(`ArrayView: ${e.message}`);
        }
    });
    context.subscriptions.push(openFolderCmd);

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
        probeArrayViewStatus,
        probeUntilVerdict,
        hedgedProbeStatus,
        PROBE_OK,
        PROBE_DEAD,
        PROBE_UNKNOWN,
        PROBE_RELAY_DOWN,
        localBackendIdentity,
        LOCAL_MINE,
        LOCAL_FOREIGN,
        LOCAL_UNKNOWN,
        _setRetryTiming,
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
        processSignalData,
        isSignalQueueBusy: () => isProcessingSignal,
        _remainingSignalMs,
        tryOpenSignalFile,
        _registerHandoffPlaceholder,
        openFolderInArrayView,
        pendingPlaceholders: _pendingPlaceholders,
        _viewerPanelHtml,
        _integratedBrowserLaunchUrl,
        integratedBrowserCommandAvailable,
        reserveDirectIntegratedBrowser,
        _runIntegratedBrowserCommand,
        waitForBackendViewerReady,
        waitForViewerReady,
        openInIntegratedBrowser,
        openInExternalBrowser,
        openInWebviewPanel,
        _openPanels,
        extensionInstanceId: EXTENSION_INSTANCE_ID,
        signalDir: SIGNAL_DIR,
        setWindowId(windowId) { logWindowId = windowId; },
        setTargetedSignalFile(filePath) { TARGETED_SIGNAL_FILE = filePath; },
    },
};
