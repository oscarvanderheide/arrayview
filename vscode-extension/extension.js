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
    shouldRemoveSameTunnelRegistration,
} = require('./lifecycle_helpers');

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

// Track open webview panels by URL so we can reveal instead of re-creating.
const _openPanels = new Map(); // url -> vscode.WebviewPanel

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

function _shellCommand(command, args) {
    return [command, ...args].map((part) => {
        if (/^[A-Za-z0-9_./:=+-]+$/.test(part)) return part;
        return `'${String(part).replace(/'/g, `'\\''`)}'`;
    }).join(' ');
}

function _arrayviewLaunchCandidates() {
    const candidates = [];
    const folders = vscode.workspace.workspaceFolders || [];
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
        const candidates = _arrayviewLaunchCandidates();

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
                    cwd: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || path.dirname(filePath),
                    detached: true,
                    stdio: ['ignore', 'pipe', 'pipe'],
                    env: { ...process.env, TERM_PROGRAM: 'vscode', ARRAYVIEW_WINDOW_ID: logWindowId || '' },
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
        try {
            await launchArrayViewFile(filePath, title);
            log(`CUSTOM-EDITOR: launched network viewer for ${filePath}`);
            _pendingPlaceholders.set(filePath, { panel: webviewPanel, basename: title });
            webviewPanel.onDidDispose(() => {
                _pendingPlaceholders.delete(filePath);
                log(`CUSTOM-EDITOR: placeholder disposed for ${title}`);
            });
            // Safety timeout: if no signal arrives within 30 s, show an error.
            setTimeout(() => {
                if (_pendingPlaceholders.has(filePath)) {
                    _pendingPlaceholders.delete(filePath);
                    try {
                        webviewPanel.webview.html = `<html><body style="color:#c00;padding:2em;font-family:monospace;background:#1e1e1e">
                            <h2>ArrayView failed to start</h2>
                            <p>The Python server did not respond. Check ~/.arrayview/extension.log for details.</p></body></html>`;
                    } catch (_) { /* panel already disposed */ }
                }
            }, 30000);
        } catch (e) {
            log(`CUSTOM-EDITOR: error: ${e.message}\n${e.stack || ''}`);
            webviewPanel.webview.html = `<html><body style="color:#c00;padding:2em;font-family:monospace">
                <h2>ArrayView failed to open</h2><pre>${e.message}</pre>
                <p>Check ~/.arrayview/extension.log for details.</p></body></html>`;
        }
    }
}

function httpOk(url, timeoutMs = 1500) {
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
            done((res.statusCode || 0) >= 200 && (res.statusCode || 0) < 500);
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

function httpPostOk(url, timeoutMs = 1500) {
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
        const req = lib.request(parsed, { method: 'POST', timeout: timeoutMs }, (res) => {
            res.resume();
            done((res.statusCode || 0) >= 200 && (res.statusCode || 0) < 500);
        });
        req.on('timeout', () => {
            req.destroy();
            done(false);
        });
        req.on('error', () => done(false));
        req.end();
    });
}

function releaseUrlSession(url) {
    const sids = collectReleaseSidsFromUrl(url);
    if (!sids.length) return;
    const origin = new URL(url).origin;
    for (const sid of sids) {
        const releaseUrl = `${origin}/release/${encodeURIComponent(sid)}`;
        void httpPostOk(releaseUrl).then((ok) => {
            log(`PANEL: release sid=${sid.slice(0, 8)} ok=${ok}`);
        });
    }
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
                try {
                    const data = JSON.parse(fs.readFileSync(fullPath, 'utf8'));
                    if (isExpiredSignal(data)) {
                        fs.unlinkSync(fullPath);
                        log(`CLEANUP: removed expired claimed ${f}`);
                    } else {
                        // Restore to the original un-claimed basename so the
                        // poll re-claims it. Skip if that file already exists
                        // (another claim/restore or a fresh write beat us).
                        const original = f.replace(/\.claimed-\d+$/, '');
                        const originalPath = path.join(SIGNAL_DIR, original);
                        if (!fs.existsSync(originalPath)) {
                            fs.renameSync(fullPath, originalPath);
                            log(`CLEANUP: restored non-expired claimed ${f} -> ${original}`);
                        } else {
                            fs.unlinkSync(fullPath);
                            log(`CLEANUP: dropped claimed ${f} (original already present)`);
                        }
                    }
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

        log(`DISPATCH: file=${path.basename(signalFile)} mode=${data.mode} hasUrl=${!!data.url} keys=${Object.keys(data).join(',')}`);
        try {
            await processSignalData(data);
        } catch (error) {
            log(`ERROR: ${error.message}`);
        }
        return;  // processed one signal, done for this tick
    }
}

// Open or reveal a VS Code WebviewPanel for the given server URL.
// The panel is only a URL wrapper: ArrayView data and controls still flow
// through the FastAPI/WebSocket backend, never direct Python/webview IPC.
function _viewerPanelHtml(url) {
    const nonce = crypto.randomBytes(16).toString('hex');
    const jsonUrl = JSON.stringify(url);
    return `<!DOCTYPE html>
<html>
<head>
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; frame-src *; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';">
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
const arrayviewUrl = ${jsonUrl};
const frame = document.getElementById('f');
let viewerReady = false;
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
    if (viewerReady) return;
    if (reloadTimer) { clearTimeout(reloadTimer); }
    reloadTimer = setTimeout(() => {
        reloadTimer = null;
        if (viewerReady) return;
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
    if (msg.phase === 'script-loaded' || msg.phase === 'frame-rendered') {
        if (!viewerReady) {
            viewerReady = true;
            if (reloadTimer) { clearTimeout(reloadTimer); reloadTimer = null; }
            console.log('[arrayview-opener] viewer phase ' + msg.phase);
        }
    }
});
frame.addEventListener('load', () => {
    console.log('[arrayview-opener] iframe loaded ' + arrayviewUrl);
    scheduleReload();
});
frame.addEventListener('error', () => console.log('[arrayview-opener] iframe error ' + arrayviewUrl));
frame.src = arrayviewUrl;
</script>
</body>
</html>`;
}

async function openInWebviewPanel(url, title, floating = false) {
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

    panel.webview.html = _viewerPanelHtml(url);

    _openPanels.set(url, panel);
    const pingUrl = pingUrlFromViewerUrl(url);
    let panelDisposed = false;
    panel.onDidDispose(() => {
        panelDisposed = true;
        _openPanels.delete(url);
        if (vscode.env.remoteName) {
            // Remote/tunnel: the server runs with persist=True, so sessions
            // should survive tab close.  VS Code's preview-tab mode disposes
            // panels when a new file opens — releasing immediately would
            // kill the session before the user is done (e.g., they just
            // switched to another array in the same preview slot).  Delay
            // release by 60s so brief preview-tab swaps don't yank the
            // session, but stale sessions still get cleaned up.
            const releaseUrl = url;
            setTimeout(() => {
                if (!_openPanels.has(releaseUrl)) {
                    releaseUrlSession(releaseUrl);
                }
            }, 60000);
        } else {
            releaseUrlSession(url);
        }
    });
    log(`PANEL: created "${label}" for ${url}`);

    if (pingUrl) {
        setTimeout(async () => {
            for (let attempt = 0; attempt <= 10 && !panelDisposed; attempt++) {
                if (await httpOk(pingUrl)) return;
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
}

/**
 * Ensure a forwarded port has public visibility so the devtunnel URL is
 * accessible from the VS Code client.  VS Code auto-forwards ports as
 * private by default; the devtunnel URL only works if the port is public.
 *
 * Two-pronged approach:
 *   1. Write remote.portsAttributes with privacy=public via the settings
 *      API — immediate, no file-watcher delay.  This ensures FUTURE
 *      forwards of this port use public privacy.
 *   2. If the port is ALREADY forwarded as private, change its privacy
 *      via the internal `remote.tunnel.privacypublic` command.  This
 *      closes the existing private tunnel and re-forwards with public
 *      visibility.  The command is registered by VS Code's tunnel view
 *      when the tunnel provider supports privacy changes (devtunnels do).
 *      If the command doesn't exist (older VS Code, no privacy support),
 *      the error is caught and logged — the settings write from step 1
 *      still helps for future forwards.
 */
async function ensurePortPublic(port) {
    // Step 1: write portsAttributes via the settings API
    try {
        const config = vscode.workspace.getConfiguration('remote');
        let attrs = config.get('portsAttributes') || {};
        attrs[String(port)] = Object.assign({}, attrs[String(port)], {
            protocol: 'http',
            label: 'ArrayView',
            onAutoForward: 'silent',
            privacy: 'public',
        });
        await config.update('portsAttributes', attrs, vscode.ConfigurationTarget.Global);
        log(`PORT: wrote portsAttributes[${port}] privacy=public`);
    } catch (e) {
        log(`PORT: failed to write portsAttributes: ${e.message || e}`);
    }

    // Step 2: change privacy of already-forwarded port
    // The privacy command (remote.tunnel.privacypublic) is lazily
    // registered by VS Code's Forwarded Ports view.  In a pure tunnel
    // session (no Remote-SSH), it may not be loaded yet.  Try focusing
    // the forwarded ports view first to trigger lazy loading, then retry.
    const tunnelItem = {
        tunnelType: 1,
        remoteHost: 'localhost',
        remotePort: port,
        localPort: port,
        name: 'ArrayView',
        source: { source: 'user', description: 'ArrayView' },
    };

    let privacyDone = false;
    try {
        await vscode.commands.executeCommand(
            'remote.tunnel.privacypublic', tunnelItem
        );
        privacyDone = true;
        log(`PORT: changed privacy to public via command`);
    } catch (e) {
        log(`PORT: privacy command failed: ${e.message || e}`);
    }

    if (!privacyDone) {
        // Retry: force-load forwarded ports view, then retry the command
        log(`PORT: privacy not found — loading forwarded ports view...`);
        try {
            await vscode.commands.executeCommand('~remote.forwardedPorts.focus');
            await new Promise(r => setTimeout(r, 500));
        } catch (_) {}

        // Check if the command is now registered
        const cmds = await vscode.commands.getCommands(true);
        if (cmds.includes('remote.tunnel.privacypublic')) {
            try {
                log(`PORT: privacy command found after view load — retrying`);

                // Re-call asExternalUri to refresh the tunnel item reference
                // (the privacy command needs the current forwarded tunnel item)
                await vscode.env.asExternalUri(
                    vscode.Uri.parse(`http://localhost:${port}/`)
                ).catch(() => {});

                await vscode.commands.executeCommand(
                    'remote.tunnel.privacypublic', tunnelItem
                );
                privacyDone = true;
                log(`PORT: changed privacy to public via command (retry)`);
            } catch (e2) {
                log(`PORT: privacy retry failed: ${e2.message || e2}`);
            }
        } else {
            log(`PORT: privacypublic still not available after view load`);
        }
    }
}

async function resolveRemoteViewerUrl(url) {
    let port = 8000;
    try { port = parseInt(new URL(url).port, 10) || 8000; } catch (_) {}
    let origQuery = '';
    try { origQuery = new URL(url).search; } catch (_) {}
    const baseUri = vscode.Uri.parse(`http://localhost:${port}/`);
    const attempts = [
        { timeoutMs: 10000, pauseMs: 0 },
        { timeoutMs: 15000, pauseMs: 750 },
        { timeoutMs: 20000, pauseMs: 1500 },
    ];

    for (let i = 0; i < attempts.length; i++) {
        const attempt = attempts[i];
        if (attempt.pauseMs) {
            await new Promise(resolve => setTimeout(resolve, attempt.pauseMs));
        }
        try {
            log(`REMOTE: asExternalUri(http://localhost:${port}/) attempt=${i + 1}`);
            const externalUri = await Promise.race([
                vscode.env.asExternalUri(baseUri),
                new Promise((_, reject) =>
                    setTimeout(() => reject(new Error(`asExternalUri timeout after ${attempt.timeoutMs}ms`)), attempt.timeoutMs)),
            ]);
            const externalBase = externalUri.toString().replace(/\/$/, '');
            log(`REMOTE: → ${externalBase}`);

            await ensurePortPublic(port);

            const finalUrl = externalBase + '/' + origQuery;
            log(`REMOTE: final URL = ${finalUrl}`);
            return finalUrl;
        } catch (err) {
            log(`REMOTE: asExternalUri attempt ${i + 1} failed: ${err.message}`);
            if (i === 0) {
                try {
                    await vscode.commands.executeCommand('~remote.forwardedPorts.focus');
                } catch (_) {}
            }
        }
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
    // orphaned body promise is harmless: the panel either eventually opens
    // or it doesn't, but the queue is no longer blocked.
    const SIGNAL_HARD_TIMEOUT_MS = 70000;
    try {
        await Promise.race([
            _processSignalDataBody(data),
            new Promise((_, reject) =>
                setTimeout(() => reject(new Error(`processSignalData hard timeout after ${SIGNAL_HARD_TIMEOUT_MS}ms`)), SIGNAL_HARD_TIMEOUT_MS)
            ),
        ]);
    } catch (error) {
        log(`ERROR: ${error.message}`);
    } finally {
        isProcessingSignal = false;
        log(`UNLOCK: isProcessingSignal=false`);
        // Signal files for subsequent arrays remain on disk; the 1-second poll
        // will pick them up now that isProcessingSignal is false again.
    }
}

async function _processSignalDataBody(data) {
    log(`SIGNAL-DATA: mode=${data.mode} url=${data.url || '(none)'}`);
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
        const remoteUrl = await resolveRemoteViewerUrl(url);
        if (!remoteUrl) {
            log('REMOTE: failed to resolve external URI; leaving signal retry to reopen later');
            return;
        }
        openUrl = remoteUrl;
    }

    // Check for a pending placeholder (resolveCustomEditor handoff).
    // If one matches this signal, navigate the existing placeholder tab
    // instead of creating a second panel — eliminates the flicker.
    let handedOff = false;
    for (const [filePath, placeholder] of _pendingPlaceholders) {
        if (data.title && data.title.includes(placeholder.basename)) {
            _pendingPlaceholders.delete(filePath);
            try {
                placeholder.panel.webview.html = _viewerPanelHtml(openUrl);
                placeholder.panel.title = data.title || placeholder.title;
                log(`HANDOFF: navigated placeholder for ${placeholder.basename} to ${openUrl}`);
                handedOff = true;
            } catch (_) {
                log(`HANDOFF: placeholder panel disposed for ${placeholder.basename}`);
            }
            break;
        }
    }
    if (handedOff) return;

    log(`openInWebviewPanel(${openUrl})`);
    await openInWebviewPanel(openUrl, data.title, !!data.floating);
    log('openInWebviewPanel done');
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

module.exports = { activate, deactivate };
