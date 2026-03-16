const vscode = require('vscode');
const fs = require('fs');
const path = require('path');
const os = require('os');
const crypto = require('crypto');

const SIGNAL_DIR = path.join(os.homedir(), '.arrayview');
const SIGNAL_FILE = path.join(SIGNAL_DIR, 'open-request-v0900.json');  // fallback
const LOG_FILE = path.join(SIGNAL_DIR, 'extension.log');

// Per-window targeted signal file: Python writes to a file named by the SHA256
// of VSCODE_IPC_HOOK_CLI, which is unique per VS Code window on the remote.
// The extension checks its own targeted file first, then the fallback.
const OWN_IPC_HOOK = process.env.VSCODE_IPC_HOOK_CLI || '';
const OWN_HOOK_TAG = OWN_IPC_HOOK
    ? crypto.createHash('sha256').update(OWN_IPC_HOOK).digest('hex').slice(0, 16)
    : '';
const TARGETED_SIGNAL_FILE = OWN_HOOK_TAG
    ? path.join(SIGNAL_DIR, `open-request-ipc-${OWN_HOOK_TAG}.json`)
    : null;

let version = 'unknown';
let isProcessingSignal = false;
let lastHandledRequestId = null;
let lastHandledUrl = null;
let lastHandledAt = 0;

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

async function tryOpenSignalFile() {
    // If we are currently showing a URL, leave any pending signal files on disk.
    // The 1-second polling loop will pick them up once we are done.  This avoids
    // in-memory queues that can be lost when the extension host reloads.
    if (isProcessingSignal) return;

    // Check targeted file first (matches our window's IPC hook), then primary,
    // then compat signal files for older/published arrayview Python versions.
    const candidates = [];
    if (TARGETED_SIGNAL_FILE) candidates.push(TARGETED_SIGNAL_FILE);
    candidates.push(SIGNAL_FILE);
    // Compat: older arrayview releases write to these filenames
    candidates.push(
        path.join(SIGNAL_DIR, 'open-request-v0800.json'),
        path.join(SIGNAL_DIR, 'open-request-v0400.json'),
    );

    for (const signalFile of candidates) {
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
        try { fs.unlinkSync(claimedFile); } catch (_) {}

        let data;
        try { data = JSON.parse(raw); } catch (err) {
            log(`SIGNAL: invalid JSON: ${err.message}`);
            continue;
        }

        if (isExpiredSignal(data)) continue;

        try {
            await processSignalData(data);
        } catch (error) {
            log(`ERROR: ${error.message}`);
        }
        return;  // processed one signal, done for this tick
    }
}

async function processSignalData(data) {
    isProcessingSignal = true;
    try {
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

        log(`SIGNAL: requestId=${requestId || 'none'} url=${url}`);
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

        log(`simpleBrowser.show(${openUrl})`);
        // Race against a 12s timeout so isProcessingSignal is always released
        // even if the command's promise never resolves (VS Code tunnel bug).
        await Promise.race([
            vscode.commands.executeCommand('simpleBrowser.show', openUrl),
            new Promise(resolve => setTimeout(resolve, 12000)),
        ]);
        log('simpleBrowser.show done');
    } catch (error) {
        log(`ERROR: ${error.message}`);
    } finally {
        isProcessingSignal = false;
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

    // Write registration so Python can find this window's hookTag.
    // Python reads ~/.arrayview/window-<hookTag>.json to know which targeted
    // signal file to write for this specific VS Code window.
    if (OWN_HOOK_TAG) {
        const regFile = path.join(SIGNAL_DIR, `window-${OWN_HOOK_TAG}.json`);
        try {
            fs.writeFileSync(regFile, JSON.stringify({ hookTag: OWN_HOOK_TAG, pid: process.pid, ts: Date.now() }));
            log(`REGISTER: wrote ${path.basename(regFile)}`);
            context.subscriptions.push({ dispose: () => {
                try { fs.unlinkSync(regFile); } catch (_) {}
                log(`REGISTER: deleted ${path.basename(regFile)}`);
            }});
        } catch (e) {
            log(`REGISTER: failed to write: ${e.message}`);
        }
    }

    cleanupStaleFiles();

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

    log('=== ACTIVATE DONE ===');
}

function deactivate() {
    log(`deactivate v${version}`);
}

module.exports = { activate, deactivate };
