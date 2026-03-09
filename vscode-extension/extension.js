const vscode = require('vscode');
const fs = require('fs');
const path = require('path');
const os = require('os');

const SIGNAL_DIR = path.join(os.homedir(), '.arrayview');
const SIGNAL_FILE = path.join(SIGNAL_DIR, 'open-request-v0900.json');
const LOG_FILE = path.join(SIGNAL_DIR, 'extension.log');

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

async function tryOpenSignalFile() {
    try {
        if (isProcessingSignal) return;
        if (!fs.existsSync(SIGNAL_FILE)) return;

        isProcessingSignal = true;

        const raw = fs.readFileSync(SIGNAL_FILE, 'utf8');
        try { fs.unlinkSync(SIGNAL_FILE); } catch (err) {
            log(`SIGNAL: unlink failed: ${err.message}`);
        }

        let data;
        try { data = JSON.parse(raw); } catch (err) {
            log(`SIGNAL: invalid JSON: ${err.message}`);
            return;
        }

        if (isExpiredSignal(data)) return;

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
                const externalUri = await vscode.env.asExternalUri(baseUri);
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
        await vscode.commands.executeCommand('simpleBrowser.show', openUrl);
        log('simpleBrowser.show completed');
    } catch (error) {
        log(`ERROR: ${error.message}`);
    } finally {
        isProcessingSignal = false;
    }
}

function activate(context) {
    version = context.extension.packageJSON.version;
    log(`=== ACTIVATE v${version} ===`);
    log(`remoteName=${vscode.env.remoteName} appHost=${vscode.env.appHost}`);

    try { fs.mkdirSync(SIGNAL_DIR, { recursive: true }); } catch (_) {}

    void tryOpenSignalFile();

    const interval = setInterval(() => void tryOpenSignalFile(), 1000);
    context.subscriptions.push({ dispose: () => clearInterval(interval) });

    try {
        const watcher = fs.watch(SIGNAL_DIR, (eventType, filename) => {
            if (filename === path.basename(SIGNAL_FILE)) {
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
