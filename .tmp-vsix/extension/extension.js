const vscode = require('vscode');
const fs = require('fs');
const path = require('path');
const os = require('os');

const SIGNAL_DIR = path.join(os.homedir(), '.arrayview');
const SIGNAL_FILE = path.join(SIGNAL_DIR, 'open-request-v0310.json');
const LOG_FILE = path.join(SIGNAL_DIR, 'extension.log');

let _version = 'unknown';
let _panel;

function log(msg) {
    const line = `[${new Date().toISOString()}] ${msg}\n`;
    try { fs.appendFileSync(LOG_FILE, line); } catch (_) {}
    console.log(`[arrayview-opener] ${msg}`);
}

function getPanelColumn() {
    return vscode.window.activeTextEditor?.viewColumn || vscode.ViewColumn.Active;
}

function buildCandidateUrls(url) {
    const first = new URL(url);
    const candidates = [];

    if (!first.searchParams.has('transport')) {
        first.searchParams.set('transport', 'http');
    }
    candidates.push(first.toString());

    if (first.hostname === 'localhost') {
        const alt = new URL(first.toString());
        alt.hostname = '127.0.0.1';
        candidates.push(alt.toString());
    }

    return candidates;
}

async function openInSimpleBrowser(url, reason) {
    log(`REMOTE: simpleBrowser fallback starting reason=${reason} url=${url}`);
    try {
        let finalUrl = url;
        try {
            const original = new URL(url);
            const resolved = await vscode.env.asExternalUri(vscode.Uri.parse(url));
            const resolvedUrl = resolved.toString();
            log(`REMOTE: asExternalUri resolved reason=${reason} input=${url} resolved=${resolvedUrl}`);
            if (resolvedUrl) {
                const resolvedParsed = new URL(resolvedUrl);
                resolvedParsed.pathname = original.pathname || '/';
                resolvedParsed.search = original.search;
                resolvedParsed.hash = original.hash;
                finalUrl = resolvedParsed.toString();
                log(`REMOTE: reconstructed fallback url reason=${reason} finalUrl=${finalUrl}`);
            }
        } catch (e) {
            log(`REMOTE: asExternalUri FAILED reason=${reason} input=${url} error=${e.message}`);
        }

        await vscode.commands.executeCommand('simpleBrowser.show', finalUrl);
        log(`REMOTE: simpleBrowser fallback completed reason=${reason} finalUrl=${finalUrl}`);
        return true;
    } catch (e) {
        log(`REMOTE: simpleBrowser fallback FAILED reason=${reason} url=${url} error=${e.message}`);
        return false;
    }
}

function createPanel(url) {
    const parsed = new URL(url);
    const port = Number(parsed.port || '80');
    const candidateUrls = buildCandidateUrls(url);
    const title = 'ArrayView';

    if (_panel) {
        _panel.title = title;
        _panel.webview.options = {
            enableScripts: true,
            retainContextWhenHidden: true,
            portMapping: [{ webviewPort: port, extensionHostPort: port }],
        };
        _panel.webview.html = getWebviewHtml(candidateUrls, port);
        _panel.reveal(getPanelColumn(), true);
        return _panel;
    }

    _panel = vscode.window.createWebviewPanel(
        'arrayview',
        title,
        getPanelColumn(),
        {
            enableScripts: true,
            retainContextWhenHidden: true,
            portMapping: [{ webviewPort: port, extensionHostPort: port }],
        }
    );
    _panel.onDidDispose(() => {
        _panel = undefined;
    });
    _panel.webview.onDidReceiveMessage((message) => {
        if (!message) {
            return;
        }
        if (message.type === 'log') {
            log(`REMOTE BOOTSTRAP: ${message.message}`);
            return;
        }
        if (message.type === 'viewer-loaded') {
            log('REMOTE: viewer iframe loaded');
            return;
        }
        if (message.type === 'viewer-error') {
            log(`REMOTE: viewer iframe error: ${message.message}`);
            return;
        }
        if (message.type === 'viewer-phase') {
            const detail = message.detail ? ` ${JSON.stringify(message.detail)}` : '';
            log(`REMOTE: viewer phase=${message.phase}${detail}`);
            return;
        }
        if (message.type === 'open-simple-browser') {
            openInSimpleBrowser(message.url, message.reason || 'webview-timeout');
        }
    });
    _panel.webview.html = getWebviewHtml(candidateUrls, port);
    return _panel;
}

function getWebviewHtml(urls, port) {
    const escapedUrls = JSON.stringify(urls);
    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; frame-src http://localhost:${port} https://localhost:${port} http://127.0.0.1:${port} https://127.0.0.1:${port}; script-src 'unsafe-inline'; style-src 'unsafe-inline';">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        html, body {
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
            background: #111;
            color: #f3f3f3;
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        }
        body {
            display: flex;
            align-items: center;
            justify-content: center;
        }
        #status {
            font-size: 16px;
            letter-spacing: 0.04em;
            text-transform: none;
            opacity: 0.95;
            max-width: 560px;
            line-height: 1.5;
            text-align: center;
            padding: 24px;
        }
        iframe {
            width: 100%;
            height: 100%;
            border: 0;
            display: none;
        }
    </style>
</head>
<body>
    <div id="status">Connecting ArrayView...</div>
    <iframe id="viewer" referrerpolicy="no-referrer"></iframe>
    <script>
        const vscode = acquireVsCodeApi();
        const candidateUrls = ${escapedUrls};
        const status = document.getElementById('status');
        const viewer = document.getElementById('viewer');
        let candidateIndex = -1;
        let viewerReady = false;
        let fallbackTimer = null;

        function report(message) {
            try {
                vscode.postMessage({ type: 'log', message });
            } catch (_) {
            }
        }

        window.addEventListener('error', (event) => {
            report('window error: ' + event.message);
            status.textContent = 'ArrayView bootstrap error: ' + event.message;
        });

        viewer.addEventListener('load', () => {
            report('viewer iframe load event url=' + viewer.src);
            vscode.postMessage({ type: 'viewer-loaded' });
        });

        viewer.addEventListener('error', () => {
            const message = 'viewer iframe failed to load';
            report(message);
            status.textContent = 'ArrayView failed to load inside the remote preview.';
            vscode.postMessage({ type: 'viewer-error', message });
        });

        window.addEventListener('message', (event) => {
            const data = event.data;
            if (!data || data.source !== 'arrayview-viewer') {
                return;
            }
            vscode.postMessage({ type: 'viewer-phase', phase: data.phase, detail: data.detail || null });
            if (data.phase === 'frame-rendered') {
                viewerReady = true;
                status.style.display = 'none';
                viewer.style.display = 'block';
                if (fallbackTimer) {
                    clearTimeout(fallbackTimer);
                    fallbackTimer = null;
                }
            }
        });

        function scheduleFallback() {
            if (fallbackTimer) {
                clearTimeout(fallbackTimer);
            }
            fallbackTimer = window.setTimeout(() => {
                if (viewerReady) {
                    return;
                }
                if (candidateIndex + 1 < candidateUrls.length) {
                    report('viewer not ready; trying fallback candidate');
                    loadViewer('fallback');
                    return;
                }
                status.textContent = 'ArrayView opened, but no frame rendered yet.';
                report('viewer did not report a rendered frame before timeout');
                try {
                    vscode.postMessage({
                        type: 'open-simple-browser',
                        url: candidateUrls[0],
                        reason: 'no-frame-rendered',
                    });
                } catch (_) {
                }
            }, 5000);
        }

        function loadViewer(reason) {
            candidateIndex += 1;
            const viewerUrl = candidateUrls[candidateIndex];
            viewerReady = false;
            report('loading viewer iframe=' + viewerUrl + ' reason=' + reason + ' candidate=' + (candidateIndex + 1) + '/' + candidateUrls.length);
            status.style.display = 'block';
            status.textContent = 'Loading ArrayView...';
            viewer.src = viewerUrl;
            viewer.style.display = 'block';
            scheduleFallback();
        }

        report('webview bootstrap ready candidates=' + candidateUrls.join(','));
        loadViewer('initial');
    </script>
</body>
</html>`;
}

async function tryOpenSignalFile() {
    try {
        if (!fs.existsSync(SIGNAL_FILE)) return;
        log('SIGNAL: found signal file');
        const raw = fs.readFileSync(SIGNAL_FILE, 'utf8');
        try { fs.unlinkSync(SIGNAL_FILE); } catch (e) { log(`SIGNAL: unlink failed: ${e.message}`); }

        let data;
        try { data = JSON.parse(raw); } catch (e) { log(`SIGNAL: JSON parse failed: ${e.message}`); return; }

        const url = data.url;
        if (!url) { log('SIGNAL: no url in signal file'); return; }

        log(`SIGNAL: url=${url}`);

        if (vscode.env.remoteName) {
            log(`REMOTE: opening webview panel for ${url}`);
            try {
                createPanel(url);
                log('REMOTE: webview panel opened');
            } catch (e) {
                log(`REMOTE: webview FAILED: ${e.message}`);
            }
        } else {
            log(`LOCAL: calling simpleBrowser.show(${url})...`);
            try {
                await vscode.commands.executeCommand('simpleBrowser.show', url);
                log('LOCAL: simpleBrowser.show completed');
            } catch (e) {
                log(`LOCAL: simpleBrowser.show FAILED: ${e.message}`);
            }
        }
    } catch (e) {
        log(`ERROR in tryOpenSignalFile: ${e.message}\n${e.stack}`);
    }
}

function activate(context) {
    _version = context.extension.packageJSON.version;
    log(`=== ACTIVATE v${_version} ===`);
    log(`remoteName=${vscode.env.remoteName} appHost=${vscode.env.appHost}`);

    try { fs.mkdirSync(SIGNAL_DIR, { recursive: true }); } catch (_) {}

    // Check for a signal file that arrived before we activated
    tryOpenSignalFile();

    // Poll every 1s (safety net)
    const interval = setInterval(() => tryOpenSignalFile(), 1000);
    context.subscriptions.push({ dispose: () => clearInterval(interval) });

    // fs.watch for faster response
    try {
        const watcher = fs.watch(SIGNAL_DIR, (eventType, filename) => {
            if (filename === path.basename(SIGNAL_FILE)) {
                log(`WATCH: event=${eventType} file=${filename}`);
                setTimeout(() => tryOpenSignalFile(), 100);
            }
        });
        context.subscriptions.push({ dispose: () => watcher.close() });
        log(`WATCH: fs.watch active on ${SIGNAL_DIR}`);
    } catch (e) {
        log(`WATCH: fs.watch failed (polling still active): ${e.message}`);
    }

    log('=== ACTIVATE DONE ===');
}

function deactivate() {
    log(`deactivate v${_version}`);
}

module.exports = { activate, deactivate };
