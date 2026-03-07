const vscode = require('vscode');
const fs = require('fs');
const http = require('http');
const https = require('https');
const path = require('path');
const os = require('os');

const SIGNAL_DIR = path.join(os.homedir(), '.arrayview');
const SIGNAL_FILE = path.join(SIGNAL_DIR, 'open-request-v0400.json');
const LOG_FILE = path.join(SIGNAL_DIR, 'extension.log');

let _version = 'unknown';
let _panel;
let _panelState;

function log(msg) {
    const line = `[${new Date().toISOString()}] ${msg}\n`;
    try { fs.appendFileSync(LOG_FILE, line); } catch (_) {}
    console.log(`[arrayview-opener] ${msg}`);
}

function getPanelColumn() {
    return vscode.window.activeTextEditor?.viewColumn || vscode.ViewColumn.Active;
}

function flattenHeaders(headers) {
    const flat = {};
    for (const [key, value] of Object.entries(headers || {})) {
        flat[key.toLowerCase()] = Array.isArray(value) ? value.join(', ') : String(value);
    }
    return flat;
}

function requestBuffer(url, options = {}) {
    return new Promise((resolve, reject) => {
        const parsed = new URL(url);
        const client = parsed.protocol === 'https:' ? https : http;
        const req = client.request(parsed, {
            method: options.method || 'GET',
            headers: options.headers || {},
        }, (res) => {
            const chunks = [];
            res.on('data', (chunk) => chunks.push(chunk));
            res.on('end', () => {
                resolve({
                    statusCode: res.statusCode || 0,
                    headers: flattenHeaders(res.headers),
                    body: Buffer.concat(chunks),
                });
            });
        });
        req.on('error', reject);
        if (options.body) {
            req.write(options.body);
        }
        req.end();
    });
}

async function requestText(url, options = {}) {
    const res = await requestBuffer(url, options);
    return {
        ...res,
        text: res.body.toString('utf8'),
    };
}

function buildViewerState(url) {
    const parsed = new URL(url);
    parsed.searchParams.set('transport', 'http');
    return {
        origin: parsed.origin,
        viewerUrl: parsed.toString(),
        viewerQuery: parsed.search,
    };
}

function resolveProxyUrl(state, requestUrl) {
    if (/^https?:\/\//i.test(requestUrl)) {
        return requestUrl;
    }
    if (requestUrl.startsWith('/')) {
        return `${state.origin}${requestUrl}`;
    }
    return new URL(requestUrl, state.viewerUrl).toString();
}

function makeLoadingHtml(message) {
    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        html, body {
            margin: 0;
            width: 100%;
            height: 100%;
            background: #111;
            color: #ddd;
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        }
        body {
            display: flex;
            align-items: center;
            justify-content: center;
        }
    </style>
</head>
<body>${message}</body>
</html>`;
}

function makeBridgeScript(state) {
    const query = JSON.stringify(state.viewerQuery);
    return `<script>
window.__ARRAYVIEW_QUERY__ = ${query};
window.__ARRAYVIEW_VSCODE_PROXY__ = true;

if (${query} && window.location.search !== ${query}) {
    try {
        window.history.replaceState(null, '', window.location.pathname + ${query} + window.location.hash);
    } catch (_) {
    }
}

(() => {
    const vscode = acquireVsCodeApi();
    const pending = new Map();
    let nextRequestId = 1;

    function decodeBase64(base64) {
        const binary = atob(base64 || '');
        const bytes = new Uint8Array(binary.length);
        for (let index = 0; index < binary.length; index += 1) {
            bytes[index] = binary.charCodeAt(index);
        }
        return bytes;
    }

    function createHeaders(headers) {
        const map = new Map();
        for (const [key, value] of Object.entries(headers || {})) {
            map.set(key.toLowerCase(), value);
        }
        return {
            get(name) {
                return map.get(String(name).toLowerCase()) || null;
            },
        };
    }

    function createResponse(payload) {
        const bytes = decodeBase64(payload.bodyBase64);
        const headers = createHeaders(payload.headers);
        const contentType = headers.get('content-type') || 'application/octet-stream';
        return {
            ok: payload.status >= 200 && payload.status < 300,
            status: payload.status,
            headers,
            async text() {
                if (typeof payload.bodyText === 'string') {
                    return payload.bodyText;
                }
                return new TextDecoder().decode(bytes);
            },
            async json() {
                return JSON.parse(await this.text());
            },
            async blob() {
                return new Blob([bytes], { type: contentType });
            },
            async arrayBuffer() {
                return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
            },
        };
    }

    window.addEventListener('message', (event) => {
        const message = event.data;
        if (!message || message.type !== 'proxy-fetch-result') {
            return;
        }
        const entry = pending.get(message.id);
        if (!entry) {
            return;
        }
        pending.delete(message.id);
        if (message.error) {
            entry.reject(new Error(message.error));
            return;
        }
        entry.resolve(createResponse(message));
    });

    window.fetch = (input, init = {}) => {
        const requestId = nextRequestId++;
        const requestUrl = typeof input === 'string' ? input : input.url;
        const requestInit = init || {};
        const headers = {};
        const sourceHeaders = requestInit.headers || (typeof input !== 'string' ? input.headers : undefined);
        if (sourceHeaders) {
            for (const [key, value] of new Headers(sourceHeaders).entries()) {
                headers[key] = value;
            }
        }

        let body = requestInit.body;
        if (body instanceof URLSearchParams) {
            body = body.toString();
        }
        if (body && typeof body !== 'string') {
            throw new Error('ArrayView proxy fetch currently supports string request bodies only');
        }

        return new Promise((resolve, reject) => {
            pending.set(requestId, { resolve, reject });
            vscode.postMessage({
                type: 'proxy-fetch',
                id: requestId,
                url: requestUrl,
                method: requestInit.method || (typeof input !== 'string' && input.method) || 'GET',
                headers,
                body: body || null,
            });
        });
    };
})();
</script>`;
}

function buildViewerHtml(rawHtml, state) {
    return rawHtml.replace('<script>', `${makeBridgeScript(state)}\n    <script>`);
}

async function loadViewerHtml(state) {
    const res = await requestText(state.viewerUrl);
    if (res.statusCode !== 200) {
        throw new Error(`viewer html request failed (${res.statusCode})`);
    }
    return buildViewerHtml(res.text, state);
}

async function handlePanelMessage(message) {
    if (!_panelState || !_panel || !message) {
        return;
    }

    if (message.type === 'proxy-fetch') {
        try {
            const targetUrl = resolveProxyUrl(_panelState, message.url);
            log(`REMOTE PROXY: fetch ${message.method || 'GET'} ${targetUrl}`);
            const res = await requestBuffer(targetUrl, {
                method: String(message.method || 'GET').toUpperCase(),
                headers: message.headers || {},
                body: message.body || undefined,
            });
            const contentType = res.headers['content-type'] || '';
            await _panel.webview.postMessage({
                type: 'proxy-fetch-result',
                id: message.id,
                status: res.statusCode,
                headers: res.headers,
                bodyText: contentType.includes('json') || contentType.startsWith('text/') ? res.body.toString('utf8') : null,
                bodyBase64: res.body.toString('base64'),
            });
        } catch (e) {
            log(`REMOTE PROXY: fetch failed ${message.url}: ${e.message}`);
            await _panel.webview.postMessage({
                type: 'proxy-fetch-result',
                id: message.id,
                error: e.message,
            });
        }
        return;
    }

    if (message.type === 'log') {
        log(`REMOTE PANEL: ${message.message}`);
    }
}

async function createRealViewerPanel(url) {
    _panelState = buildViewerState(url);
    const title = 'ArrayView';

    if (_panel) {
        _panel.title = title;
        _panel.webview.html = makeLoadingHtml('Loading ArrayView...');
        _panel.reveal(getPanelColumn(), true);
    } else {
        _panel = vscode.window.createWebviewPanel(
            'arrayview',
            title,
            getPanelColumn(),
            {
                enableScripts: true,
                retainContextWhenHidden: true,
            }
        );
        _panel.onDidDispose(() => {
            _panel = undefined;
            _panelState = undefined;
        });
        _panel.webview.onDidReceiveMessage((message) => {
            handlePanelMessage(message).catch((e) => {
                log(`REMOTE PROXY: panel message failed: ${e.message}`);
            });
        });
        _panel.webview.html = makeLoadingHtml('Loading ArrayView...');
    }

    const html = await loadViewerHtml(_panelState);
    if (_panel && _panelState) {
        _panel.webview.html = html;
        log(`REMOTE: real viewer panel ready ${_panelState.viewerUrl}`);
    }
    return _panel;
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
            log(`REMOTE: opening real viewer panel for ${url}`);
            try {
                await createRealViewerPanel(url);
                log('REMOTE: real viewer panel opened');
            } catch (e) {
                log(`REMOTE: real viewer panel FAILED: ${e.message}`);
                if (_panel) {
                    _panel.webview.html = makeLoadingHtml(`ArrayView failed to load: ${e.message}`);
                }
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

    tryOpenSignalFile();

    const interval = setInterval(() => tryOpenSignalFile(), 1000);
    context.subscriptions.push({ dispose: () => clearInterval(interval) });

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
