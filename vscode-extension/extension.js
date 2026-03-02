// arrayview-opener v0.1.3
const vscode = require('vscode');
const fs = require('fs');
const path = require('path');
const os = require('os');
const http = require('http');

const VERSION = '0.1.3';
const SIGNAL_DIR = path.join(os.homedir(), '.arrayview');
const SIGNAL_FILE = path.join(SIGNAL_DIR, 'open-request.json');
const LOG_FILE = path.join(SIGNAL_DIR, 'extension.log');

function log(msg) {
    const ts = new Date().toISOString();
    try { fs.appendFileSync(LOG_FILE, `${ts} [v${VERSION}] ${msg}\n`); } catch (_) {}
}

/**
 * Read the tunnel name & cluster from ~/.vscode/cli/code_tunnel.json.
 * Returns { name, cluster } or null.
 */
function readTunnelConfig() {
    const candidates = [
        path.join(os.homedir(), '.vscode', 'cli', 'code_tunnel.json'),
        path.join(os.homedir(), '.vscode-server', 'cli', 'code_tunnel.json'),
    ];
    for (const p of candidates) {
        try {
            const cfg = JSON.parse(fs.readFileSync(p, 'utf8'));
            if (cfg.name && cfg.cluster) return cfg;
        } catch (_) {}
    }
    return null;
}

// Keep tunnel objects alive for the lifetime of the extension.
// If we dispose them, port forwarding is torn down immediately.
const _activeTunnels = new Map(); // port -> Tunnel

/**
 * Ensure port is forwarded through the dev tunnel.
 * Returns the tunnel object, or null if it fails.
 */
async function ensureTunnel(port) {
    if (_activeTunnels.has(port)) {
        return _activeTunnels.get(port);
    }

    try {
        log(`openTunnel(${port}): calling...`);
        const tunnel = await Promise.race([
            vscode.workspace.openTunnel({
                remoteAddress: { port: port, host: 'localhost' },
                localAddressPort: port,
            }),
            new Promise((_, rej) => setTimeout(() => rej(new Error('timeout 30s')), 30000))
        ]);
        log(`openTunnel(${port}): success! localAddress=${JSON.stringify(tunnel.localAddress)}`);
        // DO NOT dispose! Keep the tunnel alive.
        _activeTunnels.set(port, tunnel);
        tunnel.onDidDispose(() => {
            log(`tunnel(${port}) disposed by VS Code`);
            _activeTunnels.delete(port);
        });
        return tunnel;
    } catch (e) {
        log(`openTunnel(${port}) failed: ${e.message}`);
        return null;
    }
}

/**
 * Open an arrayview URL in the user's browser or VS Code tab.
 *
 * Strategy (tried in order):
 *  1. openTunnel() to register port forwarding — keep tunnel alive
 *  2. asExternalUri to get the forwarded URL
 *  3. Construct devtunnel URL from code_tunnel.json as fallback
 *  4. Open via simpleBrowser.show and openExternal
 */
async function openUrl(url) {
    log(`openUrl: ${url}`);

    let port = 0;
    let parsed;
    try {
        parsed = new URL(url);
        port = parseInt(parsed.port, 10) || 80;
    } catch (_) {}

    // Extract title
    let title = 'ArrayView';
    try {
        const name = parsed.searchParams.get('init_name');
        if (name) title = `ArrayView: ${name}`;
    } catch (_) {}

    // Step 1: Ensure port is forwarded through the tunnel
    let tunnel = null;
    if (port) {
        tunnel = await ensureTunnel(port);
    }

    // Step 2: Try asExternalUri (should work now that tunnel is registered)
    let resolvedUrl = null;
    try {
        const baseStr = `${parsed.protocol}//${parsed.host}`;
        const resolved = await Promise.race([
            vscode.env.asExternalUri(vscode.Uri.parse(baseStr)),
            new Promise((_, rej) => setTimeout(() => rej(new Error('timeout 8s')), 8000))
        ]);
        const resolvedStr = resolved.toString(true).replace(/\/$/, '');
        log(`asExternalUri: ${baseStr} -> ${resolvedStr}`);
        if (resolvedStr !== baseStr) {
            resolvedUrl = resolvedStr + parsed.pathname + parsed.search;
            log(`Resolved URL: ${resolvedUrl}`);
        }
    } catch (e) {
        log(`asExternalUri: ${e.message}`);
    }

    // Step 3: Construct devtunnel URL from code_tunnel.json as fallback
    if (!resolvedUrl) {
        const cfg = readTunnelConfig();
        if (cfg && port) {
            const devtunnelUrl = `https://${cfg.name}-${port}.${cfg.cluster}.devtunnels.ms`;
            resolvedUrl = devtunnelUrl + parsed.pathname + parsed.search;
            log(`Constructed devtunnel URL from config: ${resolvedUrl}`);
        }
    }

    // Step 4: Open the URL
    const targetUrl = resolvedUrl || url;

    // Try simpleBrowser first (in-VS-Code experience)
    try {
        await vscode.commands.executeCommand('simpleBrowser.show', targetUrl);
        log(`simpleBrowser.show OK: ${targetUrl}`);
    } catch (e) {
        log(`simpleBrowser.show failed: ${e.message}`);
    }

    // Also try openExternal (real browser) as backup — especially useful first time
    // when devtunnel auth dialog may appear
    if (resolvedUrl) {
        try {
            const ok = await vscode.env.openExternal(vscode.Uri.parse(resolvedUrl));
            log(`openExternal(${resolvedUrl}): ${ok}`);
        } catch (e) {
            log(`openExternal failed: ${e.message}`);
        }
    }
}

let _busy = false;

async function handleSignalFile() {
    if (_busy) return;
    _busy = true;
    try {
        let content;
        try {
            content = fs.readFileSync(SIGNAL_FILE, 'utf8');
            fs.unlinkSync(SIGNAL_FILE);
        } catch (_) {
            return;
        }

        let request;
        try {
            request = JSON.parse(content);
        } catch (e) {
            log(`Bad JSON in signal file: ${e.message}`);
            return;
        }

        if (!request.url) {
            log('Signal file has no url field');
            return;
        }

        log(`Signal file consumed: ${request.url}`);
        await openUrl(request.url);
    } finally {
        _busy = false;
    }
}

function activate(context) {
    log(`activate (remoteName=${vscode.env.remoteName}, appHost=${vscode.env.appHost})`);

    const cfg = readTunnelConfig();
    log(`tunnelConfig: ${JSON.stringify(cfg)}`);

    try { fs.mkdirSync(SIGNAL_DIR, { recursive: true }); } catch (_) {}

    // Process existing signal file if fresh (< 30s old)
    if (fs.existsSync(SIGNAL_FILE)) {
        try {
            const age = Date.now() - fs.statSync(SIGNAL_FILE).mtimeMs;
            if (age < 30000) {
                log('Found fresh signal file at startup');
                handleSignalFile();
            } else {
                log('Deleted stale signal file at startup');
                try { fs.unlinkSync(SIGNAL_FILE); } catch (_) {}
            }
        } catch (_) {}
    }

    // Watch for new signal files
    try {
        const watcher = fs.watch(SIGNAL_DIR, (_, filename) => {
            if (filename === 'open-request.json' || filename === null) {
                setTimeout(() => {
                    if (fs.existsSync(SIGNAL_FILE)) handleSignalFile();
                }, 150);
            }
        });
        context.subscriptions.push({ dispose: () => watcher.close() });
        log('fs.watch OK');
    } catch (e) {
        log(`fs.watch failed: ${e.message}`);
    }

    // Polling fallback (every 1s)
    const poll = setInterval(() => {
        if (fs.existsSync(SIGNAL_FILE)) handleSignalFile();
    }, 1000);
    context.subscriptions.push({ dispose: () => clearInterval(poll) });

    // URI handler (works for local / SSH setups where code --open-url works)
    context.subscriptions.push(vscode.window.registerUriHandler({
        async handleUri(uri) {
            log(`URI handler: ${uri.toString()}`);
            const url = new URLSearchParams(uri.query).get('url');
            if (url) await openUrl(url);
        }
    }));

    // Clean up tunnels on deactivation
    context.subscriptions.push({
        dispose: () => {
            for (const [port, tunnel] of _activeTunnels) {
                log(`Disposing tunnel for port ${port}`);
                try { tunnel.dispose(); } catch (_) {}
            }
            _activeTunnels.clear();
        }
    });

    log('Setup complete');
}

function deactivate() {
    log('deactivate');
}

module.exports = { activate, deactivate };
