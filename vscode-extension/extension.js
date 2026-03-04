// arrayview-opener v0.2.0
const vscode = require('vscode');
const fs = require('fs');
const path = require('path');
const os = require('os');

const VERSION = '0.2.0';
const SIGNAL_DIR = path.join(os.homedir(), '.arrayview');
const SIGNAL_FILE = path.join(SIGNAL_DIR, 'open-request.json');
const LOG_FILE = path.join(SIGNAL_DIR, 'extension.log');

// Detect if this extension instance runs inside a VS Code *server* (tunnel/SSH).
// Server-side extensions live under ~/.vscode-server/ or ~/.vscode/cli/servers/.
// Local desktop VS Code extensions live under ~/.vscode/extensions/.
const IS_REMOTE_SERVER = __dirname.includes('.vscode-server')
    || __dirname.includes(path.join('.vscode', 'cli', 'servers'));

function log(msg) {
    const ts = new Date().toISOString();
    try { fs.appendFileSync(LOG_FILE, `${ts} [v${VERSION}] ${msg}\n`); } catch (_) {}
}

/**
 * Open an arrayview URL in VS Code's Simple Browser.
 *
 * Simple Browser webviews automatically forward localhost ports through
 * tunnels/SSH — no need for asExternalUri, openTunnel, or devtunnel URLs.
 */
async function openUrl(url) {
    log(`openUrl: ${url}`);

    try {
        await vscode.commands.executeCommand('simpleBrowser.show', url);
        log(`simpleBrowser.show OK: ${url}`);
    } catch (e) {
        log(`simpleBrowser.show failed: ${e.message}`);
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
        } catch (_) {
            return;
        }

        let request;
        try {
            request = JSON.parse(content);
        } catch (e) {
            log(`Bad JSON in signal file: ${e.message}`);
            try { fs.unlinkSync(SIGNAL_FILE); } catch (_) {}
            return;
        }

        if (!request.url) {
            log('Signal file has no url field');
            try { fs.unlinkSync(SIGNAL_FILE); } catch (_) {}
            return;
        }

        // --- Targeting ---
        // The signal file may contain "remote": true/false to indicate which
        // kind of extension host should consume it.
        //   remote=true  → only a server-side extension (tunnel/SSH) should consume
        //   remote=false → only a local desktop extension should consume
        //   absent       → any extension may consume
        if (request.remote === true && !IS_REMOTE_SERVER) {
            log(`Signal targets remote but I'm local (${__dirname}) — leaving for remote instance`);
            return;  // don't delete the file; let the server-side extension pick it up
        }
        if (request.remote === false && IS_REMOTE_SERVER) {
            log(`Signal targets local but I'm remote (${__dirname}) — leaving for local instance`);
            return;
        }

        // Consume the signal file
        try { fs.unlinkSync(SIGNAL_FILE); } catch (_) {}
        log(`Signal file consumed: ${request.url} (remote=${request.remote}, isRemoteServer=${IS_REMOTE_SERVER})`);
        await openUrl(request.url);
    } finally {
        _busy = false;
    }
}

function activate(context) {
    log(`activate (remoteName=${vscode.env.remoteName}, appHost=${vscode.env.appHost}, uiKind=${vscode.env.uiKind}, isRemoteServer=${IS_REMOTE_SERVER}, __dirname=${__dirname})`);

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

    // Polling fallback (every 2s)
    const poll = setInterval(() => {
        if (fs.existsSync(SIGNAL_FILE)) handleSignalFile();
    }, 2000);
    context.subscriptions.push({ dispose: () => clearInterval(poll) });

    // URI handler (works for local / SSH setups where code --open-url works)
    context.subscriptions.push(vscode.window.registerUriHandler({
        async handleUri(uri) {
            log(`URI handler: ${uri.toString()}`);
            const url = new URLSearchParams(uri.query).get('url');
            if (url) await openUrl(url);
        }
    }));

    log('Setup complete');
}

function deactivate() {
    log('deactivate');
}

module.exports = { activate, deactivate };
