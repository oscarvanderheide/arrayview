// The signal queue must be released once a panel is up, not held until the
// viewer renders.
//
// The opener processes one signal at a time. Until now that lock spanned the
// whole request, including the wait for the viewer's first frame — a network
// wait that legitimately runs for tens of seconds while a large array loads,
// and ran to the full 45s timeout whenever a load failed. Every later click
// sat behind it: the log for 2026-07-27T09:44 is 45 solid seconds of
// "SKIP: isProcessingSignal=true" while the user clicked other files and got
// nothing. That is what "one bad array broke the next few" was.
//
// Only the first half of a request touches shared state (route cache, pending
// placeholders, _openPanels). The readiness wait touches nothing but its own
// panel, so it must not hold the queue.

const assert = require('assert');
const fs = require('fs');
const http = require('http');
const Module = require('module');
const os = require('os');
const path = require('path');

const tempHome = fs.mkdtempSync(path.join(os.tmpdir(), 'arrayview-queue-handoff-'));
const originalHome = process.env.HOME;
process.env.HOME = tempHome;

const panels = [];

function fakePanel(title) {
    const panel = {
        title,
        disposed: false,
        __messageHandler: null,
        webview: {
            options: {},
            html: '',
            onDidReceiveMessage(handler) {
                panel.__messageHandler = handler;
                return { dispose() {} };
            },
            postMessage() { return Promise.resolve(true); },
        },
        onDidDispose() { return { dispose() {} }; },
        reveal() {},
        dispose() { panel.disposed = true; },
        // Drive the readiness handshake by hand so a request can be held
        // deliberately in its post-panel wait.
        renderFrame() {
            if (panel.__messageHandler) {
                panel.__messageHandler({ type: 'viewer-ready', phase: 'frame-rendered' });
            }
        },
    };
    return panel;
}

const vscodeMock = {
    env: { remoteName: null, uiKind: 1 },
    UIKind: { Web: 2 },
    ViewColumn: { Active: 1, Beside: 2 },
    Uri: { parse: v => v, file: v => ({ fsPath: v, toString: () => `file://${v}` }) },
    window: {
        state: { focused: true },
        activeTextEditor: null,
        showErrorMessage() {},
        createWebviewPanel(_type, title) {
            const panel = fakePanel(title);
            panels.push(panel);
            return panel;
        },
    },
    workspace: {
        workspaceFolders: [],
        getWorkspaceFolder() { return null; },
        getConfiguration() { return { get: (_k, fallback) => fallback }; },
    },
    commands: {
        registerCommand() { return { dispose() {} }; },
        async getCommands() { return []; },
        async executeCommand() { return undefined; },
    },
};

const originalLoad = Module._load;
Module._load = function (request, parent, isMain) {
    if (request === 'vscode') return vscodeMock;
    return originalLoad.call(this, request, parent, isMain);
};
const { __test } = require('./extension');
Module._load = originalLoad;

function requestData(port, requestId, title) {
    return {
        protocolVersion: 1,
        requestId,
        serverId: 'server-queue',
        windowId: 'window-queue',
        ackPath: path.join(__test.signalDir, `open-ack-v0100-${requestId}.json`),
        sentAtMs: Date.now(),
        maxAgeMs: 120000,
        url: `http://localhost:${port}/?sid=${requestId}`,
        title,
    };
}

function ackState(data) {
    try {
        return JSON.parse(fs.readFileSync(data.ackPath, 'utf8')).state;
    } catch (_) {
        return null;
    }
}

async function waitFor(predicate, what, timeoutMs = 5000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        if (predicate()) return;
        await new Promise(resolve => setTimeout(resolve, 10));
    }
    throw new Error(`timed out waiting for ${what}`);
}

(async () => {
    const server = http.createServer((req, res) => {
        const parsed = new URL(req.url, 'http://localhost');
        if (parsed.pathname.startsWith('/metadata/')) {
            res.writeHead(200, { 'content-type': 'application/json' });
            res.end(JSON.stringify({ ready: true }));
            return;
        }
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ service: 'arrayview', instance_id: 'server-queue' }));
    });

    try {
        fs.mkdirSync(__test.signalDir, { recursive: true });
        __test.setWindowId('window-queue');
        fs.writeFileSync(
            path.join(__test.signalDir, 'window-window-queue.json'),
            JSON.stringify({
                pid: process.pid,
                windowId: 'window-queue',
                extensionInstanceId: __test.extensionInstanceId,
            })
        );
        await new Promise(resolve => server.listen(0, 'localhost', resolve));
        const port = server.address().port;

        assert.strictEqual(__test.isSignalQueueBusy(), false, 'queue starts free');

        // --- request A: panel opens, viewer never renders -------------------
        const slow = requestData(port, 'req-slow', 'slow.npy');
        assert.strictEqual(__test.claimProtocolRequest(slow), 'acquired');
        const slowDone = __test.processSignalData(slow);

        await waitFor(() => ackState(slow) === 'panel_opened', "A's panel to open");
        assert.strictEqual(panels.length, 1, 'A opened exactly one panel');

        // The claim of this whole change.
        assert.strictEqual(
            __test.isSignalQueueBusy(),
            false,
            'the queue must be free while A waits for its first frame'
        );
        assert.strictEqual(
            ackState(slow),
            'panel_opened',
            'A must still be in flight, not settled'
        );

        // --- request B proceeds while A is still waiting --------------------
        const second = requestData(port, 'req-second', 'second.npy');
        assert.strictEqual(__test.claimProtocolRequest(second), 'acquired');
        const secondDone = __test.processSignalData(second);

        await waitFor(() => ackState(second) === 'panel_opened', "B's panel to open");
        assert.strictEqual(
            panels.length,
            2,
            'B must get its own panel while A is unfinished'
        );

        // --- B settling must not disturb A, and vice versa ------------------
        panels[1].renderFrame();
        await secondDone;
        assert.strictEqual(ackState(second), 'backend_ready', 'B completed');
        assert.strictEqual(
            ackState(slow),
            'panel_opened',
            "B completing must not settle A's request"
        );

        // A finishing later must not release a queue slot it no longer owns.
        const third = requestData(port, 'req-third', 'third.npy');
        assert.strictEqual(__test.claimProtocolRequest(third), 'acquired');
        const thirdStarted = __test.processSignalData(third);
        panels[0].renderFrame();
        await slowDone;
        assert.strictEqual(ackState(slow), 'backend_ready', 'A completed');

        await waitFor(() => ackState(third) === 'panel_opened', "C's panel to open");
        panels[2].renderFrame();
        await thirdStarted;
        assert.strictEqual(ackState(third), 'backend_ready', 'C completed');

        assert.strictEqual(
            __test.isSignalQueueBusy(),
            false,
            'the queue must be free once every request has settled'
        );

        console.log('signal queue handoff tests passed');
    } finally {
        await new Promise(resolve => server.close(resolve));
        if (originalHome === undefined) delete process.env.HOME;
        else process.env.HOME = originalHome;
        fs.rmSync(tempHome, { recursive: true, force: true });
    }
})().catch(error => {
    console.error(error);
    process.exitCode = 1;
});
