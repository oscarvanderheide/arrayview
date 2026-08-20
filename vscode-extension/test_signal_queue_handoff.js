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
let externalOpenResolve = null;
let externalOpenCount = 0;
let integratedBrowserAvailable = false;
let remoteProxyEnabled = false;
const browserCommandArgs = [];
const editorTabs = [];
const progressNotifications = [];

class TabInputWebview {}

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
    ProgressLocation: { Notification: 15, Window: 10 },
    env: {
        remoteName: null,
        uiKind: 1,
        openExternal() {
            externalOpenCount += 1;
            return new Promise(resolve => {
                externalOpenResolve = () => resolve(true);
            });
        },
    },
    UIKind: { Web: 2 },
    TabInputWebview,
    ViewColumn: { Active: 1, Beside: 2 },
    Uri: { parse: v => v, file: v => ({ fsPath: v, toString: () => `file://${v}` }) },
    window: {
        withProgress(options, task) {
            const notification = {
                options,
                reports: [],
                completed: false,
            };
            progressNotifications.push(notification);
            const result = Promise.resolve(task({
                report(update) { notification.reports.push(update); },
            })).then(() => { notification.completed = true; });
            notification.task = result;
            return result;
        },
        state: { focused: true },
        activeTextEditor: null,
        tabGroups: {
            all: [{ tabs: editorTabs }],
            async close(tab) {
                const index = editorTabs.indexOf(tab);
                if (index < 0) return false;
                editorTabs.splice(index, 1);
                return true;
            },
        },
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
        getConfiguration() {
            return {
                get(key, fallback) {
                    return key === 'enableRemoteProxy' ? remoteProxyEnabled : fallback;
                },
            };
        },
    },
    commands: {
        registerCommand() { return { dispose() {} }; },
        async getCommands() {
            return integratedBrowserAvailable
                ? ['workbench.action.browser.open']
                : [];
        },
        async executeCommand(command, args) {
            if (command === 'workbench.action.browser.open') {
                browserCommandArgs.push(args);
                editorTabs.push({
                    label: 'Integrated Browser',
                    input: new TabInputWebview(),
                    isActive: true,
                });
            }
            return undefined;
        },
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
    const externalJournals = new Map();
    const server = http.createServer((req, res) => {
        const parsed = new URL(req.url, 'http://localhost');
        if (parsed.pathname.startsWith('/viewer-phase/')) {
            const parts = parsed.pathname.split('/').filter(Boolean);
            const sid = decodeURIComponent(parts[1]);
            const requestId = decodeURIComponent(parts[2]);
            if (req.method === 'POST') {
                let raw = '';
                req.on('data', chunk => { raw += chunk; });
                req.on('end', () => {
                    const body = JSON.parse(raw);
                    externalJournals.set(requestId, { sid, ...body });
                    res.writeHead(200, { 'content-type': 'application/json' });
                    res.end(JSON.stringify({
                        ...body,
                        sid,
                        request_id: requestId,
                    }));
                });
                return;
            }
            const journal = externalJournals.get(requestId);
            res.writeHead(journal ? 200 : 404, { 'content-type': 'application/json' });
            res.end(JSON.stringify(journal ? {
                sid,
                request_id: requestId,
                server_id: journal.server_id,
                window_id: journal.window_id,
                token: journal.token,
                phases: ['script-loaded', 'ws-open', 'metadata-loaded', 'frame-rendered'],
                viewer_instance_ids: ['external-viewer'],
            } : {}));
            return;
        }
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

        // --- a hung system-browser handoff cannot starve tab launches ------
        vscodeMock.env.remoteName = 'tunnel';
        vscodeMock.env.appHost = 'desktop';
        remoteProxyEnabled = true;
        integratedBrowserAvailable = true;
        const external = requestData(port, 'req-external', 'external.npy');
        external.displaySurface = 'external-browser';
        assert.strictEqual(__test.claimProtocolRequest(external), 'acquired');
        const externalDone = __test.processSignalData(external);
        await waitFor(() => externalOpenCount === 1, 'external-browser handoff to start');
        const releasedWhileExternalHung = !__test.isSignalQueueBusy();
        if (!releasedWhileExternalHung) {
            externalOpenResolve();
            await externalDone;
        }
        assert.strictEqual(
            releasedWhileExternalHung,
            true,
            'a pending external-browser API call must not own the signal queue'
        );

        const afterExternal = requestData(port, 'req-after-external', 'after.npy');
        assert.strictEqual(__test.claimProtocolRequest(afterExternal), 'acquired');
        const afterExternalDone = __test.processSignalData(afterExternal);
        await waitFor(
            () => ackState(afterExternal) === 'panel_opened',
            'a tab launch behind the pending external-browser handoff'
        );
        assert.strictEqual(
            progressNotifications.length,
            1,
            'one integrated-browser request must create one progress notification'
        );
        assert.deepStrictEqual(progressNotifications[0].options, {
            location: vscodeMock.ProgressLocation.Notification,
            title: 'Opening after.npy in ArrayView…',
            cancellable: false,
        });
        assert.strictEqual(
            browserCommandArgs.length,
            1,
            'the later launch must use the Tunnel integrated browser while the external call is pending'
        );
        await afterExternalDone;
        assert.strictEqual(ackState(afterExternal), 'backend_ready');
        await progressNotifications[0].task;
        assert.strictEqual(
            progressNotifications[0].completed,
            true,
            'the notification must close after the first rendered frame'
        );
        assert.deepStrictEqual(
            progressNotifications[0].reports,
            [],
            'a first-navigation success must not claim VS Code is still connecting'
        );
        assert.strictEqual(editorTabs.length, 1, 'the later launch gets its own browser tab');
        assert.strictEqual(
            externalOpenCount,
            1,
            'the later Tunnel launch must not fall back to the system browser'
        );

        externalOpenResolve();
        await externalDone;
        assert.strictEqual(ackState(external), 'backend_ready');

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
