// A render failure must settle the request immediately.
//
// The opener serialises on isProcessingSignal for the whole request, so a
// panel that waits out its full timeout does not just fail one file — it holds
// every later click behind it. Observed 2026-07-27: a 1-D array the backend
// could not draw produced 45s of "SKIP: isProcessingSignal=true" before
// failing, and the next clicks were dead in the meantime.
//
// Once the backend has said it cannot draw the array, the answer is known.
// Retrying repeats the same failure, so the panel must report it upward
// instead of sitting on the timeout.

const assert = require('assert');
const fs = require('fs');
const Module = require('module');
const os = require('os');
const path = require('path');
const vm = require('vm');

const tempHome = fs.mkdtempSync(path.join(os.tmpdir(), 'arrayview-render-fail-'));
const originalHome = process.env.HOME;
process.env.HOME = tempHome;

const vscodeMock = {
    env: { remoteName: null, uiKind: 1 },
    UIKind: { Web: 2 },
    ViewColumn: { Active: 1, Beside: 2 },
};

const originalLoad = Module._load;
Module._load = function (request, parent, isMain) {
    if (request === 'vscode') return vscodeMock;
    return originalLoad.call(this, request, parent, isMain);
};
const { __test } = require('./extension');
Module._load = originalLoad;

function panelRuntime(url) {
    const html = __test._viewerPanelHtml(url);
    const script = html.match(/<script nonce="[^"]+">([\s\S]*)<\/script>/)[1];
    const messages = [];
    const srcWrites = [];
    const timers = new Map();
    const windowHandlers = {};
    const frameHandlers = {};
    let next = 1;
    const schedule = (handler) => { const id = next++; timers.set(id, handler); return id; };
    const context = {
        AbortController: class { constructor() { this.signal = {}; } abort() {} },
        acquireVsCodeApi: () => ({ postMessage: m => messages.push(m) }),
        clearTimeout: id => timers.delete(id),
        console: { log() {} },
        document: {
            getElementById: id => ({
                f: { style: {}, addEventListener: (t, h) => { frameHandlers[t] = h; }, set src(v) { srcWrites.push(v); } },
            }[id] || { textContent: '', classList: { add() {} }, style: {} }),
        },
        fetch: () => Promise.resolve({}),
        setTimeout: schedule,
        window: {
            addEventListener: (t, h) => { windowHandlers[t] = h; },
            clearTimeout: id => timers.delete(id),
            setTimeout: schedule,
        },
    };
    vm.runInNewContext(script, context);
    return { messages, srcWrites, timers, windowHandlers, frameHandlers };
}

// Minimal stand-in for a VS Code webview panel.
function fakePanel() {
    let messageHandler = null;
    let disposeHandler = null;
    return {
        emit(message) { if (messageHandler) messageHandler(message); },
        dispose() { if (disposeHandler) disposeHandler(); },
        webview: {
            onDidReceiveMessage(handler) {
                messageHandler = handler;
                return { dispose() {} };
            },
        },
        onDidDispose(handler) {
            disposeHandler = handler;
            return { dispose() {} };
        },
    };
}

(async () => {
    try {
        // --- the wrapper turns a render error into a terminal verdict --------
        const runtime = panelRuntime('http://localhost:8123/?sid=unrenderable');
        runtime.frameHandlers.load();
        assert.strictEqual(runtime.timers.size, 1, 'a retry is pending before the verdict');

        runtime.windowHandlers.message({
            data: {
                source: 'arrayview-viewer',
                phase: 'render-error',
                detail: 'buffer is not large enough',
            },
        });

        const failure = runtime.messages.find(m => m.type === 'viewer-failed');
        assert(failure, 'the wrapper must report the failure to the extension');
        assert.strictEqual(failure.message, 'buffer is not large enough');
        assert.strictEqual(
            runtime.timers.size,
            0,
            'no reload may remain armed once the array is known to be undrawable'
        );
        const srcWritesAtVerdict = runtime.srcWrites.length;
        for (const handler of [...runtime.timers.values()]) handler();
        assert.strictEqual(
            runtime.srcWrites.length,
            srcWritesAtVerdict,
            'a failure that will repeat must not be retried'
        );

        // --- the opener settles at once rather than on the timeout ----------
        const panel = fakePanel();
        const startedAt = Date.now();
        const ready = __test.waitForViewerReady(panel, 45000);
        panel.emit({ type: 'viewer-failed', message: 'buffer is not large enough' });
        const error = await ready;
        const elapsedMs = Date.now() - startedAt;

        assert(error, 'an undrawable array must resolve as an error, not success');
        assert.strictEqual(error.message, 'buffer is not large enough');
        // The real incident spent the full 45s here with the queue locked.
        assert.ok(elapsedMs < 1000, `must settle immediately, took ${elapsedMs}ms`);

        // --- a healthy panel is unaffected ----------------------------------
        const healthy = fakePanel();
        const healthyReady = __test.waitForViewerReady(healthy, 45000);
        healthy.emit({ type: 'viewer-ready', phase: 'frame-rendered' });
        assert.strictEqual(
            await healthyReady,
            null,
            'a rendered frame must still resolve as success'
        );

        console.log('render failure fast-fail tests passed');
    } finally {
        if (originalHome === undefined) delete process.env.HOME;
        else process.env.HOME = originalHome;
        fs.rmSync(tempHome, { recursive: true, force: true });
    }
})().catch(error => {
    console.error(error);
    process.exitCode = 1;
});
