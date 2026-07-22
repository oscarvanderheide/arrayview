const assert = require('assert');
const fs = require('fs');
const Module = require('module');
const os = require('os');
const path = require('path');
const vm = require('vm');

const tempHome = fs.mkdtempSync(path.join(os.tmpdir(), 'arrayview-panel-ready-'));
const originalHome = process.env.HOME;
process.env.HOME = tempHome;

const vscodeMock = {
    env: { remoteName: null, uiKind: 1 },
    UIKind: { Web: 2 },
    ViewColumn: { Active: 1, Beside: 2 },
};

const originalLoad = Module._load;
Module._load = function(request, parent, isMain) {
    if (request === 'vscode') return vscodeMock;
    return originalLoad.call(this, request, parent, isMain);
};
const { __test } = require('./extension');
Module._load = originalLoad;

function createPanelRuntime(url, warmupUrl = null, fetchError = null) {
    const html = __test._viewerPanelHtml(url, warmupUrl);
    const scriptMatch = html.match(/<script nonce="[^"]+">([\s\S]*)<\/script>/);
    assert(scriptMatch, 'panel wrapper script must be present');

    const messages = [];
    const fetchCalls = [];
    const srcWrites = [];
    const timers = new Map();
    const windowHandlers = {};
    const frameHandlers = {};
    let nextTimer = 1;

    const frame = {
        style: {},
        addEventListener(type, handler) { frameHandlers[type] = handler; },
        set src(value) { srcWrites.push(value); },
    };
    const elements = {
        f: frame,
        'backend-url': { textContent: '' },
        'backend-error': { classList: { add() {} } },
    };
    const context = {
        AbortController: class {
            constructor() { this.signal = {}; }
            abort() { this.signal.aborted = true; }
        },
        acquireVsCodeApi: () => ({ postMessage(message) { messages.push(message); } }),
        clearTimeout(id) { timers.delete(id); },
        console: { log() {} },
        document: { getElementById(id) { return elements[id]; } },
        fetch(...args) {
            fetchCalls.push(args);
            if (fetchError) return Promise.reject(fetchError);
            return Promise.resolve({});
        },
        setTimeout(handler) {
            const id = nextTimer++;
            timers.set(id, handler);
            return id;
        },
        window: {
            addEventListener(type, handler) { windowHandlers[type] = handler; },
            clearTimeout(id) { timers.delete(id); },
            setTimeout(handler) {
                const id = nextTimer++;
                timers.set(id, handler);
                return id;
            },
        },
    };
    vm.runInNewContext(scriptMatch[1], context);
    return { fetchCalls, frameHandlers, messages, srcWrites, timers, windowHandlers };
}

(async () => {
try {
    const handshaken = createPanelRuntime('http://localhost:8123/?sid=ready');
    assert.strictEqual(handshaken.srcWrites.length, 1);
    handshaken.frameHandlers.load();
    assert.strictEqual(handshaken.timers.size, 1);
    handshaken.windowHandlers.message({
        data: { source: 'arrayview-viewer', phase: 'script-loaded' },
    });
    assert.strictEqual(
        handshaken.timers.size,
        0,
        'script-loaded must cancel the reload that can interrupt first rendering'
    );
    assert.strictEqual(handshaken.srcWrites.length, 1);

    handshaken.windowHandlers.message({
        data: { source: 'arrayview-viewer', phase: 'frame-rendered' },
    });
    assert.deepStrictEqual(
        JSON.parse(JSON.stringify(handshaken.messages)),
        [
            { type: 'panel-phase', phase: 'wrapper-started' },
            { type: 'panel-phase', phase: 'iframe-loaded' },
            { type: 'viewer-phase', phase: 'script-loaded' },
            { type: 'viewer-phase', phase: 'frame-rendered' },
            { type: 'viewer-ready', phase: 'frame-rendered' },
        ]
    );

    const unresponsive = createPanelRuntime('http://localhost:8124/?sid=retry');
    unresponsive.frameHandlers.load();
    assert.strictEqual(unresponsive.timers.size, 1);
    const retry = [...unresponsive.timers.values()][0];
    retry();
    assert.strictEqual(
        unresponsive.srcWrites.length,
        2,
        'a document that never handshakes must still be retried'
    );

    const warmed = createPanelRuntime(
        'http://localhost:8125/?sid=warmed',
        'http://localhost:8125/ping'
    );
    assert.strictEqual(
        warmed.srcWrites.length,
        0,
        'the nested iframe must not navigate before transport warmup completes'
    );
    await new Promise(resolve => setImmediate(resolve));
    assert.strictEqual(warmed.fetchCalls.length, 1);
    assert.strictEqual(warmed.fetchCalls[0][0], 'http://localhost:8125/ping');
    assert.strictEqual(warmed.fetchCalls[0][1].mode, 'no-cors');
    assert.deepStrictEqual(
        JSON.parse(JSON.stringify(warmed.messages.slice(0, 3))),
        [
            { type: 'panel-phase', phase: 'wrapper-started' },
            { type: 'panel-phase', phase: 'transport-warmup-started' },
            { type: 'panel-phase', phase: 'transport-warmup-complete' },
        ]
    );
    assert.deepStrictEqual(warmed.srcWrites, [
        'http://localhost:8125/?sid=warmed',
    ]);

    const failedWarmup = createPanelRuntime(
        'http://localhost:8126/?sid=fallback',
        'http://localhost:8126/ping',
        new Error('forward unavailable')
    );
    await new Promise(resolve => setImmediate(resolve));
    assert.deepStrictEqual(failedWarmup.srcWrites, [
        'http://localhost:8126/?sid=fallback',
    ]);
    assert.deepStrictEqual(
        JSON.parse(JSON.stringify(failedWarmup.messages.slice(0, 3))),
        [
            { type: 'panel-phase', phase: 'wrapper-started' },
            { type: 'panel-phase', phase: 'transport-warmup-started' },
            {
                type: 'panel-phase',
                phase: 'transport-warmup-failed',
                message: 'forward unavailable',
            },
        ]
    );

    console.log('panel readiness tests passed');
} finally {
    if (originalHome === undefined) delete process.env.HOME;
    else process.env.HOME = originalHome;
    fs.rmSync(tempHome, { recursive: true, force: true });
}
})().catch(error => {
    console.error(error);
    process.exitCode = 1;
});
