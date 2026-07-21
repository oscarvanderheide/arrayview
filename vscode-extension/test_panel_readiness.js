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

function createPanelRuntime(url) {
    const html = __test._viewerPanelHtml(url);
    const scriptMatch = html.match(/<script nonce="[^"]+">([\s\S]*)<\/script>/);
    assert(scriptMatch, 'panel wrapper script must be present');

    const messages = [];
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
        acquireVsCodeApi: () => ({ postMessage(message) { messages.push(message); } }),
        clearTimeout(id) { timers.delete(id); },
        console: { log() {} },
        document: { getElementById(id) { return elements[id]; } },
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
    return { frameHandlers, messages, srcWrites, timers, windowHandlers };
}

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

    console.log('panel readiness tests passed');
} finally {
    if (originalHome === undefined) delete process.env.HOME;
    else process.env.HOME = originalHome;
    fs.rmSync(tempHome, { recursive: true, force: true });
}
