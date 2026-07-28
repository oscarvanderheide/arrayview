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
Module._load = function (request, parent, isMain) {
    if (request === 'vscode') return vscodeMock;
    return originalLoad.call(this, request, parent, isMain);
};
const { __test } = require('./extension');
Module._load = originalLoad;

function createPanelRuntime(url) {
    const html = __test._viewerPanelHtml(url, false);
    const script = html.match(/<script nonce="[^"]+">([\s\S]*)<\/script>/)[1];
    const messages = [];
    const timers = new Map();
    const windowHandlers = {};
    const frames = [];
    let nextTimer = 1;
    const root = { appendChild(frame) { frames.push(frame); } };
    const context = {
        URL,
        acquireVsCodeApi: () => ({ postMessage: message => messages.push(message) }),
        clearTimeout: id => timers.delete(id),
        console: { log() {} },
        document: {
            createElement() {
                const handlers = {};
                return {
                    contentWindow: {},
                    handlers,
                    style: {},
                    addEventListener: (type, handler) => { handlers[type] = handler; },
                    remove() {},
                    set src(value) { this.srcValue = value; },
                };
            },
            getElementById: id => ({
                frames: root,
                'backend-url': { textContent: '' },
                'backend-error': { classList: { add() {} } },
            }[id]),
        },
        setTimeout(handler) {
            const id = nextTimer++;
            timers.set(id, handler);
            return id;
        },
        window: {
            addEventListener: (type, handler) => { windowHandlers[type] = handler; },
        },
    };
    vm.runInNewContext(script, context);
    return { frames, messages, timers, windowHandlers };
}

function emitViewer(runtime, phase, detail = null) {
    runtime.windowHandlers.message({
        source: runtime.frames[0].contentWindow,
        data: { source: 'arrayview-viewer', phase, detail },
    });
}

try {
    const runtime = createPanelRuntime('http://localhost:8123/?sid=ready');
    assert.strictEqual(runtime.frames.length, 1);
    assert.strictEqual(runtime.frames[0].srcValue,
        'http://localhost:8123/?sid=ready');
    runtime.frames[0].handlers.load();
    assert.ok(runtime.messages.some(message =>
        message.type === 'panel-phase' && message.phase === 'iframe-loaded'
    ));

    emitViewer(runtime, 'script-loaded');
    assert.strictEqual(
        runtime.timers.size,
        0,
        'script-loaded chooses the iframe and cancels navigation retries'
    );
    emitViewer(runtime, 'frame-rendered');
    assert.ok(runtime.messages.some(message =>
        message.type === 'viewer-ready' && message.phase === 'frame-rendered'
    ));

    console.log('panel readiness tests passed');
} finally {
    if (originalHome === undefined) delete process.env.HOME;
    else process.env.HOME = originalHome;
    fs.rmSync(tempHome, { recursive: true, force: true });
}
