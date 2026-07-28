// Regression coverage for relay document navigation black holes.
//
// A devtunnel connection either answers promptly or can remain silent forever.
// The viewer document is large enough that cancelling one request on a short
// timeout is unsafe, so tunnel panels start staggered fresh navigations and
// keep the first iframe whose own viewer script reports readiness.

const assert = require('assert');
const fs = require('fs');
const Module = require('module');
const os = require('os');
const path = require('path');
const vm = require('vm');

const tempHome = fs.mkdtempSync(path.join(os.tmpdir(), 'arrayview-nav-hedge-'));
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

function createPanelRuntime(url, hedgeNavigation) {
    const html = __test._viewerPanelHtml(url, hedgeNavigation);
    const scriptMatch = html.match(/<script nonce="[^"]+">([\s\S]*)<\/script>/);
    assert(scriptMatch, 'panel wrapper script must be present');

    const messages = [];
    const timers = new Map();
    const scheduled = [];
    const windowHandlers = {};
    const frames = [];
    let nextTimer = 1;

    function schedule(handler, delay) {
        const id = nextTimer++;
        timers.set(id, handler);
        scheduled.push({ id, delay });
        return id;
    }

    const framesRoot = {
        appendChild(frame) {
            frames.push(frame);
            frame.parentNode = this;
        },
    };
    function createFrame() {
        const handlers = {};
        const contentWindow = {};
        return {
            allow: '',
            contentWindow,
            handlers,
            removed: false,
            style: {},
            addEventListener(type, handler) { handlers[type] = handler; },
            remove() { this.removed = true; },
            set src(value) { this.srcValue = value; },
        };
    }
    const elements = {
        frames: framesRoot,
        'backend-url': { textContent: '' },
        'backend-error': { classList: { add() {} } },
    };
    const context = {
        URL,
        acquireVsCodeApi: () => ({
            postMessage(message) { messages.push(message); },
        }),
        clearTimeout(id) { timers.delete(id); },
        console: { log() {} },
        document: {
            createElement(tag) {
                assert.strictEqual(tag, 'iframe');
                return createFrame();
            },
            getElementById(id) { return elements[id]; },
        },
        setTimeout: schedule,
        window: {
            addEventListener(type, handler) { windowHandlers[type] = handler; },
        },
    };
    vm.runInNewContext(scriptMatch[1], context);
    return { frames, messages, scheduled, timers, windowHandlers };
}

function fireDelay(runtime, delay) {
    const entry = runtime.scheduled.find(
        item => item.delay === delay && runtime.timers.has(item.id)
    );
    assert(entry, `expected a live ${delay}ms timer`);
    const handler = runtime.timers.get(entry.id);
    runtime.timers.delete(entry.id);
    handler();
}

function viewerMessage(runtime, frame, phase) {
    runtime.windowHandlers.message({
        source: frame.contentWindow,
        data: { source: 'arrayview-viewer', phase },
    });
}

try {
    // Ordinary local/SSH panels keep one navigation and the existing bounded
    // retry. Hedging is tunnel-only because only the relay has the measured
    // silent-drop mode.
    const local = createPanelRuntime(
        'http://localhost:8123/?sid=local',
        false
    );
    assert.strictEqual(local.frames.length, 1);
    assert.deepStrictEqual(
        local.scheduled.map(entry => entry.delay),
        [8000]
    );
    fireDelay(local, 8000);
    assert.strictEqual(local.frames.length, 2, 'a stalled local navigation retries');
    assert.strictEqual(local.frames[0].removed, true);

    // Tunnel panels overlap three fresh document connections per bounded wave.
    // Query markers force distinct requests while preserving launch parameters.
    const tunnel = createPanelRuntime(
        'https://relay.example/?sid=abc&compare_sid=def',
        true
    );
    assert.strictEqual(tunnel.frames.length, 1);
    assert.strictEqual(
        new URL(tunnel.frames[0].srcValue).searchParams.get('_av_relay_hedge'),
        '1'
    );
    fireDelay(tunnel, 700);
    fireDelay(tunnel, 1400);
    assert.strictEqual(tunnel.frames.length, 3);
    assert.match(tunnel.frames[1].srcValue, /_av_nav_hedge=2/);
    assert.match(tunnel.frames[2].srcValue, /_av_nav_hedge=3/);
    assert.match(tunnel.frames[2].srcValue, /compare_sid=def/);

    // The first viewer-owned marker, not iframe load, chooses the winner.
    // Error/interstitial documents can emit load and must not win.
    tunnel.frames[0].handlers.load();
    assert.strictEqual(
        tunnel.messages.some(message => message.phase === 'navigation-winner'),
        false
    );
    viewerMessage(tunnel, tunnel.frames[1], 'script-loaded');
    assert.strictEqual(tunnel.frames[0].removed, true);
    assert.strictEqual(tunnel.frames[1].removed, false);
    assert.strictEqual(tunnel.frames[1].style.visibility, 'visible');
    assert.strictEqual(tunnel.frames[2].removed, true);
    assert.strictEqual(tunnel.timers.size, 0, 'a winner cancels every hedge/watchdog');
    assert.ok(tunnel.messages.some(message =>
        message.phase === 'navigation-winner' && message.attempt === 2
    ));

    // A loser that finishes later cannot report readiness for this panel.
    const beforeLoser = tunnel.messages.length;
    viewerMessage(tunnel, tunnel.frames[2], 'frame-rendered');
    assert.strictEqual(tunnel.messages.length, beforeLoser);
    viewerMessage(tunnel, tunnel.frames[1], 'frame-rendered');
    assert.ok(tunnel.messages.some(message =>
        message.type === 'viewer-ready' && message.phase === 'frame-rendered'
    ));

    // Iframe load is diagnostic only. In a real webview, appending a src-less
    // frame first emitted an initial about:blank load before navigation-attempt.
    // It must neither win nor alter the continuous hedge schedule.
    const blankLoad = createPanelRuntime(
        'https://relay.example/?sid=blank-load',
        true
    );
    blankLoad.frames[0].handlers.load();
    assert.strictEqual(blankLoad.timers.size, 3);
    fireDelay(blankLoad, 700);
    assert.strictEqual(blankLoad.frames.length, 2);
    viewerMessage(blankLoad, blankLoad.frames[1], 'script-loaded');
    assert.strictEqual(
        blankLoad.timers.size,
        0,
        'a viewer-owned marker cancels the remaining stream and watchdog'
    );

    // Three fully silent waves terminate and retire every iframe; swallowed
    // navigations cannot leave hidden documents or network work behind forever.
    const exhausted = createPanelRuntime(
        'https://relay.example/?sid=exhausted',
        true
    );
    for (let wave = 0; wave < 3; wave++) {
        fireDelay(exhausted, 700);
        fireDelay(exhausted, 1400);
        fireDelay(exhausted, 10000);
    }
    assert.strictEqual(exhausted.frames.length, 9);
    assert.ok(exhausted.frames.every(frame => frame.removed));
    assert.strictEqual(exhausted.timers.size, 0);

    console.log('panel navigation hedge tests passed');
} finally {
    if (originalHome === undefined) delete process.env.HOME;
    else process.env.HOME = originalHome;
    fs.rmSync(tempHome, { recursive: true, force: true });
}
