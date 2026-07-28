// Regression coverage for a navigation that never completes.
//
// Observed 2026-07-27T09:25:21Z (~/.arrayview/extension.log): a viewer panel
// opened while three viewers were already streaming through the same tunnel.
// The warmup fetch to /ping succeeded, the panel posted transport-warmup-
// complete, and then the iframe's request for the viewer page simply never
// came back. A stalled request fires neither 'load' nor 'error', so every
// handler in the wrapper stayed silent and the tab sat blank for 45s until the
// outer request timeout failed it. The backend was healthy throughout — it
// served that exact sid in 30ms over loopback.
//
// The wrapper already had a retry loop, but it was armed only inside the
// 'load' handler, so it covered "page arrived, viewer never booted" and could
// not cover "page never arrived". These cases assert the watchdog is armed at
// navigation time instead.

const assert = require('assert');
const fs = require('fs');
const Module = require('module');
const os = require('os');
const path = require('path');
const vm = require('vm');

const tempHome = fs.mkdtempSync(path.join(os.tmpdir(), 'arrayview-nav-watchdog-'));
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

// Same shape as test_panel_readiness.js, plus the scheduled delay: the whole
// point of the fix is that a stalled transfer gets a longer budget than a
// script that failed to boot after arriving, so the delay is load-bearing.
function createPanelRuntime(url, warmupUrl = null, fetchImpl = null) {
    const html = __test._viewerPanelHtml(url, warmupUrl);
    const scriptMatch = html.match(/<script nonce="[^"]+">([\s\S]*)<\/script>/);
    assert(scriptMatch, 'panel wrapper script must be present');

    const messages = [];
    const srcWrites = [];
    const timers = new Map();
    const scheduled = [];
    const windowHandlers = {};
    const frameHandlers = {};
    let nextTimer = 1;

    function schedule(handler, delay) {
        const id = nextTimer++;
        timers.set(id, handler);
        scheduled.push({ id, delay });
        return id;
    }

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
        fetch: fetchImpl || (() => Promise.resolve({})),
        setTimeout: schedule,
        window: {
            addEventListener(type, handler) { windowHandlers[type] = handler; },
            clearTimeout(id) { timers.delete(id); },
            setTimeout: schedule,
        },
    };
    vm.runInNewContext(scriptMatch[1], context);
    return { frameHandlers, messages, scheduled, srcWrites, timers, windowHandlers };
}

function liveTimer(runtime) {
    const live = [...runtime.timers.entries()];
    assert.strictEqual(live.length, 1, 'exactly one watchdog may be armed');
    return live[0];
}

// Real timers do not survive firing, so drop the entry before invoking it;
// otherwise a spent watchdog would masquerade as a second armed one.
function fireWatchdog(runtime) {
    const [id, handler] = liveTimer(runtime);
    runtime.timers.delete(id);
    handler();
}

function delayOf(runtime, id) {
    return runtime.scheduled.find(entry => entry.id === id).delay;
}

// The wrapper awaits the warmup fetch, so its continuation — including the
// frame.src assignment — lands on the microtask queue rather than running
// during vm.runInNewContext.
function flushMicrotasks() {
    return new Promise(resolve => setImmediate(resolve));
}

(async () => {
    try {
        // --- a navigation that never loads is retried -------------------------
        const stalled = createPanelRuntime('http://localhost:8123/?sid=stalled');
        assert.strictEqual(stalled.srcWrites.length, 1, 'the iframe must navigate');
        assert.strictEqual(
            delayOf(stalled, liveTimer(stalled)[0]),
            8000,
            'a stalled transfer must get the navigation budget, not the boot budget'
        );

        // No 'load' and no 'error' ever arrive — exactly the observed failure.
        fireWatchdog(stalled);
        assert.strictEqual(
            stalled.srcWrites.length,
            2,
            'a navigation that never completes must be retried'
        );
        assert.match(
            stalled.srcWrites[1],
            /_avretry=1$/,
            'the retry must be distinguishable from the stalled navigation'
        );

        // The replacement can stall in exactly the same way.
        assert.strictEqual(
            delayOf(stalled, liveTimer(stalled)[0]),
            8000,
            'a retry must re-arm the navigation watchdog'
        );
        fireWatchdog(stalled);
        assert.strictEqual(
            stalled.srcWrites.length,
            3,
            'retries must continue while the page never arrives'
        );

        // --- a black-holed warmup must not hold the navigation ----------------
        // Observed 2026-07-28T14:00:41Z: the warmup /ping was swallowed by the
        // relay, and because the navigation waited for it the panel sat doing
        // nothing for the full 12 s warmup budget before it even asked for the
        // page. The warmup is advisory — the navigation is a separate
        // connection — so a failed warmup must cost its budget and no more,
        // and must still navigate.
        const coldWarmup = createPanelRuntime(
            'http://localhost:8126/?sid=cold',
            'http://localhost:8126/ping',
            () => Promise.reject(new Error('aborted')),
        );
        await flushMicrotasks();
        assert.strictEqual(
            coldWarmup.srcWrites.length,
            1,
            'a failed warmup must still navigate'
        );
        assert.ok(
            coldWarmup.messages.some(m => m.phase === 'transport-warmup-failed'),
            'the failed warmup must still be reported'
        );
        // And having just been told the relay is swallowing connections, the
        // watchdog must not sit on the full transfer budget before retrying.
        assert.strictEqual(
            delayOf(coldWarmup, liveTimer(coldWarmup)[0]),
            3000,
            'a navigation issued into a known-bad relay window must be retried '
            + 'sooner than one issued after a healthy warmup'
        );

        // A healthy warmup means the relay is answering, so a slow navigation
        // is a real 1.9 MB transfer and must be given the full budget.
        const warmWarmup = createPanelRuntime(
            'http://localhost:8127/?sid=warm',
            'http://localhost:8127/ping',
        );
        await flushMicrotasks();
        assert.strictEqual(warmWarmup.srcWrites.length, 1);
        assert.strictEqual(
            delayOf(warmWarmup, liveTimer(warmWarmup)[0]),
            8000,
            'a healthy warmup must not shorten the transfer budget'
        );

        // --- arriving swaps to the shorter boot watchdog ----------------------
        const arrived = createPanelRuntime('http://localhost:8124/?sid=arrived');
        assert.strictEqual(arrived.srcWrites.length, 1);
        arrived.frameHandlers.load();
        assert.strictEqual(
            delayOf(arrived, liveTimer(arrived)[0]),
            1500,
            'once the page has arrived the shorter boot watchdog applies'
        );

        // --- a healthy viewer disarms everything ------------------------------
        const healthy = createPanelRuntime('http://localhost:8125/?sid=healthy');
        healthy.frameHandlers.load();
        healthy.windowHandlers.message({
            data: { source: 'arrayview-viewer', phase: 'script-loaded' },
        });
        assert.strictEqual(
            healthy.timers.size,
            0,
            'script-loaded must cancel the watchdog that would interrupt rendering'
        );
        assert.strictEqual(
            healthy.srcWrites.length,
            1,
            'a healthy viewer must never be renavigated'
        );

        // --- warmup still gates the first navigation --------------------------
        const warmed = createPanelRuntime(
            'http://localhost:8126/?sid=warmed',
            'http://localhost:8126/ping'
        );
        assert.strictEqual(
            warmed.srcWrites.length,
            0,
            'the iframe must not navigate before transport warmup completes'
        );
        // The warmup abort timer is live at this point, so count watchdogs by
        // their distinctive delay rather than by how many timers exist.
        const watchdogs = runtime => runtime.scheduled
            .filter(entry => entry.delay === 8000).length;
        assert.strictEqual(
            watchdogs(warmed),
            0,
            'the navigation watchdog must not run before navigation starts'
        );
        await new Promise(resolve => setImmediate(resolve));
        await new Promise(resolve => setImmediate(resolve));
        assert.strictEqual(warmed.srcWrites.length, 1);
        assert.strictEqual(
            watchdogs(warmed),
            1,
            'the watchdog must arm once the warmed navigation begins'
        );

        console.log('panel navigation watchdog tests passed');
    } finally {
        if (originalHome === undefined) delete process.env.HOME;
        else process.env.HOME = originalHome;
        fs.rmSync(tempHome, { recursive: true, force: true });
    }
})().catch(error => {
    console.error(error);
    process.exitCode = 1;
});
