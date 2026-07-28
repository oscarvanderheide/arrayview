// Regression coverage for tunnel route probes that time out rather than fail.
//
// The route cache tests next door answer every probe instantly with a 200, and
// their fake request object stubs `on()` to a no-op, so a timeout cannot be
// expressed there at all. That harness can only distinguish "route is good"
// from "route is the wrong server" — the binary cases. A devtunnel relay's real
// failure mode is neither: the route is correct and the backend is up, but the
// probe stalls. This file models that, because a stall used to be read as proof
// the route was dead, which discarded a working URL mid-session and failed the
// launch.
const assert = require('assert');
const fs = require('fs');
const http = require('http');
const https = require('https');
const Module = require('module');
const os = require('os');
const path = require('path');

const tempHome = fs.mkdtempSync(path.join(os.tmpdir(), 'arrayview-probe-'));
const originalHome = process.env.HOME;
process.env.HOME = tempHome;
const signalDir = path.join(tempHome, '.arrayview');
const routeCachePath = path.join(signalDir, 'tunnel-routes.json');
fs.mkdirSync(signalDir);

let resolverBehavior = () => { throw new Error('asExternalUri timeout'); };
let resolverCalls = 0;
const vscodeMock = {
    env: {
        remoteName: 'tunnel',
        appHost: 'web',
        asExternalUri: async uri => {
            resolverCalls += 1;
            return resolverBehavior(uri);
        },
    },
    Uri: { parse: value => value },
    ConfigurationTarget: { Global: 1 },
    workspace: {
        getConfiguration: () => ({
            get: (key, fallback) => (key === 'portsAttributes' ? {} : fallback),
            update: async () => {},
        }),
    },
    commands: {
        executeCommand: async () => undefined,
        getCommands: async () => [],
    },
};

const originalLoad = Module._load;
Module._load = function(request, parent, isMain) {
    if (request === 'vscode') return vscodeMock;
    return originalLoad.call(this, request, parent, isMain);
};
const { __test } = require('./extension');
Module._load = originalLoad;

// Shrink both schedules so the suite exercises the real control flow without
// waiting out the production timings. The hedge keeps its production shape —
// three overlapping attempts — because the attempt count is what the assertions
// below are about.
__test._setRetryTiming({
    relayProbeHedge: { attempts: 3, staggerMs: 10, attemptTimeoutMs: 40 },
    externalUriAttempts: [
        { timeoutMs: 20, pauseMs: 0 },
        { timeoutMs: 20, pauseMs: 0 },
    ],
});

// hostname -> queue of behaviors, consumed one per probe. 'stall' never calls
// back, so the request's own timeout fires exactly as it does against a relay
// that has accepted the connection and then gone quiet.
let routeBehaviors = new Map();
const probeLog = [];

// The resolver now asks loopback which backend owns the port before doing any
// remote work. Stub it so the answer is hermetic: unstubbed, the probe reaches
// whatever real service happens to occupy that port on the dev machine, and a
// foreign answer would make the resolver correctly abandon the request. This
// stub always claims ownership so the stall/flake cases below stay reachable.
const originalHttpGet = http.get;
http.get = (url, options, callback) => {
    const request = {
        on() { return request; },
        destroy() {},
    };
    queueMicrotask(() => {
        const handlers = {};
        callback({
            statusCode: 200,
            setEncoding() {},
            on(event, handler) {
                handlers[event] = handler;
                if (event === 'end') {
                    queueMicrotask(() => {
                        handlers.data(JSON.stringify({
                            service: 'arrayview',
                            instance_id: 'current-server',
                        }));
                        handler();
                    });
                }
            },
        });
    });
    return request;
};

const originalHttpsGet = https.get;
https.get = (url, options, callback) => {
    const hostname = new URL(String(url)).hostname;
    probeLog.push(hostname);
    const queue = routeBehaviors.get(hostname) || [];
    const behavior = queue.length > 1 ? queue.shift() : (queue[0] || 'dead');

    const handlers = {};
    let destroyed = false;
    const request = {
        on(event, handler) {
            handlers[event] = handler;
            return request;
        },
        destroy() { destroyed = true; },
    };

    if (behavior === 'stall') {
        // Mirror Node: the socket timeout fires, the caller destroys the
        // request, and no response ever arrives.
        setTimeout(() => {
            if (handlers.timeout) handlers.timeout();
        }, options.timeout);
    } else if (behavior === 'refused') {
        setTimeout(() => {
            const error = new Error('connect ECONNREFUSED');
            error.code = 'ECONNREFUSED';
            if (handlers.error) handlers.error(error);
        }, 1);
    } else if (behavior === 'reset') {
        setTimeout(() => {
            const error = new Error('socket hang up');
            error.code = 'ECONNRESET';
            if (handlers.error) handlers.error(error);
        }, 1);
    } else if (behavior === 'relay-down') {
        // The relay itself answers, fast, saying it is not carrying the port.
        setTimeout(() => {
            if (destroyed) return;
            callback({
                statusCode: 502,
                setEncoding() {},
                resume() {},
                on() {},
            });
        }, 1);
    } else {
        setTimeout(() => {
            if (destroyed) return;
            const resHandlers = {};
            callback({
                statusCode: 200,
                setEncoding() {},
                resume() {},
                on(event, handler) {
                    resHandlers[event] = handler;
                    if (event === 'end') {
                        setTimeout(() => {
                            resHandlers.data(JSON.stringify({
                                service: 'arrayview',
                                instance_id: behavior === 'foreign'
                                    ? 'another-server'
                                    : 'current-server',
                            }));
                            handler();
                        }, 1);
                    }
                },
            });
        }, 1);
    }
    return request;
};

function reset(cache, behaviors) {
    fs.writeFileSync(routeCachePath, JSON.stringify(cache));
    routeBehaviors = new Map(Object.entries(behaviors));
    probeLog.length = 0;
    resolverCalls = 0;
}

(async () => {
    try {
        // 1. A cached route that stalls once and then answers is still the
        //    right route. This is the exact shape of the observed failure: the
        //    first probe timed out, and a healthy tunnel was declared stale.
        reset(
            { 'prior-window:8000': 'https://slow-8000.devtunnels.ms' },
            { 'slow-8000.devtunnels.ms': ['stall', 'ok'] }
        );
        assert.strictEqual(
            await __test.resolveRemoteViewerUrl(
                'http://localhost:8000/?sid=slow', 'current-server'
            ),
            'https://slow-8000.devtunnels.ms/?sid=slow',
            'a cached route that stalls once must be re-probed, not discarded'
        );
        assert.strictEqual(
            resolverCalls, 0,
            'a recoverable cached route must never reach asExternalUri'
        );
        assert.strictEqual(probeLog.length, 2, 'the stalled probe must be retried');

        // 2. A transient reset is also not proof of death.
        reset(
            { 'prior-window:8001': 'https://flaky-8001.devtunnels.ms' },
            { 'flaky-8001.devtunnels.ms': ['reset', 'ok'] }
        );
        assert.strictEqual(
            await __test.resolveRemoteViewerUrl(
                'http://localhost:8001/?sid=flaky', 'current-server'
            ),
            'https://flaky-8001.devtunnels.ms/?sid=flaky',
            'ECONNRESET is a network fact, not a verdict on the route'
        );
        assert.strictEqual(resolverCalls, 0);

        // 3. The reported production failure: every hedged connection to a
        //    perfectly good relay is black-holed, so the route cannot be
        //    verified at all. A no-answer is not evidence, so the cached route
        //    must be used as-is — immediately, without paying for asExternalUri
        //    and port promotion to re-derive the identical URL.
        //
        //    This is the assertion that would have caught the 2026-07-28
        //    incident: the old code discarded the route here and spent 20 s
        //    arriving back at the same string.
        reset(
            { 'prior-window:8002': 'https://wedged-8002.devtunnels.ms' },
            { 'wedged-8002.devtunnels.ms': ['stall'] }
        );
        resolverBehavior = () => { throw new Error('asExternalUri timeout'); };
        assert.strictEqual(
            await __test.resolveRemoteViewerUrl(
                'http://localhost:8002/?sid=wedged', 'current-server'
            ),
            'https://wedged-8002.devtunnels.ms/?sid=wedged',
            'a route that only ever stalls must still be used, not discarded'
        );
        assert.strictEqual(
            resolverCalls, 0,
            'an unverifiable route must never reach asExternalUri: a stall is '
            + 'an absence of evidence, and re-deriving returns the same URL'
        );
        assert.strictEqual(
            probeLog.filter(h => h === 'wedged-8002.devtunnels.ms').length,
            3,
            'all three hedged attempts should have been spent before giving up '
            + 'on a verdict'
        );

        // 3b. Hedging is the reason 3 is cheap: the attempts overlap, so one
        //     black-holed connection does not delay the next. With a 40 ms
        //     attempt budget and a 10 ms stagger, a verdict that only the
        //     second connection can supply must arrive in well under the two
        //     attempt budgets a sequential ladder would have cost.
        reset(
            { 'prior-window:8006': 'https://hedge-8006.devtunnels.ms' },
            { 'hedge-8006.devtunnels.ms': ['stall', 'ok'] }
        );
        const hedgeStart = Date.now();
        assert.strictEqual(
            await __test.resolveRemoteViewerUrl(
                'http://localhost:8006/?sid=hedged', 'current-server'
            ),
            'https://hedge-8006.devtunnels.ms/?sid=hedged'
        );
        const hedgeMs = Date.now() - hedgeStart;
        assert.ok(
            hedgeMs < 40,
            `a hedged verdict must not wait out the stalled attempt `
            + `(took ${hedgeMs}ms, stalled attempt budget is 40ms)`
        );

        // 3c. A 502 is not a stall, and must not be treated like one. The relay
        //     is answering — it just is not carrying the port — so handing the
        //     panel this route would hand it a URL known to 502. Port promotion
        //     is what reattaches the connector, so the request must fall
        //     through to it rather than short-circuit on the cache.
        reset(
            { 'prior-window:8007': 'https://detached-8007.devtunnels.ms' },
            { 'detached-8007.devtunnels.ms': ['relay-down'] }
        );
        resolverBehavior = () => ({
            toString: () => 'https://detached-8007.devtunnels.ms/',
        });
        assert.strictEqual(
            await __test.resolveRemoteViewerUrl(
                'http://localhost:8007/?sid=detached', 'current-server'
            ),
            null,
            'a detached relay must not be served straight from the cache'
        );
        assert.ok(
            resolverCalls > 0,
            'a detached connector must reach the promotion path that fixes it'
        );

        // 4. A foreign server ID is proof. It must not be retried, and it must
        //    not be resurrected by the last-resort path either.
        reset(
            { 'prior-window:8003': 'https://foreign-8003.devtunnels.ms' },
            { 'foreign-8003.devtunnels.ms': ['foreign'] }
        );
        resolverBehavior = () => { throw new Error('asExternalUri timeout'); };
        assert.strictEqual(
            await __test.resolveRemoteViewerUrl(
                'http://localhost:8003/?sid=foreign', 'current-server'
            ),
            null,
            'a route belonging to another backend must never be served'
        );
        assert.strictEqual(
            probeLog.filter(h => h === 'foreign-8003.devtunnels.ms').length,
            2,
            'a wrong-server verdict is final per check: one probe, not a retry'
        );

        // 5. A refused route is equally final.
        reset(
            { 'prior-window:8004': 'https://gone-8004.devtunnels.ms' },
            { 'gone-8004.devtunnels.ms': ['refused'] }
        );
        resolverBehavior = () => { throw new Error('asExternalUri timeout'); };
        assert.strictEqual(
            await __test.resolveRemoteViewerUrl(
                'http://localhost:8004/?sid=gone', 'current-server'
            ),
            null
        );

        // 6. When the resolver does work, its answer is preferred and the
        //    request never depends on the cache at all.
        reset({}, { 'fresh-8005.devtunnels.ms': ['ok'] });
        resolverBehavior = () => ({
            toString: () => 'https://fresh-8005.devtunnels.ms/',
        });
        assert.strictEqual(
            await __test.resolveRemoteViewerUrl(
                'http://localhost:8005/?sid=fresh', 'current-server'
            ),
            'https://fresh-8005.devtunnels.ms/?sid=fresh'
        );

        console.log('tunnel probe resilience tests passed');
    } finally {
        https.get = originalHttpsGet;
        if (originalHome === undefined) delete process.env.HOME;
        else process.env.HOME = originalHome;
        fs.rmSync(tempHome, { recursive: true, force: true });
    }
})().catch(error => {
    console.error(error);
    process.exitCode = 1;
});
