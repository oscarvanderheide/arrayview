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

// Shrink both backoff schedules so the suite exercises the real control flow
// without waiting out the production timings.
__test._setRetryTiming({
    cachedRouteProbeTimeoutsMs: [20, 40, 60],
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

        // 3. The production failure end to end: the cached route never answers
        //    on the initial check, and VS Code's resolver is wedged so every
        //    attempt throws without returning. The loopback branch that used to
        //    hold the only cache recovery is therefore never entered. Once the
        //    relay recovers, the verified route must still be used instead of
        //    failing the request.
        reset(
            { 'prior-window:8002': 'https://wedged-8002.devtunnels.ms' },
            {
                'wedged-8002.devtunnels.ms': [
                    'stall', 'stall', 'stall',  // initial check gives up
                    'ok',                       // relay recovers by last resort
                ],
            }
        );
        resolverBehavior = () => { throw new Error('asExternalUri timeout'); };
        assert.strictEqual(
            await __test.resolveRemoteViewerUrl(
                'http://localhost:8002/?sid=wedged', 'current-server'
            ),
            'https://wedged-8002.devtunnels.ms/?sid=wedged',
            'a wedged resolver must fall back to the verified cached route'
        );
        assert.strictEqual(
            resolverCalls, 2,
            'the resolver should still have been given its full chance first'
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
