// Regression coverage for the loopback ownership pre-check in
// resolveRemoteViewerUrl.
//
// Observed failure (see ~/.arrayview/extension.log, 2026-07-26T22:13:49Z and
// 2026-07-26T20:05:27Z): a request whose backend had already been replaced on
// its port stayed undetected until the whole asExternalUri backoff expired —
// 88s and 75s of dead waiting that ended in failure anyway. Worse, the signal
// queue serialises on isProcessingSignal, so a newer request for a live backend
// sat on disk for 69s behind it and only then rendered, in 1.3s.
//
// The cheap authority is loopback: the extension host and the backend share a
// machine even in a tunnel window, so /ping answers in milliseconds. The
// asymmetry that matters is that abandoning a request is irreversible, so only
// positive proof of a foreign occupant may do it. A port that is merely not
// listening yet — a large array still loading before it binds — must be
// indistinguishable from success, because that is exactly the state the real
// 22:13:49 request was in when its first three route probes timed out.

const assert = require('assert');
const fs = require('fs');
const http = require('http');
const https = require('https');
const Module = require('module');
const os = require('os');
const path = require('path');

const tempHome = fs.mkdtempSync(path.join(os.tmpdir(), 'arrayview-local-identity-'));
const originalHome = process.env.HOME;
process.env.HOME = tempHome;
fs.mkdirSync(path.join(tempHome, '.arrayview'));

let resolverCalls = 0;
const vscodeMock = {
    env: {
        remoteName: 'ssh-remote',
        appHost: 'web',
        asExternalUri: async (uri) => {
            resolverCalls += 1;
            const port = new URL(String(uri)).port;
            return { toString: () => `https://resolved-${port}.devtunnels.ms/` };
        },
    },
    Uri: { parse: value => value },
    ConfigurationTarget: { Global: 1 },
    workspace: {
        getConfiguration: section => ({
            get: (key, fallback) => (
                section === 'workbench.browser' && key === 'enableRemoteProxy'
                    ? false
                    : (key === 'portsAttributes' ? {} : fallback)
            ),
            update: async () => {},
        }),
    },
    commands: {
        executeCommand: async () => undefined,
        getCommands: async () => [],
    },
};

const originalLoad = Module._load;
Module._load = function (request, parent, isMain) {
    if (request === 'vscode') return vscodeMock;
    return originalLoad.call(this, request, parent, isMain);
};
const { __test } = require('./extension');
Module._load = originalLoad;

// Loopback behaviour is chosen per test case.
let localMode = 'mine';
const localProbes = [];
const originalHttpGet = http.get;
http.get = (url, options, callback) => {
    localProbes.push(String(url));
    const handlers = {};
    const request = {
        on(event, handler) { handlers[event] = handler; return request; },
        destroy() {},
    };
    if (localMode === 'refused') {
        // Nothing is listening yet: the array is still loading. Node reports
        // ECONNREFUSED, which is deliberately NOT in TRANSIENT_PROBE_ERRORS,
        // so this is the case most at risk of being misread as proof of death.
        queueMicrotask(() => {
            const error = new Error('connect ECONNREFUSED 127.0.0.1');
            error.code = 'ECONNREFUSED';
            if (handlers.error) handlers.error(error);
        });
        return request;
    }
    queueMicrotask(() => {
        const responseHandlers = {};
        callback({
            statusCode: 200,
            setEncoding() {},
            on(event, handler) {
                responseHandlers[event] = handler;
                if (event === 'end') {
                    queueMicrotask(() => {
                        responseHandlers.data(JSON.stringify({
                            service: 'arrayview',
                            instance_id: localMode === 'foreign'
                                ? 'a-different-backend'
                                : 'expected-server',
                        }));
                        handler();
                    });
                }
            },
        });
    });
    return request;
};

// Remote probes always confirm the resolved route, so any failure below is
// attributable to the loopback pre-check rather than route verification.
const originalHttpsGet = https.get;
https.get = (url, options, callback) => {
    const handlers = {};
    const request = {
        on(event, handler) { handlers[event] = handler; return request; },
        destroy() {},
    };
    queueMicrotask(() => {
        const responseHandlers = {};
        callback({
            statusCode: 200,
            setEncoding() {},
            on(event, handler) {
                responseHandlers[event] = handler;
                if (event === 'end') {
                    queueMicrotask(() => {
                        responseHandlers.data(JSON.stringify({
                            service: 'arrayview',
                            instance_id: 'expected-server',
                        }));
                        handler();
                    });
                }
            },
        });
    });
    return request;
};

(async () => {
    try {
        // --- the strict verdict itself -----------------------------------
        localMode = 'mine';
        assert.strictEqual(
            await __test.localBackendIdentity(8000, 'expected-server'),
            __test.LOCAL_MINE
        );

        localMode = 'foreign';
        assert.strictEqual(
            await __test.localBackendIdentity(8000, 'expected-server'),
            __test.LOCAL_FOREIGN
        );

        localMode = 'refused';
        assert.strictEqual(
            await __test.localBackendIdentity(8000, 'expected-server'),
            __test.LOCAL_UNKNOWN,
            'a port that is not listening yet proves nothing'
        );

        localMode = 'mine';
        assert.strictEqual(
            await __test.localBackendIdentity(8000, null),
            __test.LOCAL_UNKNOWN,
            'without an expected server id there is nothing to compare'
        );

        // --- a stale request is abandoned immediately --------------------
        localMode = 'foreign';
        resolverCalls = 0;
        const startedAt = Date.now();
        const stale = await __test.resolveRemoteViewerUrl(
            'http://localhost:8000/?sid=stale', 'expected-server'
        );
        const elapsedMs = Date.now() - startedAt;
        assert.strictEqual(stale, null, 'a replaced backend must be abandoned');
        assert.strictEqual(
            resolverCalls,
            0,
            'abandoning must happen before asExternalUri is ever consulted'
        );
        // The real incident burned 88s here. Anything in this range proves the
        // backoff chain was skipped entirely rather than merely shortened.
        assert.ok(
            elapsedMs < 2000,
            `stale detection must be immediate, took ${elapsedMs}ms`
        );

        // --- a still-starting backend is NOT abandoned -------------------
        // This is the guard on the fix: the 22:13:49 request was in exactly
        // this state and must still be given every chance to resolve.
        localMode = 'refused';
        resolverCalls = 0;
        const starting = await __test.resolveRemoteViewerUrl(
            'http://localhost:8001/?sid=starting', 'expected-server'
        );
        assert.strictEqual(
            starting,
            'https://resolved-8001.devtunnels.ms/?sid=starting',
            'a backend that has not bound its port yet must still resolve'
        );
        assert.ok(
            resolverCalls > 0,
            'a not-listening port must not short-circuit the resolver'
        );

        // --- a live backend resolves normally ----------------------------
        localMode = 'mine';
        resolverCalls = 0;
        const live = await __test.resolveRemoteViewerUrl(
            'http://localhost:8002/?sid=live', 'expected-server'
        );
        assert.strictEqual(
            live,
            'https://resolved-8002.devtunnels.ms/?sid=live',
            'the owning backend must resolve unchanged'
        );

        console.log('local identity short-circuit tests passed');
    } finally {
        http.get = originalHttpGet;
        https.get = originalHttpsGet;
        if (originalHome === undefined) delete process.env.HOME;
        else process.env.HOME = originalHome;
        fs.rmSync(tempHome, { recursive: true, force: true });
    }
})().catch(error => {
    console.error(error);
    process.exitCode = 1;
});
