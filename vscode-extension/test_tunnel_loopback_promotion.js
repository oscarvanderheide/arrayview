const assert = require('assert');
const fs = require('fs');
const https = require('https');
const Module = require('module');
const os = require('os');
const path = require('path');

const tempHome = fs.mkdtempSync(path.join(os.tmpdir(), 'arrayview-tunnel-promotion-'));
const originalHome = process.env.HOME;
process.env.HOME = tempHome;
fs.mkdirSync(path.join(tempHome, '.arrayview'));

const events = [];
let resolverCall = 0;
let privacyCommandRegistered = false;
const vscodeMock = {
    env: {
        remoteName: 'tunnel',
        appHost: 'web',
        asExternalUri: async uri => {
            resolverCall += 1;
            const call = resolverCall;
            const port = new URL(String(uri)).port;
            events.push(`resolve:${port}:${call}`);
            return {
                toString: () => `http://localhost:${port}/`,
            };
        },
    },
    Uri: { parse: value => value },
    ConfigurationTarget: { Global: 1 },
    workspace: {
        getConfiguration: section => ({
            get: (key, fallback) => (
                section === 'workbench.browser' && key === 'enableRemoteProxy'
                    ? true
                    : (key === 'portsAttributes' ? {} : fallback)
            ),
            update: async () => { events.push('promote:settings'); },
        }),
    },
    commands: {
        executeCommand: async (command, item) => {
            events.push(`promote:${command}`);
            if (command === '~remote.forwardedPorts.focus') {
                privacyCommandRegistered = true;
                return true;
            }
            if (command === 'remote.tunnel.privacypublic') {
                if (!privacyCommandRegistered) {
                    throw new Error('command not found');
                }
                return item.remotePort === 8001
                    ? 'Failed to resolve remote viewer URL'
                    : `https://public-${item.remotePort}.devtunnels.ms/`;
            }
            return true;
        },
        getCommands: async () => {
            events.push('promote:getCommands');
            return privacyCommandRegistered
                ? ['remote.tunnel.privacypublic']
                : [];
        },
    },
};

const originalLoad = Module._load;
Module._load = function(request, parent, isMain) {
    if (request === 'vscode') return vscodeMock;
    return originalLoad.call(this, request, parent, isMain);
};
const { __test } = require('./extension');
Module._load = originalLoad;

const originalHttpsGet = https.get;
const pingHosts = [];
https.get = (url, options, callback) => {
    pingHosts.push(new URL(String(url)).hostname);
    const handlers = {};
    const request = {
        on(event, handler) { handlers[event] = handler; return request; },
        destroy() {},
    };
    queueMicrotask(() => {
        const responseHandlers = {};
        const response = {
            statusCode: 200,
            setEncoding() {},
            on(event, handler) {
                responseHandlers[event] = handler;
                if (event === 'end') {
                    queueMicrotask(() => {
                        const hostname = new URL(String(url)).hostname;
                        responseHandlers.data(JSON.stringify({
                            service: 'arrayview',
                            instance_id: hostname === 'public-8002.devtunnels.ms'
                                ? 'previous-server'
                                : 'server-loopback',
                        }));
                        handler();
                    });
                }
            },
        };
        callback(response);
    });
    return request;
};

(async () => {
    try {
        const resolved = await __test.resolveRemoteViewerUrl(
            'http://localhost:8000/?sid=tunnel-loopback',
            'server-loopback'
        );
        assert.strictEqual(
            resolved,
            'https://public-8000.devtunnels.ms/?sid=tunnel-loopback'
        );
        assert.strictEqual(
            resolverCall,
            1,
            'the resolver must materialize the private route exactly once before promotion'
        );
        assert.deepStrictEqual(events, [
            'resolve:8000:1',
            'promote:remote.tunnel.privacypublic',
            'promote:~remote.forwardedPorts.focus',
            'promote:getCommands',
            'promote:remote.tunnel.privacypublic',
        ], 'no resolver call may occur after the final privacy promotion');
        assert.deepStrictEqual(
            pingHosts,
            ['public-8000.devtunnels.ms'],
            'the URL-shaped command result must pass /ping for the exact server before use'
        );

        events.length = 0;
        pingHosts.length = 0;
        const rejected = await __test.resolveRemoteViewerUrl(
            'http://localhost:8001/?sid=reject-status-string',
            'server-loopback'
        );
        assert.strictEqual(
            rejected,
            null,
            'an arbitrary string returned by the privacy command must be rejected'
        );
        assert.strictEqual(resolverCall, 2);
        assert.deepStrictEqual(events, [
            'resolve:8001:2',
            'promote:remote.tunnel.privacypublic',
            'promote:~remote.forwardedPorts.focus',
            'promote:getCommands',
            'promote:remote.tunnel.privacypublic',
        ], 'string rejection must not trigger another resolver call after promotion');
        assert.deepStrictEqual(
            pingHosts,
            [],
            'an arbitrary command string must be rejected before any route probe'
        );

        events.length = 0;
        pingHosts.length = 0;
        const originalDateNow = Date.now;
        let fakeNow = originalDateNow();
        Date.now = () => {
            fakeNow += 10000;
            return fakeNow;
        };
        let wrongIdentity;
        try {
            wrongIdentity = await __test.resolveRemoteViewerUrl(
                'http://localhost:8002/?sid=reject-wrong-server',
                'server-loopback'
            );
        } finally {
            Date.now = originalDateNow;
        }
        assert.strictEqual(
            wrongIdentity,
            null,
            'a public URL-shaped command result must be rejected when /ping identifies another server'
        );
        assert.strictEqual(resolverCall, 3);
        assert.deepStrictEqual(events, [
            'resolve:8002:3',
            'promote:remote.tunnel.privacypublic',
        ], 'a failed exact-server probe must not trigger another resolver after promotion');
        assert.deepStrictEqual(pingHosts, ['public-8002.devtunnels.ms']);
        console.log('tunnel loopback promotion tests passed');
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
