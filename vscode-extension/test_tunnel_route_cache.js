const assert = require('assert');
const fs = require('fs');
const https = require('https');
const Module = require('module');
const os = require('os');
const path = require('path');

const tempHome = fs.mkdtempSync(path.join(os.tmpdir(), 'arrayview-tunnel-cache-'));
const originalHome = process.env.HOME;
process.env.HOME = tempHome;
const signalDir = path.join(tempHome, '.arrayview');
const routeCachePath = path.join(signalDir, 'tunnel-routes.json');
fs.mkdirSync(signalDir);

let resolverCalls = 0;
const vscodeMock = {
    env: {
        remoteName: 'tunnel',
        appHost: 'web',
        asExternalUri: async uri => {
            resolverCalls += 1;
            const port = new URL(String(uri)).port;
            return {
                toString: () => `https://fresh-${port}.devtunnels.ms/`,
            };
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
Module._load = function(request, parent, isMain) {
    if (request === 'vscode') return vscodeMock;
    return originalLoad.call(this, request, parent, isMain);
};
const { __test } = require('./extension');
Module._load = originalLoad;

const pingHosts = [];
const originalHttpsGet = https.get;
https.get = (url, options, callback) => {
    const hostname = new URL(String(url)).hostname;
    pingHosts.push(hostname);
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
                        const instanceId = hostname.startsWith('cached-wrong-')
                            ? 'previous-server'
                            : 'current-server';
                        handlers.data(JSON.stringify({
                            service: 'arrayview',
                            instance_id: instanceId,
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
        fs.writeFileSync(routeCachePath, JSON.stringify({
            'prior-window:8000': 'https://cached-good-8000.devtunnels.ms',
        }));

        const cachedResolved = await __test.resolveRemoteViewerUrl(
            'http://localhost:8000/?sid=cross-window-cache',
            'current-server'
        );
        assert.strictEqual(
            cachedResolved,
            'https://cached-good-8000.devtunnels.ms/?sid=cross-window-cache'
        );
        assert.strictEqual(
            resolverCalls,
            0,
            'an exact-server route cached by a prior window should be reused'
        );
        assert.deepStrictEqual(pingHosts, [
            'cached-good-8000.devtunnels.ms',
        ]);

        fs.writeFileSync(routeCachePath, JSON.stringify({
            'prior-window:8001': 'https://cached-wrong-8001.devtunnels.ms',
        }));
        pingHosts.length = 0;

        const freshResolved = await __test.resolveRemoteViewerUrl(
            'http://localhost:8001/?sid=reject-wrong-server',
            'current-server'
        );
        assert.strictEqual(
            freshResolved,
            'https://fresh-8001.devtunnels.ms/?sid=reject-wrong-server'
        );
        assert.strictEqual(
            resolverCalls,
            1,
            'a cached route for the wrong server must fall through to resolution'
        );
        assert.deepStrictEqual(
            pingHosts,
            [
                'cached-wrong-8001.devtunnels.ms',
                'fresh-8001.devtunnels.ms',
            ],
            'the prior-window route must be rejected unless /ping matches the exact server ID'
        );

        console.log('cross-window tunnel route cache tests passed');
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
