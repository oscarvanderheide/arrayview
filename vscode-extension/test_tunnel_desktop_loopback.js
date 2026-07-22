const assert = require('assert');
const fs = require('fs');
const http = require('http');
const https = require('https');
const Module = require('module');
const os = require('os');
const path = require('path');

const tempHome = fs.mkdtempSync(path.join(os.tmpdir(), 'arrayview-desktop-tunnel-'));
const originalHome = process.env.HOME;
process.env.HOME = tempHome;
fs.mkdirSync(path.join(tempHome, '.arrayview'));

let resolverCalls = 0;
let promotionCalls = 0;
let proxyEnabled = true;
let privacyItem = null;
const vscodeMock = {
    env: {
        remoteName: 'tunnel',
        appHost: 'desktop',
        asExternalUri: async uri => {
            resolverCalls += 1;
            const value = String(uri);
            if (value.includes(':8001/')) {
                return {
                    toString: () => resolverCalls <= 2
                        ? 'http://localhost:8001/'
                        : 'https://public-8001.devtunnels.ms/',
                };
            }
            return { toString: () => 'http://localhost:8002/' };
        },
    },
    Uri: { parse: value => value },
    ConfigurationTarget: { Global: 1 },
    workspace: {
        getConfiguration: section => ({
            get: (key, fallback) => (
                section === 'workbench.browser' && key === 'enableRemoteProxy'
                    ? proxyEnabled
                    : (key === 'portsAttributes' ? {} : fallback)
            ),
            update: async () => { promotionCalls += 1; },
        }),
    },
    commands: {
        executeCommand: async (command, item) => {
            promotionCalls += 1;
            if (command === 'remote.tunnel.privacypublic') privacyItem = item;
            return {
                localAddress: 'https://public-8001.devtunnels.ms/',
                tunnelRemotePort: 8001,
                tunnelRemoteHost: 'localhost',
                privacy: 'public',
                protocol: 'http',
            };
        },
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
                            instance_id: 'desktop-server',
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
https.get = http.get;

(async () => {
    try {
        assert.strictEqual(
            __test._publicBaseFromTunnelResult(undefined, 8001),
            null
        );
        assert.strictEqual(
            __test._publicBaseFromTunnelResult(
                'Failed to resolve remote viewer URL',
                8001
            ),
            null,
            'an arbitrary privacy-command status string is not a public route'
        );
        assert.strictEqual(
            __test._publicBaseFromTunnelResult(
                'https://public-8001.devtunnels.ms/',
                8001
            ),
            'https://public-8001.devtunnels.ms',
            'a URL-shaped public command result can be considered for verification'
        );
        assert.strictEqual(
            __test._publicBaseFromTunnelResult({
                localAddress: 'https://wrong-port.devtunnels.ms/',
                tunnelRemotePort: 9000,
                tunnelRemoteHost: 'localhost',
                privacy: 'public',
            }, 8001),
            null
        );
        assert.strictEqual(
            __test._publicBaseFromTunnelResult({
                localAddress: 'public-8001.devtunnels.ms',
                tunnelRemotePort: 8001,
                tunnelRemoteHost: 'localhost',
                privacy: 'private',
                protocol: 'https',
            }, 8001),
            null
        );
        assert.strictEqual(
            __test._publicBaseFromTunnelResult({
                localAddress: 'public-8001.devtunnels.ms',
                tunnelRemotePort: 8001,
                tunnelRemoteHost: 'localhost',
                privacy: 'public',
                protocol: 'https',
            }, 8001),
            'https://public-8001.devtunnels.ms'
        );
        const resolved = await __test.resolveRemoteViewerUrl(
            'http://localhost:8000/?sid=desktop-tunnel',
            'desktop-server'
        );
        assert.strictEqual(
            resolved,
            'http://localhost:8000/?sid=desktop-tunnel'
        );
        assert.strictEqual(
            resolverCalls,
            0,
            'remote proxy must bypass asExternalUri and public promotion'
        );
        assert.strictEqual(
            promotionCalls,
            0,
            'desktop tunnel remote proxy must not require public promotion'
        );

        proxyEnabled = false;
        const publicResolved = await __test.resolveRemoteViewerUrl(
            'http://localhost:8001/?sid=desktop-public',
            'desktop-server'
        );
        assert.strictEqual(
            publicResolved,
            'https://public-8001.devtunnels.ms/?sid=desktop-public'
        );
        assert.strictEqual(
            resolverCalls,
            1,
            'the default/private forward must be resolved exactly once before final promotion'
        );
        assert.strictEqual(promotionCalls, 1);
        assert.deepStrictEqual(privacyItem, {
            tunnelType: 'Forwarded',
            remoteHost: 'localhost',
            remotePort: 8001,
            localPort: 8001,
            name: 'ArrayView',
            source: { source: 0, description: 'User Forwarded' },
        });

        vscodeMock.env.remoteName = 'ssh-remote';
        const sshResolved = await __test.resolveRemoteViewerUrl(
            'http://localhost:8002/?sid=remote-ssh',
            'desktop-server'
        );
        assert.strictEqual(
            sshResolved,
            'http://localhost:8002/?sid=remote-ssh'
        );
        assert.strictEqual(
            promotionCalls,
            1,
            'Remote SSH must not mutate tunnel port privacy'
        );
        console.log('desktop tunnel and Remote SSH routing tests passed');
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
