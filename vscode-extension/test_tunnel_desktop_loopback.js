// Private-only routing contract for tunnels, with Remote SSH preserved.
const assert = require('assert');
const { EventEmitter } = require('events');
const fs = require('fs');
const Module = require('module');
const os = require('os');
const path = require('path');

const tempHome = fs.mkdtempSync(path.join(os.tmpdir(), 'arrayview-private-tunnel-'));
const originalHome = process.env.HOME;
process.env.HOME = tempHome;
let resolverCalls = 0;
let promotionCalls = 0;

const vscodeMock = {
    env: {
        remoteName: 'tunnel',
        appHost: 'desktop',
        async asExternalUri(uri) {
            resolverCalls += 1;
            return { toString: () => String(uri) };
        },
    },
    Uri: { parse: value => ({ toString: () => value }) },
    commands: {
        async executeCommand() { promotionCalls += 1; },
    },
};

const httpMock = {
    get(_url, _options, callback) {
        const request = new EventEmitter();
        request.destroy = () => {};
        process.nextTick(() => {
            const response = new EventEmitter();
            response.statusCode = 200;
            response.setEncoding = () => {};
            response.resume = () => {};
            callback(response);
            response.emit('data', JSON.stringify({
                service: 'arrayview',
                instance_id: 'private-server',
            }));
            response.emit('end');
        });
        return request;
    },
};

const originalLoad = Module._load;
Module._load = function(request, parent, isMain) {
    if (request === 'vscode') return vscodeMock;
    if (request === 'http') return httpMock;
    return originalLoad.call(this, request, parent, isMain);
};
const { __test } = require('./extension');
Module._load = originalLoad;

(async () => {
    try {
        const tunnelResult = await __test.resolveRemoteViewerUrl(
            'http://localhost:8001/?sid=tunnel-private',
            'private-server'
        );
        assert.strictEqual(tunnelResult, null);
        assert.strictEqual(resolverCalls, 0, 'tunnel requests must not resolve a public URL');
        assert.strictEqual(promotionCalls, 0, 'tunnel requests must not invoke promotion commands');

        vscodeMock.env.remoteName = 'ssh-remote';
        const sshResult = await __test.resolveRemoteViewerUrl(
            'http://localhost:8002/?sid=remote-ssh',
            'private-server'
        );
        assert.strictEqual(sshResult, 'http://localhost:8002/?sid=remote-ssh');
        assert.strictEqual(resolverCalls, 1, 'Remote SSH keeps its existing resolver path');
        assert.strictEqual(promotionCalls, 0);
        console.log('private tunnel and Remote SSH routing tests passed');
    } finally {
        if (originalHome === undefined) delete process.env.HOME;
        else process.env.HOME = originalHome;
        fs.rmSync(tempHome, { recursive: true, force: true });
    }
})().catch(error => {
    console.error(error);
    process.exitCode = 1;
});
