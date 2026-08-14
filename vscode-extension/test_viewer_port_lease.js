const assert = require('assert');
const { EventEmitter } = require('events');
const fs = require('fs');
const Module = require('module');
const os = require('os');
const path = require('path');

const tempHome = fs.mkdtempSync(path.join(os.tmpdir(), 'arrayview-viewer-port-'));
const originalHome = process.env.HOME;
process.env.HOME = tempHome;

const requests = [];
const notifications = [];
let responsePayload = { port: 43123, reused: true, requestId: 'request-1' };
const vscodeMock = {
    env: { remoteName: 'tunnel' },
    window: {
        showErrorMessage(message) {
            notifications.push(message);
            return Promise.resolve(undefined);
        },
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
                instance_id: 'server-1',
            }));
            response.emit('end');
        });
        return request;
    },
    request(url, options, callback) {
        const request = new EventEmitter();
        let body = '';
        request.write = chunk => { body += String(chunk); };
        request.end = chunk => {
            if (chunk) body += String(chunk);
            requests.push({
                url: String(url),
                options,
                body: body ? JSON.parse(body) : {},
            });
            const pathname = new URL(String(url)).pathname;
            const payload = pathname === '/load'
                ? { sid: 'loaded-session' }
                : pathname.startsWith('/release/')
                    ? {}
                    : responsePayload;
            const response = new EventEmitter();
            response.statusCode = 200;
            response.setEncoding = () => {};
            response.resume = () => {};
            process.nextTick(() => {
                callback(response);
                response.emit('data', JSON.stringify(payload));
                response.emit('end');
            });
        };
        request.destroy = () => {};
        request.setTimeout = () => {};
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
        const port = await __test._coldStartPort(
            8000, 'request-1', 'server-1', 12345
        );
        assert.deepStrictEqual(port, { port: 43123, error: null });
        assert.deepStrictEqual(requests[0].body, {
            requestId: 'request-1',
            expectedServerId: 'server-1',
            ttlMs: 12345,
        });
        assert.strictEqual(requests[0].url, 'http://localhost:8000/cold-start-port');

        responsePayload = { port: null };
        assert.deepStrictEqual(
            await __test._coldStartPort(8000, 'request-2', 'server-1', 12345),
            {
                port: null,
                error: 'ArrayView could not prepare a private connection for this viewer.',
            },
            'a missing private viewer port must not fall back to the stale main port'
        );

        responsePayload = { port: 43124, reused: false };
        const oldBackend = await __test._coldStartPort(
            8000, 'request-legacy-backend', 'server-1', 12345
        );
        assert.strictEqual(oldBackend.port, null);
        assert.match(oldBackend.error, /restart once/);

        vscodeMock.env.remoteName = 'ssh-remote';
        assert.deepStrictEqual(
            await __test._coldStartPort(8000, 'request-3', 'server-1', 12345),
            { port: 8000, error: null },
            'Remote SSH must keep its existing main-port path'
        );
        assert.strictEqual(requests.length, 3);

        vscodeMock.env.remoteName = 'tunnel';
        responsePayload = { port: 43125, reused: false };
        let placeholderError = null;
        await __test._launchWithStatusProgress(
            '/tmp/arrayview-port-failure.npy',
            'arrayview-port-failure.npy',
            'LEASE-FAILURE',
            { reportError(error) { placeholderError = error; } }
        );
        await new Promise(resolve => setTimeout(resolve, 10));
        assert(placeholderError, 'the placeholder must receive the failure immediately');
        assert.match(placeholderError.message, /restart once/);
        assert.strictEqual(
            __test.pendingLaunchProgress.has(
                path.resolve('/tmp/arrayview-port-failure.npy')
            ),
            false,
            'the opening spinner must settle immediately'
        );
        assert(
            requests.some(request => new URL(request.url).pathname.startsWith('/release/')),
            'the failed fast load must release its session'
        );
        assert.strictEqual(
            fs.existsSync(__test.signalDir)
                && fs.readdirSync(__test.signalDir).some(
                    name => name.startsWith('open-request-')
                ),
            false,
            'a failed port lease must not write a viewer signal'
        );
        assert(
            notifications.some(message => message.includes('restart once')),
            'the failure must also be visible when no placeholder is present'
        );
        console.log('viewer port lease request tests passed');
    } finally {
        if (originalHome === undefined) delete process.env.HOME;
        else process.env.HOME = originalHome;
        fs.rmSync(tempHome, { recursive: true, force: true });
    }
})().catch(error => {
    console.error(error);
    process.exitCode = 1;
});
