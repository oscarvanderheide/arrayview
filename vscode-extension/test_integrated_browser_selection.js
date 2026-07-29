// Selection contract for the desktop-tunnel direct browser path.
//
// This deliberately mocks loopback HTTP instead of binding a listener: the
// execution sandbox cannot listen on a port, and the policy under test is the
// decision made from a verified /ping response, not Node's TCP stack.

const assert = require('assert');
const { EventEmitter } = require('events');
const fs = require('fs');
const Module = require('module');
const os = require('os');
const path = require('path');

const tempHome = fs.mkdtempSync(path.join(os.tmpdir(), 'arrayview-browser-select-'));
const originalHome = process.env.HOME;
process.env.HOME = tempHome;

let remoteProxyEnabled = true;
let browserCommandAvailable = true;
let httpCalls = 0;
let nextStatus = {
    service: 'arrayview',
    instance_id: 'server-selection',
    active_viewer_sockets: 0,
};

const httpMock = {
    get(_url, _options, callback) {
        httpCalls += 1;
        const request = new EventEmitter();
        request.destroy = () => {};
        process.nextTick(() => {
            if (nextStatus === null) {
                request.emit('error', new Error('unreachable'));
                return;
            }
            const response = new EventEmitter();
            response.statusCode = 200;
            response.setEncoding = () => {};
            response.resume = () => {};
            callback(response);
            response.emit('data', JSON.stringify(nextStatus));
            response.emit('end');
        });
        return request;
    },
};

const vscodeMock = {
    commands: {
        async getCommands() {
            return browserCommandAvailable
                ? ['workbench.action.browser.open']
                : [];
        },
    },
    env: {
        appHost: 'desktop',
        remoteName: 'tunnel',
        uiKind: 1,
    },
    UIKind: { Web: 2 },
    workspace: {
        getConfiguration(section) {
            assert.strictEqual(section, 'workbench.browser');
            return {
                get(name, fallback) {
                    assert.strictEqual(name, 'enableRemoteProxy');
                    return remoteProxyEnabled ?? fallback;
                },
            };
        },
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
        const backendUrl = 'http://localhost:8123/?sid=selection';

        // The proxy capability is load-bearing. Without it, localhost would
        // resolve on the user's desktop rather than in the tunnel workspace.
        remoteProxyEnabled = false;
        const beforeDisabled = httpCalls;
        assert.strictEqual(
            await __test.reserveDirectIntegratedBrowser(
                backendUrl,
                'server-selection'
            ),
            false
        );
        assert.strictEqual(httpCalls, beforeDisabled);

        // A verified backend is eligible for a request-scoped browser tab.
        remoteProxyEnabled = true;
        nextStatus = {
            service: 'arrayview',
            instance_id: 'server-selection',
            active_viewer_sockets: 0,
        };
        assert.strictEqual(
            await __test.reserveDirectIntegratedBrowser(
                backendUrl,
                'server-selection'
            ),
            true
        );

        // Existing viewers do not force later requests onto the public
        // webview route. Distinct request IDs provide distinct browser tabs.
        nextStatus = {
            service: 'arrayview',
            instance_id: 'server-selection',
            active_viewer_sockets: 5,
        };
        assert.strictEqual(
            await __test.reserveDirectIntegratedBrowser(
                backendUrl,
                'server-selection'
            ),
            true
        );

        // Viewer-count telemetry is not a routing capability and older
        // backends that omit it remain eligible after identity verification.
        nextStatus = {
            service: 'arrayview',
            instance_id: 'server-selection',
        };
        assert.strictEqual(
            await __test.reserveDirectIntegratedBrowser(
                backendUrl,
                'server-selection'
            ),
            true
        );

        // A compatible service on a reused port is not the requested backend.
        nextStatus = {
            service: 'arrayview',
            instance_id: 'another-server',
            active_viewer_sockets: 0,
        };
        assert.strictEqual(
            await __test.reserveDirectIntegratedBrowser(
                backendUrl,
                'server-selection'
            ),
            false
        );

        // The protocol identity is mandatory. A generic ArrayView answer is
        // insufficient because a recently reused port may now belong to a
        // different launch.
        nextStatus = {
            service: 'arrayview',
            instance_id: 'server-selection',
            active_viewer_sockets: 0,
        };
        const beforeMissingIdentity = httpCalls;
        assert.strictEqual(
            await __test.reserveDirectIntegratedBrowser(backendUrl),
            false
        );
        assert.strictEqual(httpCalls, beforeMissingIdentity);

        // Unknown ownership fails closed rather than exposing a public route.
        nextStatus = null;
        assert.strictEqual(
            await __test.reserveDirectIntegratedBrowser(
                backendUrl,
                'server-selection'
            ),
            false
        );

        // A missing built-in browser command also leaves the direct path idle.
        nextStatus = {
            service: 'arrayview',
            instance_id: 'server-selection',
            active_viewer_sockets: 0,
        };
        browserCommandAvailable = false;
        assert.strictEqual(
            await __test.reserveDirectIntegratedBrowser(
                backendUrl,
                'server-selection'
            ),
            false
        );

        let activeCommands = 0;
        let maxActiveCommands = 0;
        const commandOrder = [];
        await Promise.all(
            Array.from({ length: 5 }, (_, index) =>
                __test._runIntegratedBrowserCommand(async () => {
                    activeCommands += 1;
                    maxActiveCommands = Math.max(maxActiveCommands, activeCommands);
                    commandOrder.push(index);
                    await new Promise(resolve => setTimeout(resolve, 2));
                    activeCommands -= 1;
                })
            )
        );
        assert.strictEqual(maxActiveCommands, 1);
        assert.deepStrictEqual(commandOrder, [0, 1, 2, 3, 4]);
        await assert.rejects(
            __test._runIntegratedBrowserCommand(async () => {
                throw new Error('command failed');
            }),
            /command failed/
        );
        let ranAfterFailure = false;
        await __test._runIntegratedBrowserCommand(async () => {
            ranAfterFailure = true;
        });
        assert.strictEqual(
            ranAfterFailure,
            true,
            'one failed browser command must not poison later launches'
        );

        console.log('integrated browser selection tests passed');
    } finally {
        if (originalHome === undefined) delete process.env.HOME;
        else process.env.HOME = originalHome;
        fs.rmSync(tempHome, { recursive: true, force: true });
    }
})().catch(error => {
    console.error(error);
    process.exitCode = 1;
});
