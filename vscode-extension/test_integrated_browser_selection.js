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
        __test.setIntegratedBrowserState('idle');
        const beforeDisabled = httpCalls;
        assert.strictEqual(
            await __test.reserveDirectIntegratedBrowser(
                backendUrl,
                'server-selection'
            ),
            false
        );
        assert.strictEqual(httpCalls, beforeDisabled);
        assert.strictEqual(__test.integratedBrowserState(), 'idle');

        // With a verified idle backend, reserve the one reusable browser tab.
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
        assert.strictEqual(
            __test.integratedBrowserState(),
            'idle',
            'selection alone must not leave a reservation if later setup fails'
        );

        // A second request arriving while browser navigation is pending must
        // not reuse and replace that in-flight tab.
        __test.setIntegratedBrowserState('pending');
        const beforePending = httpCalls;
        assert.strictEqual(
            await __test.reserveDirectIntegratedBrowser(
                backendUrl,
                'server-selection'
            ),
            false
        );
        assert.strictEqual(httpCalls, beforePending);

        // An active viewer likewise keeps its tab. This is the historical
        // multi-viewer failure that disabled the direct path globally.
        __test.setIntegratedBrowserState('active');
        nextStatus = {
            service: 'arrayview',
            instance_id: 'server-selection',
            active_viewer_sockets: 1,
        };
        assert.strictEqual(
            await __test.reserveDirectIntegratedBrowser(
                backendUrl,
                'server-selection'
            ),
            false
        );
        assert.strictEqual(__test.integratedBrowserState(), 'active');

        // After tab close the backend count returns to zero, allowing the next
        // sequential call to reserve the direct path again.
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
        assert.strictEqual(__test.integratedBrowserState(), 'active');

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

        // Older or malformed ping responses without ownership evidence cannot
        // be interpreted as "zero viewers".
        nextStatus = {
            service: 'arrayview',
            instance_id: 'server-selection',
        };
        assert.strictEqual(
            await __test.reserveDirectIntegratedBrowser(
                backendUrl,
                'server-selection'
            ),
            false
        );

        // Unknown ownership fails closed to the dedicated webview.
        __test.setIntegratedBrowserState('active');
        nextStatus = null;
        assert.strictEqual(
            await __test.reserveDirectIntegratedBrowser(
                backendUrl,
                'server-selection'
            ),
            false
        );
        assert.strictEqual(__test.integratedBrowserState(), 'active');

        // A missing built-in browser command also leaves the direct path idle.
        __test.setIntegratedBrowserState('idle');
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
        assert.strictEqual(__test.integratedBrowserState(), 'idle');

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
