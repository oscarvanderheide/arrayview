const assert = require('assert');
const fs = require('fs');
const http = require('http');
const Module = require('module');
const os = require('os');
const path = require('path');

const tempHome = fs.mkdtempSync(path.join(os.tmpdir(), 'arrayview-panel-'));
const originalHome = process.env.HOME;
process.env.HOME = tempHome;

let panelCount = 0;
const panels = [];
const panelOptions = [];

function makePanel() {
    const messageHandlers = [];
    const disposeHandlers = [];
    return {
        title: '',
        throwOnReveal: false,
        reveal() {
            if (this.throwOnReveal) throw new Error('stale panel');
        },
        webview: {
            html: '',
            onDidReceiveMessage(handler) {
                messageHandlers.push(handler);
                return { dispose() {} };
            },
            postMessage: async () => true,
        },
        onDidDispose(handler) {
            disposeHandlers.push(handler);
            return { dispose() {} };
        },
        triggerReady() {
            for (const handler of messageHandlers) {
                handler({ type: 'viewer-ready', phase: 'frame-rendered' });
            }
        },
        triggerDispose() {
            for (const handler of disposeHandlers) handler();
        },
    };
}
const vscodeMock = {
    env: { remoteName: null, uiKind: 1 },
    UIKind: { Web: 2 },
    ViewColumn: { Active: 1, Beside: 2 },
    window: {
        activeTextEditor: null,
        createWebviewPanel(viewType, title, column, options) {
            panelCount += 1;
            const panel = makePanel();
            panel.webview.options = options;
            panels.push(panel);
            panelOptions.push(options);
            return panel;
        },
    },
    workspace: {
        getConfiguration() {
            return { get: () => false };
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

(async () => {
    try {
        fs.mkdirSync(__test.signalDir, { recursive: true });
        const firstUrl = 'old-route.example/?sid=same-sid';
        const secondUrl = 'new-route.example/?sid=same-sid';
        const requestKey = 'request:same-request';

        const firstReady = __test.openInWebviewPanel(
            firstUrl, 'ArrayView', false, firstUrl, requestKey
        );
        const replayReady = __test.openInWebviewPanel(
            secondUrl, 'ArrayView', false, firstUrl, requestKey
        );

        assert.strictEqual(panelCount, 1, 'same request ID must own one logical panel');
        assert.strictEqual(__test._openPanels.get(requestKey), panels[0]);
        assert.match(panels[0].webview.html, /new-route\.example/);

        panels[0].triggerReady();
        assert.strictEqual(await firstReady, null);
        assert.strictEqual(await replayReady, null);

        const replacementKey = 'request:replacement-race';
        const oldPanelReady = __test.openInWebviewPanel(
            'replace-old/?sid=same-sid',
            'ArrayView',
            false,
            'replace-old/?sid=same-sid',
            replacementKey
        );
        panels[1].throwOnReveal = true;
        const replacementReady = __test.openInWebviewPanel(
            'replace-new/?sid=same-sid',
            'ArrayView',
            false,
            'replace-old/?sid=same-sid',
            replacementKey
        );
        assert.strictEqual(panelCount, 3);
        assert.strictEqual(__test._openPanels.get(replacementKey), panels[2]);

        panels[1].triggerReady();
        assert.strictEqual(await oldPanelReady, null);
        panels[1].triggerDispose();
        assert.strictEqual(
            __test._openPanels.get(replacementKey),
            panels[2],
            'disposing a superseded panel must not erase its replacement'
        );

        const sameReplacementReady = __test.openInWebviewPanel(
            'replace-new/?sid=same-sid',
            'ArrayView',
            false,
            'replace-old/?sid=same-sid',
            replacementKey
        );
        assert.strictEqual(
            panelCount,
            3,
            'same-ID replay after stale disposal must reuse the live replacement'
        );
        panels[2].triggerReady();
        assert.strictEqual(await replacementReady, null);
        assert.strictEqual(await sameReplacementReady, null);

        const originalSetTimeout = global.setTimeout;
        let backgroundHealthCheck = null;
        global.setTimeout = (callback, delay) => {
            if (delay === 3500) backgroundHealthCheck = callback;
            return 0;
        };
        let mappedReady;
        try {
            mappedReady = __test.openInWebviewPanel(
                'http://localhost:49123/?sid=mapped',
                'ArrayView',
                false,
                'http://localhost:9127/?sid=mapped',
                'request:mapped'
            );
        } finally {
            global.setTimeout = originalSetTimeout;
        }
        assert.deepStrictEqual(panelOptions[3].portMapping, [
            { webviewPort: 49123, extensionHostPort: 9127 },
        ]);
        assert.match(
            panels[3].webview.html,
            /http:\/\/localhost:49123\/\?sid=mapped/
        );
        assert.match(
            panels[3].webview.html,
            /http:\/\/localhost:49123\/ping/
        );
        const mappedReplayReady = __test.openInWebviewPanel(
            'http://localhost:49124/?sid=mapped&compare_sid=other',
            'ArrayView reconciled',
            false,
            'http://localhost:9127/?sid=mapped',
            'request:mapped'
        );
        assert.strictEqual(panelCount, 4);
        assert.match(
            panels[3].webview.html,
            /http:\/\/localhost:49124\/\?sid=mapped&compare_sid=other/
        );
        assert.deepStrictEqual(panels[3].webview.options.portMapping, [
            { webviewPort: 49124, extensionHostPort: 9127 },
        ]);

        const originalHttpGet = http.get;
        let requestedPing = null;
        http.get = (url, options, callback) => {
            requestedPing = url.toString();
            const request = {
                on() { return request; },
                destroy() {},
            };
            queueMicrotask(() => callback({
                statusCode: 200,
                setEncoding() {},
                on(event, handler) {
                    if (event === 'data') {
                        queueMicrotask(() => handler('{"service":"arrayview"}'));
                    } else if (event === 'end') {
                        queueMicrotask(handler);
                    }
                },
            }));
            return request;
        };
        try {
            await backgroundHealthCheck();
        } finally {
            http.get = originalHttpGet;
        }
        assert.strictEqual(requestedPing, 'http://localhost:9127/ping');
        panels[3].triggerReady();
        assert.strictEqual(await mappedReady, null);
        assert.strictEqual(await mappedReplayReady, null);

        const publicReady = __test.openInWebviewPanel(
            'https://public-9127.devtunnels.ms/?sid=public',
            'ArrayView public',
            false,
            'http://localhost:9127/?sid=public',
            'request:public'
        );
        assert.strictEqual(
            Object.hasOwn(panelOptions[4], 'portMapping'),
            false,
            'a non-loopback public authority must not be port-remapped'
        );
        assert.match(panels[4].webview.html, /public-9127\.devtunnels\.ms\/ping/);
        panels[4].triggerReady();
        assert.strictEqual(await publicReady, null);

        const invalidMappingReady = __test.openInWebviewPanel(
            'invalid-mapping.example/?sid=invalid',
            'ArrayView',
            false,
            'ftp://localhost:9128/?sid=invalid',
            'request:invalid-mapping'
        );
        assert.strictEqual(
            Object.hasOwn(panelOptions[5], 'portMapping'),
            false,
            'invalid backend URL must omit webview portMapping safely'
        );
        panels[5].triggerReady();
        assert.strictEqual(await invalidMappingReady, null);

        console.log('panel replay tests passed');
    } finally {
        if (originalHome === undefined) delete process.env.HOME;
        else process.env.HOME = originalHome;
        fs.rmSync(tempHome, { recursive: true, force: true });
    }
})().catch(error => {
    console.error(error);
    process.exitCode = 1;
});
