const assert = require('assert');
const fs = require('fs');
const Module = require('module');
const os = require('os');
const path = require('path');

const tempHome = fs.mkdtempSync(path.join(os.tmpdir(), 'arrayview-panel-'));
const originalHome = process.env.HOME;
process.env.HOME = tempHome;

let panelCount = 0;
const messageHandlers = [];
const panel = {
    title: '',
    reveal() {},
    webview: {
        html: '',
        onDidReceiveMessage(handler) {
            messageHandlers.push(handler);
            return { dispose() {} };
        },
        postMessage: async () => true,
    },
    onDidDispose() { return { dispose() {} }; },
};
const vscodeMock = {
    env: { remoteName: null, uiKind: 1 },
    UIKind: { Web: 2 },
    ViewColumn: { Active: 1, Beside: 2 },
    window: {
        activeTextEditor: null,
        createWebviewPanel() {
            panelCount += 1;
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
        assert.strictEqual(__test._openPanels.get(requestKey), panel);
        assert.match(panel.webview.html, /new-route\.example/);

        for (const handler of messageHandlers) {
            handler({ type: 'viewer-ready', phase: 'frame-rendered' });
        }
        assert.strictEqual(await firstReady, null);
        assert.strictEqual(await replayReady, null);

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
