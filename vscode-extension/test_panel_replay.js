const assert = require('assert');
const fs = require('fs');
const Module = require('module');
const os = require('os');
const path = require('path');

const tempHome = fs.mkdtempSync(path.join(os.tmpdir(), 'arrayview-panel-'));
const originalHome = process.env.HOME;
process.env.HOME = tempHome;

let panelCount = 0;
const panels = [];

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
        createWebviewPanel() {
            panelCount += 1;
            const panel = makePanel();
            panels.push(panel);
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
