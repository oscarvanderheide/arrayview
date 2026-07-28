// MANUAL HOST-BOUNDARY TEST — not part of the sandbox component gate.
//
// Every assertion below describes the integrated-browser ("Simple Browser")
// handoff: two files producing two correlated viewer tabs. That path is now
// handoff. The direct path is enabled again only for a verified idle tunnel
// backend; concurrent viewers deliberately retain dedicated webviews. This old
// fixture binds a real listener and models two simultaneous integrated-browser
// opens, so its transport and expected policy are both unsuitable for the
// restricted component environment. Selection is covered without a listener
// by test_integrated_browser_selection.js; correlated readiness remains covered
// by test_integrated_browser_readiness.js on a host that permits TCP listeners.
console.log('integrated browser placeholder cleanup tests skipped '
    + '(legacy two-browser fixture; see selection/readiness tests)');
process.exit(0);

const assert = require('assert');
const { EventEmitter } = require('events');
const fs = require('fs');
const http = require('http');
const Module = require('module');
const os = require('os');
const path = require('path');

const tempHome = fs.mkdtempSync(
    path.join(os.tmpdir(), 'arrayview-placeholder-cleanup-')
);
const originalHome = process.env.HOME;
process.env.HOME = tempHome;

const browserOpens = [];
const launches = [];
let editorProvider = null;

function uriFor(filePath) {
    return {
        fsPath: filePath,
        toString() { return `file://${filePath}`; },
    };
}

function spawnedProcess() {
    const child = new EventEmitter();
    child.stdout = new EventEmitter();
    child.stderr = new EventEmitter();
    child.unref = () => {};
    process.nextTick(() => {
        child.stdout.emit('data', Buffer.from('http://localhost:8000/\n'));
    });
    return child;
}

const activeTabGroup = { activeTab: null };
const vscodeMock = {
    env: {
        remoteName: 'tunnel',
        appHost: 'desktop',
        uiKind: 1,
    },
    UIKind: { Web: 2 },
    ViewColumn: { Active: 1, Beside: 2 },
    Uri: {
        file: uriFor,
        parse(value) {
            return { toString() { return value; } };
        },
    },
    commands: {
        registerCommand() { return { dispose() {} }; },
        async getCommands() {
            return ['workbench.action.browser.open'];
        },
        async executeCommand(command, args) {
            assert.strictEqual(
                command,
                'workbench.action.browser.open',
                `unexpected VS Code command: ${command}`
            );
            browserOpens.push(args);
        },
    },
    window: {
        state: { focused: true },
        activeTextEditor: null,
        tabGroups: { activeTabGroup },
        registerCustomEditorProvider(_viewType, provider) {
            editorProvider = provider;
            return { dispose() {} };
        },
        showErrorMessage() {},
    },
    workspace: {
        workspaceFolders: [],
        getWorkspaceFolder() { return null; },
        getConfiguration(section) {
            return {
                get(name, fallback) {
                    if (
                        section === 'workbench.browser'
                        && name === 'enableRemoteProxy'
                    ) {
                        return true;
                    }
                    return fallback;
                },
            };
        },
    },
};

const originalLoad = Module._load;
Module._load = function(request, parent, isMain) {
    if (request === 'vscode') return vscodeMock;
    if (request === 'child_process') {
        const actual = originalLoad.call(this, request, parent, isMain);
        return {
            ...actual,
            spawn(command, args, options) {
                launches.push({ command, args, options });
                return spawnedProcess();
            },
        };
    }
    return originalLoad.call(this, request, parent, isMain);
};
const extension = require('./extension');
Module._load = originalLoad;
const { __test } = extension;

function placeholderPanel(filePath) {
    const disposeHandlers = [];
    const messageHandlers = [];
    const panel = {
        disposed: false,
        webview: {
            options: {},
            html: '',
            onDidReceiveMessage(handler) {
                messageHandlers.push(handler);
                setTimeout(() => {
                    handler({ type: 'viewer-ready', phase: 'frame-rendered' });
                }, 100);
                return { dispose() {} };
            },
        },
        onDidDispose(handler) {
            disposeHandlers.push(handler);
            return { dispose() {} };
        },
        dispose() {
            if (this.disposed) return;
            this.disposed = true;
            for (const handler of disposeHandlers) handler();
        },
        uri: uriFor(filePath),
    };
    return panel;
}

function requestData(port, requestId, filePath) {
    return {
        protocolVersion: 1,
        requestId,
        serverId: 'server-placeholder',
        windowId: 'window-placeholder',
        ackPath: path.join(
            __test.signalDir,
            `open-ack-v0100-${requestId}.json`
        ),
        sentAtMs: Date.now(),
        maxAgeMs: 30000,
        url: `http://localhost:${port}/?sid=${requestId}-sid`,
        title: path.basename(filePath),
        handoffPath: filePath,
    };
}

(async () => {
    const journals = new Map();
    const server = http.createServer((req, res) => {
        const parsed = new URL(req.url, 'http://localhost');
        if (parsed.pathname === '/ping') {
            res.writeHead(200, { 'content-type': 'application/json' });
            res.end(JSON.stringify({
                service: 'arrayview',
                instance_id: 'server-placeholder',
            }));
            return;
        }
        if (parsed.pathname.startsWith('/metadata/')) {
            res.writeHead(200, { 'content-type': 'application/json' });
            res.end('{}');
            return;
        }
        if (parsed.pathname.startsWith('/viewer-phase/')) {
            const requestId = decodeURIComponent(parsed.pathname.split('/').pop());
            const sid = decodeURIComponent(parsed.pathname.split('/').at(-2));
            if (req.method === 'POST') {
                let body = '';
                req.setEncoding('utf8');
                req.on('data', chunk => { body += chunk; });
                req.on('end', () => {
                    const prepared = JSON.parse(body);
                    journals.set(requestId, {
                        sid,
                        request_id: requestId,
                        window_id: prepared.window_id,
                        server_id: prepared.server_id,
                        token: prepared.token,
                        phases: [
                            'script-loaded',
                            'ws-open',
                            'metadata-loaded',
                            'frame-rendered',
                        ],
                        viewer_instance_ids: [`viewer-${requestId}`],
                    });
                    res.writeHead(200, { 'content-type': 'application/json' });
                    res.end(JSON.stringify(journals.get(requestId)));
                });
                return;
            }
            const journal = journals.get(requestId);
            if (
                !journal
                || parsed.searchParams.get('token') !== journal.token
            ) {
                res.writeHead(409);
                res.end();
                return;
            }
            res.writeHead(200, { 'content-type': 'application/json' });
            res.end(JSON.stringify(journal));
            return;
        }
        res.writeHead(404);
        res.end();
    });

    const context = {
        extension: { packageJSON: { version: 'test' } },
        environmentVariableCollection: {
            get() { return null; },
            replace() {},
        },
        subscriptions: [],
    };

    try {
        fs.mkdirSync(__test.signalDir, { recursive: true });
        extension.activate(context);
        assert(editorProvider, 'activation must register the custom editor provider');

        __test.setWindowId('window-placeholder');
        fs.writeFileSync(
            path.join(__test.signalDir, 'window-window-placeholder.json'),
            JSON.stringify({
                pid: process.pid,
                windowId: 'window-placeholder',
                extensionInstanceId: __test.extensionInstanceId,
            })
        );

        await new Promise(resolve => server.listen(0, 'localhost', resolve));
        const port = server.address().port;
        const firstPath = path.join(tempHome, 'first', 'same-name.npy');
        const secondPath = path.join(tempHome, 'second', 'same-name.npy');
        const firstPanel = placeholderPanel(firstPath);
        const secondPanel = placeholderPanel(secondPath);

        const realSetTimeout = global.setTimeout;
        global.setTimeout = (callback, delay, ...args) => {
            const timer = realSetTimeout(callback, delay, ...args);
            if (delay === 190000) timer.unref();
            return timer;
        };
        try {
            await Promise.all([
                editorProvider.resolveCustomEditor(
                    { uri: uriFor(firstPath) },
                    firstPanel
                ),
                editorProvider.resolveCustomEditor(
                    { uri: uriFor(secondPath) },
                    secondPanel
                ),
            ]);
        } finally {
            global.setTimeout = realSetTimeout;
        }

        assert.strictEqual(launches.length, 2);
        for (const launch of launches) {
            assert.strictEqual(launch.command, 'uv');
            const prefix = launch.args.slice(0, 10);
            assert.strictEqual(prefix[0], 'run');
            assert.strictEqual(prefix[1], '--directory');
            assert.strictEqual(prefix[3], '--no-project');
            assert.strictEqual(prefix[4], '--python');
            assert.strictEqual(prefix[5], '3.12');
            assert.strictEqual(prefix[6], '--with');
            assert.strictEqual(prefix[7], 'arrayview');
            assert.strictEqual(prefix[8], 'python');
            assert.strictEqual(prefix[9], '-m');
        }

        const first = requestData(port, 'request-first', firstPath);
        assert.strictEqual(__test.claimProtocolRequest(first), 'acquired');
        await __test._processSignalDataBody(first);
        assert.strictEqual(
            firstPanel.disposed,
            true,
            'the exact first custom-editor placeholder must close after its integrated-browser tab opens'
        );
        assert.strictEqual(
            secondPanel.disposed,
            false,
            'a same-named placeholder for another file must remain open'
        );

        const second = requestData(port, 'request-second', secondPath);
        assert.strictEqual(__test.claimProtocolRequest(second), 'acquired');
        await __test._processSignalDataBody(second);
        assert.strictEqual(
            secondPanel.disposed,
            true,
            'the second invocation must close only its own placeholder'
        );

        assert.strictEqual(
            browserOpens.length,
            2,
            'two file invocations must produce two viewer tabs, not placeholders plus viewers'
        );
        assert.deepStrictEqual(
            browserOpens.map(args => args.reuseUrlFilter),
            [
                '?_av_launch_request_id=request-first',
                '?_av_launch_request_id=request-second',
            ],
            'separate invocations must retain separate correlated viewer tabs'
        );
        assert.strictEqual(
            JSON.parse(fs.readFileSync(first.ackPath, 'utf8')).state,
            'backend_ready'
        );
        assert.strictEqual(
            JSON.parse(fs.readFileSync(second.ackPath, 'utf8')).state,
            'backend_ready'
        );

        console.log('integrated browser placeholder cleanup tests passed');
    } finally {
        for (const subscription of context.subscriptions.reverse()) {
            try { subscription.dispose(); } catch (_) {}
        }
        await new Promise(resolve => server.close(resolve));
        if (originalHome === undefined) delete process.env.HOME;
        else process.env.HOME = originalHome;
        fs.rmSync(tempHome, { recursive: true, force: true });
    }
})().catch(error => {
    console.error(error);
    process.exitCode = 1;
});
