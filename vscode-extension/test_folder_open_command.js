// Contract for opening a FOLDER from the VS Code Explorer.
//
// A folder is not a file, so none of the existing delivery machinery reaches it:
// `customEditors` selectors only ever match filenames, and VS Code never opens a
// directory in an editor at all. The only host mechanism that can start this is
// a command contributed to `explorer/context` under `explorerResourceIsFolder`,
// which is what package.json now declares. This file pins the extension-side
// half of that contract:
//
//   1. the command exists and is registered at activation;
//   2. it creates its own placeholder tab, keyed by the resolved folder path, so
//      the signal-file handoff navigates that tab instead of opening a second
//      one — without this the user stares at nothing while the backend walks a
//      DICOM folder on a network mount, which is the slowest thing ArrayView
//      does and precisely the case this feature exists for;
//   3. it launches the folder path unchanged, with NO --stack. `--stack` globs
//      every supported file below the folder into one case each, which is right
//      for a folder of arrays and badly wrong for a folder of DICOM slices
//      (measured: a 6-slice series becomes 6 identical 3D cases). Only Python
//      can tell those apart, so the extension must not guess here.

const assert = require('assert');
const { EventEmitter } = require('events');
const fs = require('fs');
const Module = require('module');
const os = require('os');
const path = require('path');

const tempHome = fs.mkdtempSync(path.join(os.tmpdir(), 'arrayview-folder-open-'));
process.env.HOME = tempHome;
fs.mkdirSync(path.join(tempHome, '.arrayview'), { recursive: true });

const folderPath = path.join(tempHome, 'dicom_case');
fs.mkdirSync(folderPath, { recursive: true });

const launches = [];
const createdPanels = [];
const registeredCommands = new Map();
const shownErrors = [];

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

function fakePanel(title) {
    const disposeHandlers = [];
    const panel = {
        title,
        disposed: false,
        webview: {
            options: {},
            html: '',
            onDidReceiveMessage() { return { dispose() {} }; },
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
    };
    createdPanels.push(panel);
    return panel;
}

const vscodeMock = {
    env: { remoteName: 'tunnel', appHost: 'desktop', uiKind: 1 },
    UIKind: { Web: 2 },
    ViewColumn: { Active: 1, Beside: 2 },
    Uri: {
        file: (p) => ({ fsPath: p, toString() { return `file://${p}`; } }),
        parse: (value) => ({ toString() { return value; } }),
    },
    commands: {
        registerCommand(id, handler) {
            registeredCommands.set(id, handler);
            return { dispose() {} };
        },
        async getCommands() { return []; },
        async executeCommand() { return undefined; },
    },
    window: {
        state: { focused: true },
        activeTextEditor: null,
        tabGroups: { activeTabGroup: { activeTab: null } },
        createWebviewPanel(_viewType, title) { return fakePanel(title); },
        registerCustomEditorProvider() { return { dispose() {} }; },
        showErrorMessage(message) { shownErrors.push(message); },
        showWarningMessage() { return { then() {} }; },
        async showOpenDialog() { return undefined; },
    },
    workspace: {
        workspaceFolders: [],
        getWorkspaceFolder() { return null; },
        getConfiguration() {
            return { get(_name, fallback) { return fallback; }, async update() {} };
        },
    },
};

// The daemon fast path targets a hardcoded localhost:8000. Whether some real
// ArrayView happens to hold that port on the machine running this test must not
// decide what the test observes, so no HTTP ever succeeds here and the launch
// falls through to the spawn path this file inspects.
function refusingRequest() {
    const req = new EventEmitter();
    req.setTimeout = () => req;
    req.destroy = () => {};
    req.end = () => {};
    req.write = () => {};
    process.nextTick(() => req.emit('error', new Error('ECONNREFUSED (stubbed)')));
    return req;
}

const originalLoad = Module._load;
Module._load = function (request, parent, isMain) {
    if (request === 'vscode') return vscodeMock;
    if (request === 'http' || request === 'https') {
        const actual = originalLoad.call(this, request, parent, isMain);
        return { ...actual, get: refusingRequest, request: refusingRequest };
    }
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

(async () => {
    const context = {
        extension: { packageJSON: { version: 'test' } },
        environmentVariableCollection: { get() { return null; }, replace() {} },
        subscriptions: [],
    };
    extension.activate(context);

    // 1. the Explorer entry point exists
    const openFolder = registeredCommands.get('arrayview.openFolder');
    assert.ok(openFolder, 'arrayview.openFolder must be registered at activation');

    await openFolder({ fsPath: folderPath });

    // 2. one placeholder tab, correlated by resolved folder path
    assert.strictEqual(
        createdPanels.length,
        1,
        `expected exactly one placeholder tab, got ${createdPanels.length}`
    );
    const panel = createdPanels[0];
    assert.strictEqual(panel.title, 'dicom_case');
    assert.ok(
        /Opening dicom_case in ArrayView/.test(panel.webview.html),
        `placeholder must show progress, got: ${panel.webview.html}`
    );
    const placeholder = __test.pendingPlaceholders.get(path.resolve(folderPath));
    assert.ok(placeholder, 'folder must be registered as a pending placeholder');
    assert.strictEqual(placeholder.panel, panel);
    assert.strictEqual(placeholder.filePath, path.resolve(folderPath));

    // 3. the folder path is launched verbatim, and the mode stays Python's call
    assert.strictEqual(launches.length, 1, 'expected exactly one launch');
    const { args } = launches[0];
    assert.ok(
        args.includes(folderPath),
        `launch args must carry the folder path: ${JSON.stringify(args)}`
    );
    assert.deepStrictEqual(
        args.slice(args.indexOf(folderPath)),
        [folderPath, '--window', 'vscode', '--name', 'dicom_case'],
        `unexpected launch tail: ${JSON.stringify(args)}`
    );
    assert.ok(
        !args.includes('--stack'),
        '--stack is wrong for a DICOM folder; Python must classify the directory'
    );
    assert.strictEqual(
        launches[0].options.env.ARRAYVIEW_HANDOFF_PATH,
        folderPath,
        'handoff correlation needs the folder path in the child environment'
    );
    assert.deepStrictEqual(shownErrors, []);

    // 4. closing the tab drops the correlation, so a later signal cannot
    //    navigate a disposed panel
    panel.dispose();
    assert.strictEqual(
        __test.pendingPlaceholders.get(path.resolve(folderPath)),
        undefined,
        'disposing the placeholder must drop its pending entry'
    );

    console.log('folder open command tests passed');
    process.exit(0);
})().catch((error) => {
    console.error(error);
    process.exit(1);
});
