const assert = require('assert');
const fs = require('fs');
const Module = require('module');
const os = require('os');
const path = require('path');

const tempHome = fs.mkdtempSync(path.join(os.tmpdir(), 'arrayview-reload-recovery-'));
const originalHome = process.env.HOME;
process.env.HOME = tempHome;

const vscodeMock = {
    env: { remoteName: 'tunnel' },
    window: {
        state: { focused: true },
        showErrorMessage: async () => undefined,
    },
    commands: { executeCommand: async () => undefined },
};
const originalLoad = Module._load;
Module._load = function(request, parent, isMain) {
    if (request === 'vscode') return vscodeMock;
    return originalLoad.call(this, request, parent, isMain);
};

const { __test } = require('./extension');
Module._load = originalLoad;

function writeJson(filePath, payload) {
    fs.writeFileSync(filePath, JSON.stringify(payload));
}

function requestData(requestId, windowId) {
    return {
        protocolVersion: 1,
        requestId,
        serverId: `server-${requestId}`,
        windowId,
        ackPath: path.join(
            __test.signalDir,
            `open-ack-v0100-${requestId}.json`
        ),
        sentAtMs: Date.now(),
        maxAgeMs: 60000,
        remoteOnly: true,
        targetRemoteName: 'tunnel',
        url: `http://localhost:8000/?sid=${requestId}`,
    };
}

function stageClaim(requestId, windowId) {
    __test.setWindowId(windowId);
    const data = requestData(requestId, windowId);
    assert.strictEqual(__test.claimProtocolRequest(data), 'acquired');
    assert.strictEqual(__test.writeProtocolAck(data, 'panel_opened'), true);

    const base = path.join(
        __test.signalDir,
        `open-request-ipc-${windowId}.request-${requestId}.json`
    );
    const claimed = `${base}.claimed-${process.pid}`;
    writeJson(claimed, data);
    return { base, claimed, data };
}

function stageReloadRecovery(requestId, windowId) {
    const { base, claimed, data } = stageClaim(requestId, windowId);
    const recovery = __test._writeReloadRecovery(claimed, base, data);

    assert(recovery, 'reload recovery must be persisted before the window reloads');
    assert.strictEqual(
        fs.existsSync(claimed),
        true,
        'the claimed request itself is the durable reload record'
    );
    assert.strictEqual(
        fs.existsSync(base),
        false,
        'the old host must not be able to reclaim it before reload'
    );
    assert.strictEqual(
        fs.existsSync(data.ackPath),
        true,
        'the terminal must keep waiting on the existing non-terminal ACK'
    );
    const ack = JSON.parse(fs.readFileSync(data.ackPath, 'utf8'));
    ack.claimOwner.pid = 99999999;
    writeJson(data.ackPath, ack);
    return { base, claimed, data, recovery };
}

(async () => {
try {
    fs.mkdirSync(__test.signalDir, { recursive: true });

    const exact = new Error(
        'Integrated browser did not start the viewer script before recovery timeout'
    );
    exact.code = 'ARRAYVIEW_INTEGRATED_BROWSER_NO_NAVIGATION';
    exact.arrayviewIntegratedBrowserOpened = true;
    exact.arrayviewRetainSession = true;
    assert.strictEqual(
        __test._isIntegratedBrowserNavigationWedge(exact),
        true,
        'the observed healthy-backend blank-tab failure must offer reload recovery'
    );

    const noOpenedBrowser = new Error(exact.message);
    noOpenedBrowser.arrayviewRetainSession = true;
    assert.strictEqual(
        __test._isIntegratedBrowserNavigationWedge(noOpenedBrowser),
        false,
        'the no-script wording alone is not proof that VS Code opened the browser tab'
    );

    const alreadyReleased = new Error(exact.message);
    alreadyReleased.arrayviewIntegratedBrowserOpened = true;
    alreadyReleased.arrayviewRetainSession = false;
    assert.strictEqual(
        __test._isIntegratedBrowserNavigationWedge(alreadyReleased),
        false,
        'a request whose sessions cannot survive reload must remain a normal failure'
    );

    const renderFailure = new Error('Viewer did not render a frame');
    renderFailure.arrayviewIntegratedBrowserOpened = true;
    renderFailure.arrayviewRetainSession = true;
    assert.strictEqual(
        __test._isIntegratedBrowserNavigationWedge(renderFailure),
        false,
        'post-navigation rendering failures must not reload the VS Code window'
    );

    const enoughTime = requestData('enough-time', 'window-before-reload');
    assert.strictEqual(__test._hasReloadRecoveryBudget(enoughTime), true);
    const almostExpired = requestData('almost-expired', 'window-before-reload');
    almostExpired.sentAtMs = Date.now() - 50000;
    assert.strictEqual(
        __test._hasReloadRecoveryBudget(almostExpired),
        false,
        'do not reload when the replacement window cannot finish before expiry'
    );

    const predecessor = stageReloadRecovery('recover-me', 'window-before-reload');
    const sibling = stageReloadRecovery('leave-me', 'window-sibling');

    const replacementBase = path.join(
        __test.signalDir,
        'open-request-ipc-window-after-reload.json'
    );
    writeJson(path.join(__test.signalDir, 'window-window-after-reload.json'), {
        pid: process.pid,
        windowId: 'window-after-reload',
        extensionInstanceId: __test.extensionInstanceId,
        remoteName: 'tunnel',
        supersedes: ['window-before-reload'],
    });
    __test.setWindowId('window-after-reload');
    __test.setTargetedSignalFile(replacementBase);
    const resumed = __test._resumeReloadRecoveries(
        'window-after-reload',
        ['window-before-reload']
    );
    assert.strictEqual(
        resumed,
        1,
        'the replacement host must resume exactly its superseded window request'
    );

    const replacementQueue = replacementBase.replace(
        /\.json$/,
        `.request-${predecessor.data.requestId}.json`
    );
    assert.strictEqual(fs.existsSync(predecessor.claimed), false);
    assert.strictEqual(
        fs.existsSync(replacementQueue),
        true,
        'the surviving launch must move to the replacement window queue'
    );
    assert.strictEqual(
        fs.existsSync(sibling.claimed),
        true,
        'reload recovery must not steal a live sibling window request'
    );
    assert.strictEqual(
        fs.existsSync(
            replacementBase.replace(/\.json$/, `.request-${sibling.data.requestId}.json`)
        ),
        false
    );
    assert.strictEqual(
        __test._resumeReloadRecoveries(
            'window-after-reload',
            ['window-before-reload']
        ),
        0,
        'recovery must be idempotent and never open the same launch twice'
    );
    const resumedData = JSON.parse(fs.readFileSync(replacementQueue, 'utf8'));
    assert.strictEqual(__test.claimProtocolRequest(resumedData), 'acquired');
    const resumedAck = JSON.parse(fs.readFileSync(resumedData.ackPath, 'utf8'));
    assert.strictEqual(
        resumedAck.windowId,
        'window-before-reload',
        'the original terminal correlation must survive the new extension owner'
    );
    assert.strictEqual(resumedAck.claimOwner.windowId, 'window-after-reload');

    const rejected = stageClaim('reload-rejected', 'window-rejected');
    vscodeMock.commands.executeCommand = async () => {
        throw new Error('reload rejected');
    };
    assert.strictEqual(
        await __test._executeReloadRecovery(
            rejected.claimed, rejected.base, rejected.data
        ),
        false
    );
    assert.strictEqual(
        JSON.parse(fs.readFileSync(rejected.data.ackPath, 'utf8')).state,
        'failed',
        'a rejected reload must terminate the request instead of leaking it'
    );

    const writeFailure = stageClaim('recovery-write-failed', 'window-write-failed');
    const originalRename = fs.renameSync;
    let failPreserveRename = true;
    fs.renameSync = function(source, destination) {
        if (failPreserveRename && path.resolve(destination) === path.resolve(writeFailure.claimed)) {
            failPreserveRename = false;
            throw new Error('injected persistence failure');
        }
        return originalRename.call(this, source, destination);
    };
    try {
        assert.strictEqual(
            await __test._executeReloadRecovery(
                writeFailure.claimed, writeFailure.base, writeFailure.data
            ),
            false
        );
    } finally {
        fs.renameSync = originalRename;
    }
    assert.strictEqual(
        JSON.parse(fs.readFileSync(writeFailure.data.ackPath, 'utf8')).state,
        'failed',
        'a persistence error must become a terminal failure'
    );
} finally {
    process.env.HOME = originalHome;
    fs.rmSync(tempHome, { recursive: true, force: true });
}
})().catch(error => {
    console.error(error);
    process.exitCode = 1;
});
