const assert = require('assert');
const fs = require('fs');
const Module = require('module');
const os = require('os');
const path = require('path');
const { spawn } = require('child_process');

const tempHome = fs.mkdtempSync(path.join(os.tmpdir(), 'arrayview-journal-'));
const originalHome = process.env.HOME;
process.env.HOME = tempHome;

const vscodeMock = {
    env: { remoteName: null },
    window: { state: { focused: false } },
};
const originalLoad = Module._load;
Module._load = function(request, parent, isMain) {
    if (request === 'vscode') return vscodeMock;
    return originalLoad.call(this, request, parent, isMain);
};

const { __test } = require('./extension');
Module._load = originalLoad;

function requestData(requestId) {
    return {
        protocolVersion: 1,
        requestId,
        serverId: `server-${requestId}`,
        ackPath: path.join(
            __test.signalDir,
            `open-ack-v0100-${requestId}.json`
        ),
        sentAtMs: Date.now(),
        maxAgeMs: 60000,
        url: `http://localhost:8000/?sid=${requestId}`,
    };
}

function writeJson(filePath, payload) {
    fs.writeFileSync(filePath, JSON.stringify(payload));
}

function readJson(filePath) {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

(async () => {
try {
    fs.mkdirSync(__test.signalDir, { recursive: true });
    __test.setWindowId('window-current');
    writeJson(path.join(__test.signalDir, 'window-window-current.json'), {
        pid: process.pid,
        extensionInstanceId: __test.extensionInstanceId,
    });

    const current = requestData('current');
    assert.strictEqual(__test.claimProtocolRequest(current), 'acquired');
    let ack = readJson(current.ackPath);
    const currentOwner = ack.claimOwner;
    assert.strictEqual(ack.state, 'claimed');
    assert.strictEqual(ack.claimOwner.pid, process.pid);
    assert.strictEqual(ack.claimOwner.windowId, 'window-current');
    assert.strictEqual(ack.claimOwner.extensionInstanceId, __test.extensionInstanceId);
    assert.strictEqual(typeof ack.claimOwner.claimToken, 'string');
    assert.strictEqual(__test.writeProtocolAck(current, 'panel_opened'), true);
    ack = readJson(current.ackPath);
    assert.strictEqual(ack.state, 'panel_opened');
    assert.strictEqual(
        ack.claimOwner.extensionInstanceId,
        __test.extensionInstanceId,
        'progress ACKs must preserve the exact claim owner'
    );
    assert.strictEqual(
        __test.claimProtocolRequest(requestData('current')),
        'duplicate',
        'a live owner must not be replayed by another poll'
    );

    const atomic = requestData('atomic');
    const reader = spawn(process.execPath, ['-e', `
        const fs = require('fs');
        const ackPath = process.argv[1];
        const deadline = Date.now() + 500;
        let invalid = false;
        while (Date.now() < deadline) {
            try { JSON.parse(fs.readFileSync(ackPath, 'utf8')); }
            catch (error) { if (error.code !== 'ENOENT') invalid = true; }
        }
        process.stdout.write(invalid ? 'invalid' : 'ok');
    `, atomic.ackPath], { stdio: ['ignore', 'pipe', 'inherit'] });
    let readerOutput = '';
    reader.stdout.on('data', chunk => { readerOutput += chunk.toString(); });
    for (let attempt = 0; attempt < 100; attempt++) {
        const freshAtomic = requestData('atomic');
        assert.strictEqual(__test.claimProtocolRequest(freshAtomic), 'acquired');
        fs.unlinkSync(atomic.ackPath);
        await new Promise(resolve => setImmediate(resolve));
    }
    await new Promise((resolve, reject) => {
        reader.on('error', reject);
        reader.on('exit', code => code === 0 ? resolve() : reject(new Error(`reader exited ${code}`)));
    });
    assert.strictEqual(
        readerOutput,
        'ok',
        'claim publication is atomic to the Python ACK reader'
    );

    const unknown = requestData('unknown');
    writeJson(unknown.ackPath, {
        protocolVersion: 1,
        state: 'panel_opened',
        requestId: unknown.requestId,
        claimOwner: {
            pid: process.pid,
            windowId: 'window-unknown',
            extensionInstanceId: 'old-extension-instance',
            claimToken: 'old-claim-token',
        },
    });
    writeJson(path.join(__test.signalDir, 'window-window-unknown.json'), {
        pid: process.pid,
        extensionInstanceId: 'replacement-extension-instance',
    });
    assert.strictEqual(
        __test.claimProtocolRequest(unknown),
        'retry',
        'a live but mismatched owner is unknown, not proof of death'
    );

    const malformedLock = path.join(__test.signalDir, 'malformed.lock');
    fs.writeFileSync(malformedLock, '');
    assert.strictEqual(
        __test._acquireAckLock(malformedLock, currentOwner),
        false,
        'an unreadable fresh lock is busy, not proof that its owner died'
    );
    assert.strictEqual(fs.existsSync(malformedLock), true);
    fs.unlinkSync(malformedLock);

    const fencedRegistration = path.join(
        __test.signalDir,
        'window-registration-race.json'
    );
    const replacementRegistration = {
        pid: process.pid,
        windowId: 'registration-race',
        extensionInstanceId: 'new-instance',
    };
    writeJson(fencedRegistration, replacementRegistration);
    assert.strictEqual(
        __test._removeRegistrationIfOwned(fencedRegistration, {
            ...replacementRegistration,
            extensionInstanceId: 'old-instance',
        }),
        false
    );
    assert.deepStrictEqual(readJson(fencedRegistration), replacementRegistration);
    assert.strictEqual(
        __test._removeRegistrationIfOwned(
            fencedRegistration,
            replacementRegistration
        ),
        true
    );

    const forwardedOne = __test._targetedSignalPath(
        'same-window',
        requestData('forward-one')
    );
    const forwardedTwo = __test._targetedSignalPath(
        'same-window',
        requestData('forward-two')
    );
    assert.notStrictEqual(forwardedOne, forwardedTwo);
    assert.match(forwardedOne, /\.request-forward-one\.json$/);
    assert.match(forwardedTwo, /\.request-forward-two\.json$/);

    const stale = requestData('stale');
    const staleOwner = {
        pid: 99999999,
        windowId: 'window-stale',
        extensionInstanceId: 'old-extension-instance',
        claimToken: 'old-claim-token',
    };
    writeJson(stale.ackPath, {
        protocolVersion: 1,
        state: 'panel_opened',
        requestId: stale.requestId,
        claimOwner: staleOwner,
    });
    writeJson(path.join(__test.signalDir, 'window-window-stale.json'), {
        pid: 99999999,
        extensionInstanceId: 'old-extension-instance',
    });
    assert.strictEqual(
        __test.claimProtocolRequest(stale),
        'acquired',
        'a restarted extension host must take over the same request ID'
    );
    ack = readJson(stale.ackPath);
    assert.strictEqual(ack.state, 'claimed');
    assert.strictEqual(ack.claimOwner.extensionInstanceId, __test.extensionInstanceId);
    assert.strictEqual(
        __test.writeProtocolAck(
            { ...stale, __claimOwner: staleOwner },
            'backend_ready'
        ),
        false,
        'a stale owner must be fenced after takeover'
    );
    assert.strictEqual(readJson(stale.ackPath).state, 'claimed');

    const terminal = requestData('terminal');
    writeJson(terminal.ackPath, {
        protocolVersion: 1,
        state: 'backend_ready',
        requestId: terminal.requestId,
    });
    assert.strictEqual(
        __test.claimProtocolRequest(terminal),
        'duplicate',
        'terminal requests must never be replayed'
    );

    const retryable = requestData('retryable');
    const retryableQueue = path.join(
        __test.signalDir,
        'open-request-v0900.request-retryable.json'
    );
    const retryableJournal = `${retryableQueue}.claimed-${process.pid}`;
    writeJson(retryableJournal, retryable);
    assert.strictEqual(__test.claimProtocolRequest(retryable), 'acquired');
    assert.strictEqual(
        __test._requeueOwnedClaim(retryableJournal, retryableQueue, retryable),
        true,
        'a live owner must make a non-terminal write failure retryable without reload'
    );
    assert.strictEqual(fs.existsSync(retryableJournal), false);
    assert.strictEqual(fs.existsSync(retryableQueue), true);
    assert.strictEqual(fs.existsSync(retryable.ackPath), false);

    const activeJournal = path.join(
        __test.signalDir,
        `open-request-v0900.request-active.json.claimed-${process.pid}`
    );
    const active = requestData('active');
    writeJson(activeJournal, active);
    writeJson(active.ackPath, {
        protocolVersion: 1,
        state: 'port_resolved',
        requestId: active.requestId,
        claimOwner: {
            ...currentOwner,
        },
    });
    __test.cleanupStaleFiles();
    assert.strictEqual(
        fs.existsSync(activeJournal),
        true,
        'cleanup must not steal a request from any live exact owner'
    );

    const interruptedJournal = path.join(
        __test.signalDir,
        'open-request-v0900.request-interrupted.json.claimed-99999999'
    );
    const interruptedQueue = interruptedJournal.replace(/\.claimed-\d+$/, '');
    const interrupted = requestData('interrupted');
    writeJson(interruptedJournal, interrupted);
    writeJson(interrupted.ackPath, {
        protocolVersion: 1,
        state: 'panel_opened',
        requestId: interrupted.requestId,
        claimOwner: {
            pid: 99999999,
            windowId: 'window-dead',
            extensionInstanceId: 'dead-extension-instance',
        },
    });
    writeJson(path.join(__test.signalDir, 'window-window-dead.json'), {
        pid: 99999999,
        extensionInstanceId: 'dead-extension-instance',
    });
    __test.setWindowId('window-dead');
    writeJson(path.join(__test.signalDir, 'window-window-dead.json'), {
        pid: process.pid,
        extensionInstanceId: __test.extensionInstanceId,
    });
    __test.cleanupStaleFiles();
    __test.setWindowId('window-current');
    assert.strictEqual(fs.existsSync(interruptedJournal), false);
    assert.strictEqual(
        fs.existsSync(interruptedQueue),
        true,
        'an interrupted non-terminal request must return to the queue'
    );

    const corruptJournal = path.join(
        __test.signalDir,
        'open-request-v0900.request-corrupt.json.claimed-99999999'
    );
    const corruptQueue = corruptJournal.replace(/\.claimed-\d+$/, '');
    const corrupt = requestData('corrupt');
    writeJson(corruptJournal, corrupt);
    fs.writeFileSync(corrupt.ackPath, '{incomplete');
    __test.cleanupStaleFiles();
    assert.strictEqual(fs.existsSync(corruptJournal), false);
    assert.strictEqual(fs.existsSync(corruptQueue), true);
    assert.strictEqual(
        fs.existsSync(corrupt.ackPath),
        false,
        'a corrupt ACK from a dead owner must not make the sole journal disposable'
    );

    const terminalJournal = path.join(
        __test.signalDir,
        `open-request-v0900.request-done.json.claimed-${process.pid}`
    );
    const done = requestData('done');
    writeJson(terminalJournal, done);
    writeJson(done.ackPath, {
        protocolVersion: 1,
        state: 'failed',
        requestId: done.requestId,
    });
    __test.cleanupStaleFiles();
    assert.strictEqual(
        fs.existsSync(terminalJournal),
        false,
        'terminal journals must be cleaned without replay'
    );

    const expiredJournal = path.join(
        __test.signalDir,
        `open-request-v0900.request-expired.json.claimed-${process.pid}`
    );
    const expired = {
        ...requestData('expired'),
        sentAtMs: Date.now() - 10000,
        maxAgeMs: 1,
    };
    writeJson(expiredJournal, expired);
    writeJson(expired.ackPath, {
        protocolVersion: 1,
        state: 'panel_opened',
        requestId: expired.requestId,
        windowId: 'window-current',
        claimOwner: currentOwner,
    });
    __test.cleanupStaleFiles();
    assert.strictEqual(fs.existsSync(expiredJournal), false);
    assert.strictEqual(
        readJson(expired.ackPath).state,
        'failed',
        'absolute expiry must fence even a live hung owner'
    );

    const targetedBase = path.join(
        __test.signalDir,
        'open-request-pid-window-current.json'
    );
    __test.setTargetedSignalFile(targetedBase);
    const recoveredBroadcast = requestData('recovered-broadcast');
    delete recoveredBroadcast.url;
    recoveredBroadcast.broadcast = true;
    const targetedQueue = targetedBase.replace(/\.json$/, '')
        + `.request-${recoveredBroadcast.requestId}.json`;
    writeJson(targetedQueue, recoveredBroadcast);
    await __test.tryOpenSignalFile();
    assert.strictEqual(
        readJson(recoveredBroadcast.ackPath).state,
        'failed',
        'an assigned broadcast remains owned by its window after focus changes'
    );
    assert.strictEqual(fs.existsSync(targetedQueue), false);

    console.log('request journal tests passed');
} finally {
    if (originalHome === undefined) delete process.env.HOME;
    else process.env.HOME = originalHome;
    fs.rmSync(tempHome, { recursive: true, force: true });
}
})().catch(error => {
    console.error(error);
    process.exitCode = 1;
});
