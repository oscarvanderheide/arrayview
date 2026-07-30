const assert = require('assert');
const fs = require('fs');
const Module = require('module');

const vscodeMock = {
    env: { remoteName: 'tunnel', appHost: 'desktop' },
    commands: {
        executeCommand: async () => undefined,
        getCommands: async () => [],
    },
    workspace: {
        getConfiguration: () => ({
            get: (_key, fallback) => fallback,
            update: async () => undefined,
        }),
    },
};

const originalLoad = Module._load;
Module._load = function (request, parent, isMain) {
    if (request === 'vscode') return vscodeMock;
    return originalLoad.call(this, request, parent, isMain);
};
const { __test } = require('./extension');
Module._load = originalLoad;

const originalAppendFile = fs.appendFile;
const originalConsoleLog = console.log;
const callbacks = [];
fs.appendFile = (_path, _data, callback) => {
    callbacks.push(callback);
};
console.log = () => {};

(async () => {
    try {
        for (let index = 0; index < 400; index += 1) {
            __test.log(`slow-home-${index}`);
        }

        const blocked = __test.logQueueState();
        assert.strictEqual(callbacks.length, 1, 'only one filesystem write may be active');
        assert.strictEqual(blocked.writeActive, true);
        assert.ok(blocked.queued <= 256, 'the pending log queue must stay bounded');
        assert.ok(blocked.dropped > 0, 'excess diagnostics should be dropped');

        callbacks.shift()(new Error('simulated NFS failure'));
        await new Promise(resolve => setImmediate(resolve));

        const failed = __test.logQueueState();
        assert.strictEqual(failed.writeDisabled, true);
        assert.strictEqual(failed.writeActive, false);
        assert.strictEqual(failed.queued, 0);

        __test.log('console-only-after-failure');
        assert.strictEqual(callbacks.length, 0, 'a failed log path must not be retried');
        process.stdout.write('nonblocking logging tests passed\n');
    } finally {
        fs.appendFile = originalAppendFile;
        console.log = originalConsoleLog;
    }
})().catch(error => {
    fs.appendFile = originalAppendFile;
    console.log = originalConsoleLog;
    console.error(error);
    process.exit(1);
});
