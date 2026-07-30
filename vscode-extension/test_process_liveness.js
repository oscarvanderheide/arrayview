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

const originalKill = process.kill;
const originalReadFileSync = fs.readFileSync;

try {
    process.kill = () => true;
    fs.readFileSync = (filePath, ...args) => {
        if (filePath === '/proc/101/status') {
            return 'Name:\tMainThread\nState:\tZ (zombie)\n';
        }
        if (filePath === '/proc/102/status') {
            return 'Name:\tnode\nState:\tS (sleeping)\n';
        }
        return originalReadFileSync(filePath, ...args);
    };

    assert.strictEqual(__test.isProcessAlive(101), false);
    assert.strictEqual(__test.isProcessAlive(102), true);
    assert.strictEqual(
        __test.requestMatchesRemoteName({ targetRemoteName: 'tunnel' }, 'tunnel'),
        true
    );
    assert.strictEqual(
        __test.requestMatchesRemoteName({ targetRemoteName: 'tunnel' }, 'ssh-remote'),
        false
    );
    assert.strictEqual(__test.requestMatchesRemoteName({}, 'ssh-remote'), true);

    process.kill = () => {
        throw new Error('ESRCH');
    };
    assert.strictEqual(__test.isProcessAlive(103), false);
    process.stdout.write('process liveness tests passed\n');
} finally {
    process.kill = originalKill;
    fs.readFileSync = originalReadFileSync;
}
