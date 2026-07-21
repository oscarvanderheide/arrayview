const assert = require('assert');
const fs = require('fs');
const http = require('http');
const Module = require('module');
const os = require('os');
const path = require('path');

const tempHome = fs.mkdtempSync(path.join(os.tmpdir(), 'arrayview-deadline-'));
const originalHome = process.env.HOME;
process.env.HOME = tempHome;

let panelCount = 0;
const vscodeMock = {
    env: { remoteName: null, uiKind: 1 },
    UIKind: { Web: 2 },
    ViewColumn: { Active: 1, Beside: 2 },
    window: {
        state: { focused: true },
        activeTextEditor: null,
        createWebviewPanel() {
            panelCount += 1;
            throw new Error('expired request must not create a panel');
        },
    },
    workspace: {
        getConfiguration() { return { get: () => false }; },
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
    const server = http.createServer((req, res) => {
        if (req.url.startsWith('/metadata/')) {
            res.writeHead(404);
            res.end();
            return;
        }
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ service: 'arrayview', instance_id: 'server-deadline' }));
    });
    try {
        fs.mkdirSync(__test.signalDir, { recursive: true });
        __test.setWindowId('window-deadline');
        fs.writeFileSync(
            path.join(__test.signalDir, 'window-window-deadline.json'),
            JSON.stringify({
                pid: process.pid,
                windowId: 'window-deadline',
                extensionInstanceId: __test.extensionInstanceId,
            })
        );
        await new Promise(resolve => server.listen(0, 'localhost', resolve));
        const port = server.address().port;
        const requestId = 'deadline-request';
        const data = {
            protocolVersion: 1,
            requestId,
            serverId: 'server-deadline',
            ackPath: path.join(
                __test.signalDir,
                `open-ack-v0100-${requestId}.json`
            ),
            sentAtMs: Date.now(),
            maxAgeMs: 50,
            url: `http://localhost:${port}/?sid=deadline-sid`,
        };
        assert.strictEqual(__test.claimProtocolRequest(data), 'acquired');
        await assert.rejects(
            __test._processSignalDataBody(data),
            /expired before (?:a panel could be opened|display side effect)/
        );
        await new Promise(resolve => setTimeout(resolve, 100));
        assert.strictEqual(panelCount, 0);
        assert.strictEqual(
            JSON.parse(fs.readFileSync(data.ackPath, 'utf8')).state,
            'failed'
        );
        console.log('request deadline tests passed');
    } finally {
        await new Promise(resolve => server.close(resolve));
        if (originalHome === undefined) delete process.env.HOME;
        else process.env.HOME = originalHome;
        fs.rmSync(tempHome, { recursive: true, force: true });
    }
})().catch(error => {
    console.error(error);
    process.exitCode = 1;
});
