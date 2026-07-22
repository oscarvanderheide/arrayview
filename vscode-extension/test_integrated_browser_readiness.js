const assert = require('assert');
const fs = require('fs');
const http = require('http');
const Module = require('module');
const os = require('os');
const path = require('path');

const tempHome = fs.mkdtempSync(path.join(os.tmpdir(), 'arrayview-browser-ready-'));
const originalHome = process.env.HOME;
process.env.HOME = tempHome;

let commandArgs = null;
const commandArgsHistory = [];
let remoteProxyEnabled = false;
let commandFailure = null;
let commandObserver = null;
let getCommandsCalls = 0;
const vscodeMock = {
    commands: {
        async getCommands() {
            getCommandsCalls += 1;
            return new Promise(() => {});
        },
        async executeCommand(command, args) {
            assert.strictEqual(command, 'workbench.action.browser.open');
            commandArgs = args;
            commandArgsHistory.push(args);
            if (commandFailure) throw commandFailure;
            if (commandObserver) commandObserver(args);
        },
    },
    workspace: {
        getConfiguration() {
            return { get(_name, _fallback) { return remoteProxyEnabled; } };
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
    let duplicateViewers = false;
    let deferReady = false;
    let journal = null;
    const releases = [];
    const server = http.createServer((req, res) => {
        if (req.method === 'POST') {
            if (req.url.startsWith('/release/')) {
                releases.push(req.url);
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end('{}');
                return;
            }
            let body = '';
            req.setEncoding('utf8');
            req.on('data', chunk => { body += chunk; });
            req.on('end', () => {
                const prepared = JSON.parse(body);
                const requestId = decodeURIComponent(
                    req.url.split('?')[0].split('/').pop()
                );
                journal = {
                    sid: 'sid-one',
                    request_id: requestId,
                    window_id: prepared.window_id,
                    server_id: prepared.server_id,
                    token: prepared.token,
                    phases: deferReady ? [] : [
                        'script-loaded',
                        'ws-open',
                        'metadata-loaded',
                        'frame-rendered',
                    ],
                    viewer_instance_ids: deferReady ? [] : ['viewer-one'],
                };
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ ...journal, phases: [], viewer_instance_ids: [] }));
            });
            return;
        }
        const requestedToken = new URL(req.url, 'http://localhost')
            .searchParams.get('token');
        if (!journal || requestedToken !== journal.token) {
            res.writeHead(409);
            res.end();
            return;
        }
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
            ...journal,
            viewer_instance_ids: duplicateViewers
                ? ['viewer-one', 'viewer-two']
                : ['viewer-one'],
        }));
    });
    await new Promise(resolve => server.listen(0, 'localhost', resolve));
    const address = server.address();
    const backendUrl = `http://localhost:${address.port}/?sid=sid-one`;

    try {
        assert.strictEqual(
            await __test.integratedBrowserCommandAvailable(10),
            true,
            'blocked command discovery must fall through to a direct command attempt'
        );
        const discoveryCalls = getCommandsCalls;
        const opened = await __test.openInIntegratedBrowser(
            'http://localhost:9000/?sid=sid-one',
            backendUrl,
            'request-one',
            'server-one',
            'window-one',
            2000
        );
        assert(commandArgs, 'integrated browser command must be invoked');
        assert.strictEqual(
            getCommandsCalls,
            discoveryCalls,
            'opening must not repeat the potentially blocking command enumeration'
        );
        assert.strictEqual(
            typeof commandArgs,
            'object',
            'workbench.action.browser.open requires structured reuse arguments'
        );
        assert.deepStrictEqual(Object.keys(commandArgs).sort(), [
            'openToSide',
            'reuseUrlFilter',
            'url',
        ]);
        assert.strictEqual(
            commandArgs.openToSide,
            false,
            'distinct ArrayView calls need new tabs, not permanently locked side groups'
        );
        assert.strictEqual(
            commandArgs.reuseUrlFilter,
            '?_av_launch_request_id=request-one',
            'only retries of the same request may reuse its browser tab'
        );
        const openedUrl = new URL(commandArgs.url);
        assert.strictEqual(openedUrl.origin, 'http://localhost:9000');
        assert.strictEqual(openedUrl.searchParams.get('sid'), 'sid-one');
        assert.strictEqual(openedUrl.searchParams.get('_av_integrated_browser'), '1');
        assert.strictEqual(
            openedUrl.searchParams.get('_av_launch_request_id'),
            'request-one'
        );
        assert.strictEqual(
            openedUrl.searchParams.get('_av_launch_server_id'),
            'server-one'
        );
        assert.strictEqual(
            openedUrl.searchParams.get('_av_launch_window_id'),
            'window-one'
        );
        assert.match(
            openedUrl.searchParams.get('_av_launch_token'),
            /^[0-9a-f]{32}$/
        );
        assert.strictEqual(
            commandArgs.url,
            `http://localhost:9000/?sid=sid-one&_av_integrated_browser=1&_av_launch_request_id=request-one&_av_launch_server_id=server-one&_av_launch_window_id=window-one&_av_launch_token=${openedUrl.searchParams.get('_av_launch_token')}`,
            'the integrated browser command must receive the exact correlated launch URL'
        );
        assert.strictEqual(await opened.viewerReady, null);

        const firstCommandArgs = commandArgs;
        const replayed = await __test.openInIntegratedBrowser(
            'http://localhost:9000/?sid=sid-one',
            backendUrl,
            'request-two',
            'server-one',
            'window-two',
            2000
        );
        assert.strictEqual(commandArgsHistory.length, 2);
        assert.notStrictEqual(
            commandArgs.reuseUrlFilter,
            firstCommandArgs.reuseUrlFilter,
            'distinct invocations must open distinct browser tabs'
        );
        assert.notStrictEqual(
            commandArgs.url,
            firstCommandArgs.url,
            'a distinct request must navigate with a new correlated launch URL'
        );
        assert.strictEqual(
            new URL(commandArgs.url).searchParams.get('_av_launch_request_id'),
            'request-two'
        );
        assert.strictEqual(
            new URL(commandArgs.url).searchParams.get('_av_launch_window_id'),
            'window-two'
        );
        assert.notStrictEqual(
            new URL(commandArgs.url).searchParams.get('_av_launch_token'),
            new URL(firstCommandArgs.url).searchParams.get('_av_launch_token'),
            'a distinct request must use a fresh readiness token'
        );
        assert.strictEqual(await replayed.viewerReady, null);

        duplicateViewers = true;
        const currentToken = new URL(commandArgs.url)
            .searchParams.get('_av_launch_token');
        const duplicateError = await __test.waitForBackendViewerReady(
            backendUrl,
            'sid-one',
            'request-two',
            'server-one',
            'window-two',
            currentToken,
            2000
        );
        assert.match(duplicateError.message, /opened 2 viewer instances/);

        duplicateViewers = false;
        remoteProxyEnabled = true;
        const proxied = await __test.openInIntegratedBrowser(
            'http://localhost:9000/?sid=sid-one',
            backendUrl,
            'request-proxy',
            'server-one',
            'window-one',
            2000
        );
        assert.strictEqual(new URL(commandArgs.url).origin, new URL(backendUrl).origin);
        assert.strictEqual(
            commandArgs.reuseUrlFilter,
            '?_av_launch_request_id=request-proxy'
        );
        assert.strictEqual(await proxied.viewerReady, null);

        const recoveryStart = commandArgsHistory.length;
        journal = null;
        deferReady = true;
        commandObserver = args => {
            const parsed = new URL(args.url);
            if (
                parsed.searchParams.get('_av_launch_request_id') === 'request-recovery'
                && parsed.searchParams.get('_av_navigation_attempt') === '1'
            ) {
                journal.phases = [
                    'script-loaded',
                    'ws-open',
                    'metadata-loaded',
                    'frame-rendered',
                ];
                journal.viewer_instance_ids = ['viewer-one'];
            }
        };
        const recovered = await __test.openInIntegratedBrowser(
            'http://localhost:9000/?sid=sid-one',
            backendUrl,
            'request-recovery',
            'server-one',
            'window-one',
            6000
        );
        assert.strictEqual(await recovered.viewerReady, null);
        const recoveryCommands = commandArgsHistory.slice(recoveryStart);
        assert.strictEqual(
            recoveryCommands.length,
            2,
            'a pre-script transport failure must retry exactly once in the same tab'
        );
        assert.strictEqual(recoveryCommands[0].openToSide, false);
        assert.strictEqual(recoveryCommands[1].openToSide, false);
        assert.strictEqual(
            recoveryCommands[0].reuseUrlFilter,
            '?_av_launch_request_id=request-recovery'
        );
        assert.strictEqual(
            recoveryCommands[1].reuseUrlFilter,
            recoveryCommands[0].reuseUrlFilter,
            'pre-script recovery must reuse the one request-specific tab'
        );
        assert.strictEqual(
            new URL(recoveryCommands[0].url).searchParams.get('_av_navigation_attempt'),
            null
        );
        assert.strictEqual(
            new URL(recoveryCommands[1].url).searchParams.get('_av_navigation_attempt'),
            '1',
            'the retry must bypass a failed navigation cache'
        );
        assert.notStrictEqual(
            new URL(recoveryCommands[1].url).searchParams.get('_av_launch_token'),
            new URL(recoveryCommands[0].url).searchParams.get('_av_launch_token'),
            'each navigation attempt must fence stale documents with a fresh token'
        );
        await new Promise(resolve => setTimeout(resolve, 3200));
        assert.strictEqual(
            commandArgsHistory.length,
            recoveryStart + 2,
            'navigation retries must stop permanently after script-loaded'
        );
        commandObserver = null;

        const cappedStart = commandArgsHistory.length;
        journal = null;
        const capped = await __test.openInIntegratedBrowser(
            'http://localhost:9000/?sid=sid-one',
            backendUrl,
            'request-capped',
            'server-one',
            'window-one',
            5200
        );
        assert.match(
            (await capped.viewerReady).message,
            /did not render a frame/,
            'a permanently blank tab must still fail at the original deadline'
        );
        const cappedCommands = commandArgsHistory.slice(cappedStart);
        assert.strictEqual(
            cappedCommands.length,
            3,
            'pre-script recovery must stop after two same-tab retries'
        );
        assert.deepStrictEqual(
            cappedCommands.map(args => args.reuseUrlFilter),
            Array(3).fill('?_av_launch_request_id=request-capped'),
            'all bounded recovery attempts must target one request tab'
        );
        assert.deepStrictEqual(
            cappedCommands.map(args => new URL(args.url).searchParams.get('_av_navigation_attempt')),
            [null, '1', '2']
        );
        deferReady = false;

        commandFailure = new Error('browser command failed after dispatch');
        await assert.rejects(
            __test.openInIntegratedBrowser(
                'http://localhost:9000/?sid=sid-one',
                backendUrl,
                'request-failure',
                'server-one',
                'window-one',
                2000
            ),
            /browser command failed/
        );
        await new Promise(resolve => setTimeout(resolve, 50));
        assert(
            releases.some(value => value.includes('sid-one')),
            'a rejected browser command must release its prepared session'
        );

        console.log('integrated browser readiness tests passed');
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
