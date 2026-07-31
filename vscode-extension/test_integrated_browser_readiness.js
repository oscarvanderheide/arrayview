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
const commandHistory = [];
let remoteProxyEnabled = false;
let commandFailure = null;
let commandObserver = null;
let getCommandsCalls = 0;
const externalOpens = [];
const vscodeMock = {
    commands: {
        async getCommands() {
            getCommandsCalls += 1;
            return new Promise(() => {});
        },
        async executeCommand(command, args) {
            assert(
                [
                    'workbench.action.browser.open',
                    'workbench.action.browser.hardReload',
                ].includes(command),
                `unexpected command: ${command}`
            );
            commandHistory.push({ command, args });
            if (command === 'workbench.action.browser.open') {
                commandArgs = args;
                commandArgsHistory.push(args);
                if (commandFailure) throw commandFailure;
            }
            if (commandObserver) commandObserver(args, command);
        },
    },
    workspace: {
        getConfiguration() {
            return { get(_name, _fallback) { return remoteProxyEnabled; } };
        },
    },
    env: {
        async openExternal(uri) {
            externalOpens.push(uri.toString());
            return true;
        },
    },
    Uri: {
        parse(value) {
            return { toString() { return value; } };
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
    const preparedBodies = [];
    const releases = [];
    const server = http.createServer((req, res) => {
        if (req.method === 'GET' && req.url === '/ping') {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({
                service: 'arrayview',
                instance_id: 'server-one',
            }));
            return;
        }
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
                preparedBodies.push(prepared);
                const requestId = decodeURIComponent(
                    req.url.split('?')[0].split('/').pop()
                );
                journal = {
                    sid: 'sid-one',
                    request_id: requestId,
                    window_id: prepared.window_id,
                    server_id: prepared.server_id,
                    token: prepared.token,
                    tab_key: prepared.tab_key,
                    navigation_key: prepared.navigation_key,
                    navigation_attempt: prepared.navigation_attempt,
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
            commandArgs.reuseUrlFilter.startsWith('/_av/'),
            true,
            'only retries of the same request may reuse its browser tab'
        );
        const openedUrl = new URL(commandArgs.url);
        const firstPrepared = preparedBodies.at(-1);
        assert.strictEqual(openedUrl.origin, 'http://localhost:9000');
        assert.strictEqual(openedUrl.search, '');
        assert.strictEqual(openedUrl.hash, '');
        assert.strictEqual(firstPrepared.viewer_query, '?sid=sid-one');
        assert.match(firstPrepared.tab_key, /^[A-Za-z0-9_-]{16}$/);
        assert.match(firstPrepared.navigation_key, /^[A-Za-z0-9_-]{16}$/);
        assert.match(firstPrepared.token, /^[0-9a-f]{32}$/);
        assert.strictEqual(firstPrepared.navigation_attempt, 0);
        assert.strictEqual(
            openedUrl.pathname,
            `/_av/${firstPrepared.tab_key}/${firstPrepared.navigation_key}`,
            'the browser command must receive only the short launch route'
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
        assert.strictEqual(new URL(commandArgs.url).search, '');
        assert.notStrictEqual(
            preparedBodies.at(-1).token,
            firstPrepared.token,
            'a distinct request must use a fresh readiness token'
        );
        assert.strictEqual(await replayed.viewerReady, null);

        duplicateViewers = true;
        const currentToken = preparedBodies.at(-1).token;
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
            commandArgs.reuseUrlFilter.startsWith('/_av/'),
            true
        );
        assert.strictEqual(await proxied.viewerReady, null);

        const external = await __test.openInExternalBrowser(
            backendUrl,
            'request-external',
            'server-one',
            'window-external',
            2000
        );
        assert.strictEqual(externalOpens.length, 1);
        const externalUrl = new URL(externalOpens[0]);
        assert.strictEqual(externalUrl.origin, new URL(backendUrl).origin);
        assert.strictEqual(
            externalUrl.searchParams.get('_av_launch_request_id'),
            'request-external'
        );
        assert.strictEqual(
            externalUrl.searchParams.get('_av_launch_window_id'),
            'window-external'
        );
        assert.strictEqual(await external.viewerReady, null);

        const slowRenderStart = commandArgsHistory.length;
        journal = null;
        deferReady = true;
        let framePublishedAt = null;
        commandObserver = (args, command) => {
            if (command !== 'workbench.action.browser.open' || !args.url) return;
            if (journal.request_id !== 'request-slow-render') return;
            journal.phases = ['script-loaded'];
            journal.viewer_instance_ids = ['viewer-one'];
            setTimeout(() => {
                journal.phases = [
                    'script-loaded',
                    'ws-open',
                    'metadata-loaded',
                    'frame-rendered',
                ];
                framePublishedAt = Date.now();
            }, 800);
        };
        const slowRenderStartedAt = Date.now();
        const slowRender = await __test.openInIntegratedBrowser(
            'http://localhost:9000/?sid=sid-one',
            backendUrl,
            'request-slow-render',
            'server-one',
            'window-one',
            3000,
            () => {},
            500
        );
        assert.strictEqual(
            await slowRender.viewerReady,
            null,
            'script-loaded must switch readiness to the full render deadline'
        );
        assert(
            framePublishedAt - slowRenderStartedAt >= 700,
            'the first frame must arrive after the deliberately shorter pre-script budget'
        );
        assert.strictEqual(
            commandArgsHistory.length,
            slowRenderStart + 1,
            'a slow post-script render must not trigger navigation recovery'
        );
        commandObserver = null;

        const delayedPreScriptStart = commandArgsHistory.length;
        const delayedPreScriptCommandStart = commandHistory.length;
        const delayedPreScriptPreparedStart = preparedBodies.length;
        journal = null;
        deferReady = true;
        let delayedPreScriptPublished = false;
        commandObserver = (args, command) => {
            if (
                command === 'workbench.action.browser.open'
                && args && args.url
                && journal.request_id === 'request-delayed-pre-script'
                && !delayedPreScriptPublished
            ) {
                delayedPreScriptPublished = true;
                setTimeout(() => {
                    journal.phases = [
                        'script-loaded',
                        'ws-open',
                        'metadata-loaded',
                        'frame-rendered',
                    ];
                    journal.viewer_instance_ids = ['viewer-one'];
                }, 600);
            }
        };
        const delayedPreScript = await __test.openInIntegratedBrowser(
            'http://localhost:9000/?sid=sid-one',
            backendUrl,
            'request-delayed-pre-script',
            'server-one',
            'window-one',
            6000,
            () => {},
            1000
        );
        assert.strictEqual(await delayedPreScript.viewerReady, null);
        const delayedPreScriptCommands = commandArgsHistory.slice(delayedPreScriptStart);
        assert.strictEqual(
            delayedPreScriptCommands.length,
            1,
            'a delayed viewer script must not open extra integrated-browser tabs'
        );
        const delayedPreScriptPrepared = preparedBodies.slice(
            delayedPreScriptPreparedStart
        );
        assert.deepStrictEqual(
            delayedPreScriptPrepared.map(body => body.navigation_attempt),
            [0],
            'a delayed viewer script must keep its original prepared navigation'
        );
        const delayedPreScriptSequence = commandHistory.slice(
            delayedPreScriptCommandStart
        );
        assert.deepStrictEqual(
            delayedPreScriptSequence.map(entry => entry.command),
            ['workbench.action.browser.open'],
            'one request must issue exactly one integrated-browser open command'
        );
        commandObserver = null;

        const cappedStart = commandArgsHistory.length;
        const cappedCommandStart = commandHistory.length;
        const cappedPreparedStart = preparedBodies.length;
        journal = null;
        const cappedStartedAt = Date.now();
        const capped = await __test.openInIntegratedBrowser(
            'http://localhost:9000/?sid=sid-one',
            backendUrl,
            'request-capped',
            'server-one',
            'window-one',
            5000,
            () => {},
            1000
        );
        assert.match(
            (await capped.viewerReady).message,
            /Integrated browser did not start the viewer script before recovery timeout/,
            'a permanently blank tab must fail on the short pre-script recovery budget'
        );
        assert(
            Date.now() - cappedStartedAt < 2000,
            'a permanently blank tab must not consume the full render deadline'
        );
        const cappedCommands = commandArgsHistory.slice(cappedStart);
        assert.strictEqual(
            cappedCommands.length,
            1,
            'a permanently blank request must not open extra integrated-browser tabs'
        );
        assert.deepStrictEqual(
            preparedBodies
                .slice(cappedPreparedStart)
                .map(body => body.navigation_attempt),
            [0],
            'a permanently blank request must keep its original prepared navigation'
        );
        assert.deepStrictEqual(
            commandHistory.slice(cappedCommandStart).map(entry => entry.command),
            ['workbench.action.browser.open'],
            'a permanently blank request must issue exactly one browser command'
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
