// Page over VS Code's channel, data over the relay.
//
// Measured 2026-07-28: the ~1.9 MB viewer page took 8.4 s to cross the
// devtunnel relay when it was healthy and 27.3 s when it was not, and it is
// byte-identical on every launch (verified: two different sids produce the same
// md5). It is also the only large transfer in a launch. VS Code's portMapping
// routes an http://localhost:PORT request from the webview through the
// connection VS Code already holds to this machine, which does not black-hole
// the way the relay does.
//
// The socket cannot move: VS Code does not remap WebSocket ports. So the page
// is delivered over the mapping and told, via ?data_origin=, to address the
// backend absolutely for everything else — leaving the data path exactly where
// it was. These cases pin that split, and pin that no other display path is
// touched, since a wrong answer here breaks delivery for every launch.

const assert = require('assert');
const fs = require('fs');
const Module = require('module');
const os = require('os');
const path = require('path');

const tempHome = fs.mkdtempSync(path.join(os.tmpdir(), 'arrayview-delivery-'));
const originalHome = process.env.HOME;
process.env.HOME = tempHome;
fs.mkdirSync(path.join(tempHome, '.arrayview'));

let remoteName = 'tunnel';
const vscodeMock = {
    env: { get remoteName() { return remoteName; }, appHost: 'desktop' },
    Uri: { parse: value => value },
    ConfigurationTarget: { Global: 1 },
    workspace: { getConfiguration: () => ({ get: (k, f) => f, update: async () => {} }) },
    commands: { executeCommand: async () => undefined, getCommands: async () => [] },
};

const originalLoad = Module._load;
Module._load = function (request, parent, isMain) {
    if (request === 'vscode') return vscodeMock;
    return originalLoad.call(this, request, parent, isMain);
};
const { __test } = require('./extension');
Module._load = originalLoad;

const RELAY = 'https://v54z0psh-8000.euw.devtunnels.ms/?sid=abc123';
const BACKEND = 'http://localhost:8000/?sid=abc123';

try {
    // --- the split itself ---------------------------------------------------
    const delivery = __test._tunnelPageDelivery(RELAY, BACKEND);
    assert.ok(delivery, 'a tunnel window with a loopback backend must use the mapping');

    const parsed = new URL(delivery.deliveryUrl);
    assert.strictEqual(parsed.protocol, 'http:', 'the page must be fetched over the mapping');
    assert.strictEqual(parsed.hostname, 'localhost');
    assert.strictEqual(parsed.port, '8000');
    assert.strictEqual(
        parsed.searchParams.get('data_origin'),
        'https://v54z0psh-8000.euw.devtunnels.ms',
        'the viewer must be told to keep addressing the backend through the relay'
    );
    assert.deepStrictEqual(
        delivery.portMapping,
        [{ webviewPort: 8000, extensionHostPort: 8000 }],
        'the webview port must match the port the page asks for'
    );

    // --- the whole query has to survive -------------------------------------
    // sid, compare_sid, overlay_names and the launch token all ride in it, and
    // dropping any of them silently opens the wrong thing.
    const rich = __test._tunnelPageDelivery(
        'https://relay.devtunnels.ms/?sid=abc&compare_sid=def&compare_sids=def'
        + '&overlay_sid=ghi&overlay_names=Regions&_av_launch_token=tok',
        BACKEND
    );
    const richParams = new URL(rich.deliveryUrl).searchParams;
    for (const [key, value] of Object.entries({
        sid: 'abc',
        compare_sid: 'def',
        compare_sids: 'def',
        overlay_sid: 'ghi',
        overlay_names: 'Regions',
        _av_launch_token: 'tok',
    })) {
        assert.strictEqual(richParams.get(key), value, `${key} must survive delivery`);
    }

    // --- every other display path must be left alone ------------------------
    assert.strictEqual(
        __test._tunnelPageDelivery('http://localhost:8000/?sid=x', BACKEND),
        null,
        'a loopback display URL already avoids the relay; do not touch it'
    );
    assert.strictEqual(
        __test._tunnelPageDelivery(RELAY, null),
        null,
        'without a known loopback backend there is nothing to map to'
    );
    assert.strictEqual(
        __test._tunnelPageDelivery(RELAY, 'http://example.com:8000/'),
        null,
        'a non-loopback backend cannot be reached through the extension host'
    );
    remoteName = 'ssh-remote';
    assert.strictEqual(
        __test._tunnelPageDelivery(RELAY, BACKEND),
        null,
        'only tunnel windows put the page on a relay; Remote SSH must not change'
    );
    remoteName = null;
    assert.strictEqual(
        __test._tunnelPageDelivery(RELAY, BACKEND),
        null,
        'a local window must not change'
    );
    remoteName = 'tunnel';

    // --- the fallback is real, not decorative -------------------------------
    // If the mapping does not carry the page, the panel must return to the
    // relay rather than retrying a route that is not working.
    const html = __test._viewerPanelHtml(
        delivery.deliveryUrl, null, 2000, RELAY
    );
    assert.ok(
        html.includes(JSON.stringify(RELAY)),
        'the wrapper must carry the relay URL as its fallback'
    );
    assert.ok(
        html.includes('delivery-fallback'),
        'the fallback must be observable in the phase journal'
    );
    const noFallback = __test._viewerPanelHtml(RELAY, null, 2000, null);
    assert.ok(
        noFallback.includes('const fallbackUrl = null'),
        'paths that were never remapped must have no fallback armed'
    );

    console.log('tunnel page delivery tests passed');
} finally {
    if (originalHome === undefined) delete process.env.HOME;
    else process.env.HOME = originalHome;
    fs.rmSync(tempHome, { recursive: true, force: true });
}
