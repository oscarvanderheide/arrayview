const assert = require('assert');
const path = require('path');
const {
    collectReleaseSidsFromUrl,
    pingUrlFromViewerUrl,
    sessionMetadataUrlFromViewerUrl,
    shouldDeferBroadcast,
    shouldRemoveSameTunnelRegistration,
    validatedAckPath,
    ackPayload,
    isArrayViewStatus,
} = require('./lifecycle_helpers');

assert.deepStrictEqual(
    collectReleaseSidsFromUrl(
        'http://localhost:8123/?sid=base&compare_sid=cmp1&compare_sids=cmp1,cmp2&overlay_sid=ov1,ov2'
    ),
    ['base', 'cmp1', 'cmp2', 'ov1', 'ov2']
);

assert.deepStrictEqual(
    collectReleaseSidsFromUrl('http://localhost:8123/?sid=__welcome__&compare_sids=cmp'),
    ['cmp']
);

assert.deepStrictEqual(collectReleaseSidsFromUrl('not a url'), []);
assert.strictEqual(pingUrlFromViewerUrl('http://localhost:8123/?sid=abc'), 'http://localhost:8123/ping');
assert.strictEqual(pingUrlFromViewerUrl('not a url'), null);
assert.strictEqual(
    sessionMetadataUrlFromViewerUrl('https://example.test/?sid=base&overlay_sid=mask'),
    'https://example.test/metadata/base'
);
assert.strictEqual(sessionMetadataUrlFromViewerUrl('https://example.test/?sid=__welcome__'), null);
assert.strictEqual(sessionMetadataUrlFromViewerUrl('not a url'), null);

assert.strictEqual(
    shouldRemoveSameTunnelRegistration('current', [10], 'old', { pid: 123, ppids: [10] }, false),
    true
);
assert.strictEqual(
    shouldRemoveSameTunnelRegistration('current', [10], 'old', { pid: 123, ppids: [10] }, true),
    false
);
assert.strictEqual(
    shouldRemoveSameTunnelRegistration('current', [10], 'current', { pid: 123, ppids: [10] }, false),
    false
);
assert.strictEqual(
    shouldRemoveSameTunnelRegistration('current', [10], 'old', { pid: 123, ppids: [20] }, false),
    false
);

assert.strictEqual(shouldDeferBroadcast(false, false, { broadcast: true }), true);
assert.strictEqual(shouldDeferBroadcast(false, true, { broadcast: true }), false);
assert.strictEqual(shouldDeferBroadcast(true, false, { broadcast: true }), false);
assert.strictEqual(shouldDeferBroadcast(false, false, { broadcast: false }), false);

const home = path.join(path.parse(process.cwd()).root, 'home', 'tester');
const ackPath = path.join(home, '.arrayview', 'open-ack-v0100-req-1.json');
assert.strictEqual(
    validatedAckPath(ackPath, 'req-1', home),
    path.resolve(ackPath)
);
assert.strictEqual(validatedAckPath(path.join(path.parse(process.cwd()).root, 'tmp', 'open-ack-v0100-req-1.json'), 'req-1', home), null);
assert.strictEqual(
    validatedAckPath(path.join(home, '.arrayview', 'open-ack-v0100-other.json'), 'req-1', home),
    null
);
assert.strictEqual(
    validatedAckPath(path.join(home, '.arrayview', 'sub', 'open-ack-v0100-req-1.json'), 'req-1', home),
    null
);

const ack = ackPayload(
    'panel_opened',
    { requestId: 'req-1', serverId: 'server-1' },
    'window-1'
);
assert.strictEqual(ack.protocolVersion, 1);
assert.strictEqual(ack.state, 'panel_opened');
assert.strictEqual(ack.requestId, 'req-1');
assert.strictEqual(ack.windowId, 'window-1');
assert.strictEqual(ack.serverId, 'server-1');
assert.strictEqual(typeof ack.timestampMs, 'number');

const failedAck = ackPayload('failed', { requestId: 'req-2' }, 'window-2', 'boom');
assert.strictEqual(failedAck.serverId, null);
assert.strictEqual(failedAck.message, 'boom');

assert.strictEqual(isArrayViewStatus({ service: 'arrayview' }), true);
assert.strictEqual(
    isArrayViewStatus({ service: 'arrayview', instance_id: 'server-1' }, 'server-1'),
    true
);
assert.strictEqual(
    isArrayViewStatus({ service: 'arrayview', instance_id: 'server-2' }, 'server-1'),
    false
);
assert.strictEqual(isArrayViewStatus({ service: 'other' }), false);

console.log('lifecycle helper tests passed');
