const assert = require('assert');
const {
    collectReleaseSidsFromUrl,
    pingUrlFromViewerUrl,
    shouldRemoveSameTunnelRegistration,
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

console.log('lifecycle helper tests passed');
