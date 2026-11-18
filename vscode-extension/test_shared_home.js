// Several machines can share one home folder (network home), and several users
// can share one machine. The opener must only judge windows and requests that
// live on its own host, and only hand files to the user's own server.

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

const home = fs.mkdtempSync(path.join(os.tmpdir(), 'av-shared-home-'));
const runtime = fs.mkdtempSync(path.join(os.tmpdir(), 'av-runtime-'));
process.env.HOME = home;
process.env.ARRAYVIEW_RUNTIME_DIR = runtime;

const originalLoad = Module._load;
Module._load = function (request, ...rest) {
    if (request === 'vscode') {
        return new Proxy({}, { get: () => new Proxy(function () {}, { get: () => undefined }) });
    }
    return originalLoad.call(this, request, ...rest);
};
const ext = require('./extension.js').__test;
Module._load = originalLoad;

const dir = ext.SIGNAL_DIR;
assert.strictEqual(dir, path.join(home, '.arrayview'));
fs.mkdirSync(dir, { recursive: true });
const write = (name, value) => fs.writeFileSync(path.join(dir, name), JSON.stringify(value));

// Registrations from another host are foreign; legacy ones (no host) are not.
assert.strictEqual(ext.isForeignRegistration({ host: ext.OWN_HOST + '-x', pid: 1 }), true);
assert.strictEqual(ext.isForeignRegistration({ host: ext.OWN_HOST, pid: 1 }), false);
assert.strictEqual(ext.isForeignRegistration({ pid: 1 }), false);

// In-flight requests: only own window, own chain, or same-host windows.
ext.setOwnWindowChain(['oldid']);
write('window-samehost.json', { host: ext.OWN_HOST, pid: process.pid });
write('window-elsewhere.json', { host: ext.OWN_HOST + '-x', pid: process.pid });
const claim = (id) => `open-request-ipc-${id}.request-abc.json.claimed-123`;
assert.strictEqual(ext._mayRecoverClaim(claim('oldid')), true);
assert.strictEqual(ext._mayRecoverClaim(claim('samehost')), true);
assert.strictEqual(ext._mayRecoverClaim(claim('elsewhere')), false);
assert.strictEqual(ext._mayRecoverClaim(claim('unregistered')), false);
assert.strictEqual(ext._mayRecoverClaim('open-request-v0900.json.claimed-123'), true);

// Cleanup leaves another host's registration (dead pid here) and its claim alone,
// and does not delete a young .tmp that may be a write in progress elsewhere.
write('window-farpid.json', { host: ext.OWN_HOST + '-x', pid: 2 ** 22 + 7 });
write(claim('elsewhere'), { requestId: 'abc', sentAtMs: Date.now() - 3600e3, maxAgeMs: 1 });
write('open-request-ipc-elsewhere.json.tmp', {});
ext.cleanupStaleFiles();
assert.ok(fs.existsSync(path.join(dir, 'window-farpid.json')));
assert.ok(fs.existsSync(path.join(dir, claim('elsewhere'))));
assert.ok(fs.existsSync(path.join(dir, 'open-request-ipc-elsewhere.json.tmp')));

// Clicks look for this user's own servers: 8000 first, then registered ports.
fs.mkdirSync(path.join(runtime, 'instances'), { recursive: true });
fs.writeFileSync(path.join(runtime, 'instances', 'a.json'), JSON.stringify({
    instance_id: 'own-8001', pid: process.pid, port: 8001, started_at: 1,
}));
fs.writeFileSync(path.join(runtime, 'instances', 'b.json'), JSON.stringify({
    instance_id: 'dead', pid: 2 ** 22 + 7, port: 8005, started_at: 0,
}));
const ports = ext._ownServerPortCandidates();
assert.deepStrictEqual(ports.map((c) => c.port), [8000, 8001]);
assert.strictEqual(ports[1].instanceId, 'own-8001');

fs.rmSync(home, { recursive: true, force: true });
fs.rmSync(runtime, { recursive: true, force: true });
console.log('shared home tests passed');
