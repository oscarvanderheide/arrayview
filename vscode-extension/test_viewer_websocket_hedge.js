// The tunnel relay can accept a WebSocket upgrade and then stay silent. The
// primary viewer therefore races staggered fresh sockets and adopts exactly one
// winner; related compare/multiview sockets keep their existing path.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const viewer = fs.readFileSync(
    path.join(__dirname, '..', 'src', 'arrayview', '_viewer.html'),
    'utf8'
);
const match = viewer.match(
    /function _createHedgedWebSocket\([\s\S]*?\n        }\n\n        function createTransport/
);
assert(match, 'hedged WebSocket helper must be present in the viewer');
const helperSource = match[0].replace(/\n\n        function createTransport$/, '');
assert.match(
    viewer,
    /_relayWsHedgeEnabled\s*&&\s*transportSid === sid/,
    'only the primary tunnel socket may enter the hedge'
);

function runtime() {
    const timers = new Map();
    let nextTimer = 1;
    const context = {
        WebSocket: { CONNECTING: 0, OPEN: 1, CLOSED: 3 },
        clearTimeout(id) { timers.delete(id); },
        queueMicrotask,
        setTimeout(handler, delay) {
            const id = nextTimer++;
            timers.set(id, { handler, delay });
            return id;
        },
    };
    vm.runInNewContext(`${helperSource}; this.hedge = _createHedgedWebSocket;`, context);
    return { hedge: context.hedge, timers };
}

function fakeSocket() {
    return {
        binaryType: 'blob',
        closeCalls: 0,
        readyState: 0,
        sent: [],
        close() {
            this.closeCalls++;
            this.readyState = 3;
            if (this.onclose) this.onclose({ target: this });
        },
        open() {
            this.readyState = 1;
            if (this.onopen) this.onopen({ target: this });
        },
        send(value) { this.sent.push(value); },
    };
}

function fireDelay(state, delay) {
    const entry = [...state.timers.entries()].find(([, timer]) => timer.delay === delay);
    assert(entry, `expected ${delay}ms hedge timer`);
    state.timers.delete(entry[0]);
    entry[1].handler();
}

// A later fresh socket can win without allowing the swallowed first upgrade to
// delay the viewer for its full timeout.
const state = runtime();
const sockets = [];
const transport = state.hedge(() => {
    const socket = fakeSocket();
    sockets.push(socket);
    return socket;
});
let opens = 0;
transport.onopen = () => { opens++; };
transport.binaryType = 'arraybuffer';
fireDelay(state, 300);
assert.strictEqual(sockets.length, 2);
sockets[1].open();
assert.strictEqual(opens, 1);
assert.strictEqual(sockets[0].closeCalls, 1, 'the black-holed loser is cancelled');
assert.strictEqual(sockets[1].binaryType, 'arraybuffer');
transport.send('render');
assert.deepStrictEqual(sockets[1].sent, ['render']);
assert.strictEqual(state.timers.size, 0, 'the winner cancels unlaunched hedges');

// Fast failure of every candidate becomes one transport error, letting the
// existing reconnect policy start a new bounded wave.
const failed = runtime();
const failedSockets = [];
const failedTransport = failed.hedge(() => {
    const socket = fakeSocket();
    failedSockets.push(socket);
    return socket;
});
let errors = 0;
failedTransport.onerror = () => { errors++; };
failedSockets[0].close();
fireDelay(failed, 300);
failedSockets[1].close();
fireDelay(failed, 600);
failedSockets[2].close();
assert.strictEqual(errors, 1);

console.log('viewer WebSocket hedge tests passed');
