const assert = require('assert');
const Module = require('module');

let resolverCalls = 0;
const pendingResolvers = [];
const vscodeMock = {
    env: {
        remoteName: null,
        asExternalUri: () => {
            resolverCalls += 1;
            return new Promise(resolve => { pendingResolvers.push(resolve); });
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
    const baseUri = { toString: () => 'http://localhost:8000/' };
    const first = __test._asExternalUriAttempt(baseUri);
    await Promise.resolve();
    assert.strictEqual(resolverCalls, 1);
    await assert.rejects(
        __test._withTimeout(first, 5, 'asExternalUri'),
        /asExternalUri timeout after 5ms/
    );

    const next = __test._asExternalUriAttempt(baseUri);
    await Promise.resolve();
    assert.strictEqual(
        resolverCalls,
        2,
        'a timed-out resolver must not poison the next request'
    );
    pendingResolvers[1]({ toString: () => 'https://fresh-8000.devtunnels.ms/' });
    assert.strictEqual(
        (await next).toString(),
        'https://fresh-8000.devtunnels.ms/'
    );

    pendingResolvers[0]({ toString: () => 'https://late-8000.devtunnels.ms/' });
    assert.strictEqual(
        (await first).toString(),
        'https://late-8000.devtunnels.ms/',
        'a late old resolver may settle but has no shared routing side effect'
    );

    console.log('tunnel resolution tests passed');
})().catch(error => {
    console.error(error);
    process.exitCode = 1;
});
