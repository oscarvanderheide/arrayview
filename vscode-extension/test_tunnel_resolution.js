const assert = require('assert');
const Module = require('module');

let resolverCalls = 0;
let resolvePending = null;
const vscodeMock = {
    env: {
        remoteName: null,
        asExternalUri: () => {
            resolverCalls += 1;
            return new Promise(resolve => { resolvePending = resolve; });
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
    const first = __test._asExternalUriSingleFlight(8000, baseUri);
    const second = __test._asExternalUriSingleFlight(8000, baseUri);

    assert.strictEqual(first, second);
    await Promise.resolve();
    assert.strictEqual(resolverCalls, 1);
    await assert.rejects(
        __test._withTimeout(first, 5, 'asExternalUri'),
        /asExternalUri timeout after 5ms/
    );

    const afterTimeout = __test._asExternalUriSingleFlight(8000, baseUri);
    assert.strictEqual(afterTimeout, first);
    assert.strictEqual(resolverCalls, 1, 'a caller timeout must not start another VS Code resolver');

    resolvePending({ toString: () => 'https://example-8000.devtunnels.ms/' });
    await afterTimeout;
    assert.strictEqual(__test._externalUriInFlight.has(8000), false);

    const next = __test._asExternalUriSingleFlight(8000, baseUri);
    await Promise.resolve();
    assert.strictEqual(resolverCalls, 2, 'a settled resolver may be replaced by a future request');
    resolvePending({ toString: () => 'https://example-8000.devtunnels.ms/' });
    await next;

    console.log('tunnel resolution tests passed');
})().catch(error => {
    console.error(error);
    process.exitCode = 1;
});
