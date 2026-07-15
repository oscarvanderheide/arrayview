const path = require('path');

function collectReleaseSidsFromUrl(url) {
    let parsed;
    try {
        parsed = new URL(url);
    } catch (_) {
        return [];
    }

    const sids = new Set();
    const addSids = (value) => {
        if (!value) return;
        for (const sid of value.split(',')) {
            const trimmed = sid.trim();
            if (trimmed && trimmed !== '__welcome__') {
                sids.add(trimmed);
            }
        }
    };

    addSids(parsed.searchParams.get('sid'));
    addSids(parsed.searchParams.get('compare_sid'));
    addSids(parsed.searchParams.get('compare_sids'));
    addSids(parsed.searchParams.get('overlay_sid'));
    return Array.from(sids);
}

function pingUrlFromViewerUrl(url) {
    try {
        return `${new URL(url).origin}/ping`;
    } catch (_) {
        return null;
    }
}

function sessionMetadataUrlFromViewerUrl(url) {
    try {
        const parsed = new URL(url);
        const sid = parsed.searchParams.get('sid');
        if (!sid || sid === '__welcome__') return null;
        return `${parsed.origin}/metadata/${encodeURIComponent(sid)}`;
    } catch (_) {
        return null;
    }
}

function releaseUrlForSid(viewerUrl, backendUrl, sid) {
    try {
        const origin = new URL(backendUrl || viewerUrl).origin;
        return `${origin}/release/${encodeURIComponent(sid)}`;
    } catch (_) {
        return null;
    }
}

function isVersionAtLeast(actual, required) {
    const parse = (value) => String(value || '').split('.').map(Number);
    const left = parse(actual);
    const right = parse(required);
    if (left.some(Number.isNaN) || right.some(Number.isNaN)) return false;
    const length = Math.max(left.length, right.length);
    for (let index = 0; index < length; index++) {
        const a = left[index] || 0;
        const b = right[index] || 0;
        if (a !== b) return a > b;
    }
    return true;
}

function shouldRemoveSameTunnelRegistration(currentWindowId, currentPpids, candidateWindowId, candidateData, candidateAlive) {
    if (candidateWindowId === currentWindowId) return false;
    if (!candidateData || !candidateData.pid || candidateAlive) return false;
    const candidatePpids = Array.isArray(candidateData.ppids) ? candidateData.ppids : [];
    if (!Array.isArray(currentPpids) || currentPpids.length < 1 || candidatePpids.length < 1) {
        return false;
    }
    return candidatePpids[0] === currentPpids[0];
}

function shouldDeferBroadcast(isOwnTargetedFile, isFocused, data) {
    return !isOwnTargetedFile && !isFocused && data?.broadcast === true;
}

function isLoopbackUrl(url) {
    try {
        const hostname = new URL(url).hostname.toLowerCase();
        return hostname === 'localhost'
            || hostname === '127.0.0.1'
            || hostname === '::1'
            || hostname === '[::1]';
    } catch (_) {
        return false;
    }
}

function validatedAckPath(ackPath, requestId, homeDir) {
    if (typeof ackPath !== 'string' || typeof requestId !== 'string' || !requestId) return null;
    const signalDir = path.resolve(homeDir, '.arrayview');
    const expectedName = `open-ack-v0100-${requestId}.json`;
    const resolved = path.resolve(ackPath);
    if (path.dirname(resolved) !== signalDir || path.basename(resolved) !== expectedName) return null;
    return resolved;
}

function ackPayload(state, data, windowId, message, extensionVersion = null) {
    const payload = {
        protocolVersion: 1,
        state,
        requestId: data.requestId,
        windowId,
        serverId: data.serverId ?? null,
        extensionVersion,
        timestampMs: Date.now(),
    };
    if (message) payload.message = String(message);
    return payload;
}

function isArrayViewStatus(payload, expectedServerId = null) {
    return Boolean(
        payload
        && payload.service === 'arrayview'
        && (!expectedServerId || payload.instance_id === expectedServerId)
    );
}

module.exports = {
    collectReleaseSidsFromUrl,
    pingUrlFromViewerUrl,
    sessionMetadataUrlFromViewerUrl,
    releaseUrlForSid,
    isVersionAtLeast,
    isLoopbackUrl,
    shouldDeferBroadcast,
    shouldRemoveSameTunnelRegistration,
    validatedAckPath,
    ackPayload,
    isArrayViewStatus,
};
