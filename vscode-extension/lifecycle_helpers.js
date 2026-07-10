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

function validatedAckPath(ackPath, requestId, homeDir) {
    if (typeof ackPath !== 'string' || typeof requestId !== 'string' || !requestId) return null;
    const signalDir = path.resolve(homeDir, '.arrayview');
    const expectedName = `open-ack-v0100-${requestId}.json`;
    const resolved = path.resolve(ackPath);
    if (path.dirname(resolved) !== signalDir || path.basename(resolved) !== expectedName) return null;
    return resolved;
}

function ackPayload(state, data, windowId, message) {
    const payload = {
        protocolVersion: 1,
        state,
        requestId: data.requestId,
        windowId,
        serverId: data.serverId ?? null,
        timestampMs: Date.now(),
    };
    if (message) payload.message = String(message);
    return payload;
}

module.exports = {
    collectReleaseSidsFromUrl,
    pingUrlFromViewerUrl,
    shouldDeferBroadcast,
    shouldRemoveSameTunnelRegistration,
    validatedAckPath,
    ackPayload,
};
