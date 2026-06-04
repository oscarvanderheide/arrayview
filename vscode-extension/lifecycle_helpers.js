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

module.exports = {
    collectReleaseSidsFromUrl,
    pingUrlFromViewerUrl,
};
