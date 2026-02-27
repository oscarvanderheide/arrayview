const vscode = require('vscode');

function activate(context) {
    // Register URI handler: vscode://arrayview.arrayview-opener/open?url=<encoded-url>
    context.subscriptions.push(vscode.window.registerUriHandler({
        async handleUri(uri) {
            const params = new URLSearchParams(uri.query);
            const url = params.get('url');
            if (!url) return;

            // In remote/tunnel contexts, ask VS Code to resolve the URI
            // through its port forwarding mechanism. This ensures that
            // http://127.0.0.1:<port> on the remote is accessible locally.
            let resolvedUrl = url;
            try {
                const resolved = await vscode.env.asExternalUri(
                    vscode.Uri.parse(url)
                );
                resolvedUrl = resolved.toString(/* skipEncoding */ true);
            } catch (_) {
                // Not in a remote context, or forwarding not needed
            }

            vscode.commands.executeCommand('simpleBrowser.show', resolvedUrl);
        }
    }));
}

function deactivate() {}

module.exports = { activate, deactivate };
