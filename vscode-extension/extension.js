const vscode = require('vscode');

function activate(context) {
    // Register URI handler: vscode://arrayview.arrayview-opener/open?url=<encoded-url>
    context.subscriptions.push(vscode.window.registerUriHandler({
        handleUri(uri) {
            const params = new URLSearchParams(uri.query);
            const url = params.get('url');
            if (url) {
                vscode.commands.executeCommand('simpleBrowser.show', url);
            }
        }
    }));
}

function deactivate() {}

module.exports = { activate, deactivate };
