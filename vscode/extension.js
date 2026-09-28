// runs the bundled python server (server/, copied from ../tinyshape at package time) with `tinyshape.python`
// or python3 from PATH. like zed, one server per workspace folder, and one per directory for files outside any folder.
const path = require("path");
const vscode = require("vscode");
const { LanguageClient } = require("vscode-languageclient/node");

const clients = new Map();  // folder uri -> LanguageClient
let output, context;

// `loose` is a directory holding files outside any workspace folder, served like its own folder minus subdirectories
function startClient(folder, loose) {
  const key = folder.uri.toString();
  if (clients.has(key)) return;
  const cfg = vscode.workspace.getConfiguration("tinyshape", folder);
  const opt = (k) => cfg.get(k) ?? null;
  const server = opt("server") || context.asAbsolutePath(path.join("server", "server.py"));
  const init = { python: opt("python"), nameDims: cfg.get("nameDims"), chainHints: cfg.get("chainHints") };
  const settings = JSON.stringify({ python: opt("python"), server: opt("server"), nameDims: init.nameDims, chainHints: init.chainHints });
  const run = { command: opt("python") || "python3", args: [server], options: { env: { ...process.env, TINYSHAPE_SETTINGS: settings } } };
  const pattern = `${folder.uri.fsPath.replace(/[[\]{}*?]/g, "[$&]")}/${loose ? "*" : "**/*"}`;
  const client = new LanguageClient("tinyshape", "tinyshape", run, {
    documentSelector: [{ scheme: "file", language: "python", pattern }],
    workspaceFolder: folder,
    initializationOptions: init,
    outputChannel: output,
  });
  clients.set(key, client);
  client.start().catch((e) => output.appendLine(`tinyshape: failed to start: ${e}`));
}

async function stopClient(key) {
  const client = clients.get(key);
  clients.delete(key);
  if (client) await client.stop().catch(() => {});
}

function onOpen(doc) {
  if (doc.languageId !== "python" || doc.uri.scheme !== "file") return;
  const folder = vscode.workspace.getWorkspaceFolder(doc.uri);
  if (folder) return startClient(folder, false);
  const dir = vscode.Uri.file(path.dirname(doc.uri.fsPath));
  startClient({ uri: dir, name: path.basename(dir.fsPath), index: -1 }, true);
}

async function restart() {
  await Promise.all([...clients.keys()].map(stopClient));
  vscode.workspace.textDocuments.forEach(onOpen);
}

function activate(ctx) {
  context = ctx;
  output = vscode.window.createOutputChannel("tinyshape");
  ctx.subscriptions.push(
    output,
    vscode.commands.registerCommand("tinyshape.restart", restart),
    vscode.workspace.onDidOpenTextDocument(onOpen),
    vscode.workspace.onDidChangeConfiguration((e) => e.affectsConfiguration("tinyshape") && restart()),
    vscode.workspace.onDidChangeWorkspaceFolders((e) => e.removed.forEach((f) => stopClient(f.uri.toString()))),
  );
  vscode.workspace.textDocuments.forEach(onOpen);
}

function deactivate() {
  return Promise.all([...clients.keys()].map(stopClient));
}

module.exports = { activate, deactivate };
