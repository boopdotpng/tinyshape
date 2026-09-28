// runs the bundled python server (server/, copied from ../tinyshape at package time). analysis uses `tinyshape.python`, else the
// interpreter picked in the Python extension, else the server's own .venv search. like zed, one server per workspace folder, and
// one per directory for files outside any folder.
const fs = require("fs");
const path = require("path");
const vscode = require("vscode");
const { LanguageClient } = require("vscode-languageclient/node");

const clients = new Map();  // folder uri -> LanguageClient, or a placeholder object while its python is being resolved
const warned = new Set();  // pythons already reported as missing tinygrad since the last restart
let output, context, pythonApi;

// the Python extension's api, if it's installed
async function getPythonApi() {
  const ext = vscode.extensions.getExtension("ms-python.python");
  if (!ext) return null;
  try {
    const api = ext.isActive ? ext.exports : await ext.activate();
    return api?.environments ? api : null;
  } catch (e) {
    output.appendLine(`tinyshape: Python extension unavailable: ${e}`);
    return null;
  }
}

// the interpreter selected in the Python extension for this folder
async function selectedPython(uri) {
  if (!pythonApi) return null;
  try {
    const env = await pythonApi.environments.resolveEnvironment(pythonApi.environments.getActiveEnvironmentPath(uri));
    return env?.executable?.uri?.fsPath ?? null;
  } catch (e) {
    output.appendLine(`tinyshape: couldn't resolve the selected interpreter: ${e}`);
    return null;
  }
}

// `loose` is a directory holding files outside any workspace folder, served like its own folder minus subdirectories
async function startClient(folder, loose) {
  const key = folder.uri.toString();
  if (clients.has(key)) return;
  const pending = {};
  clients.set(key, pending);
  const cfg = vscode.workspace.getConfiguration("tinyshape", folder.uri);
  const opt = (k) => cfg.get(k) || null;
  const python = opt("python") || await selectedPython(folder.uri);
  if (clients.get(key) !== pending) return;  // stopped while resolving
  const server = opt("server") || context.asAbsolutePath(path.join("server", "server.py"));
  const init = { python, tinygradPath: opt("tinygradPath"), nameDims: cfg.get("nameDims"), chainHints: cfg.get("chainHints") };
  const settings = JSON.stringify({ ...init, pythonSetting: opt("python"), server: opt("server") });
  const run = { command: python || "python3", args: [server], options: { env: { ...process.env, TINYSHAPE_SETTINGS: settings } } };
  const pattern = `${folder.uri.fsPath.replace(/[[\]{}*?]/g, "[$&]")}/${loose ? "*" : "**/*"}`;
  const client = new LanguageClient("tinyshape", "tinyshape", run, {
    documentSelector: [{ scheme: "file", language: "python", pattern }],
    workspaceFolder: folder,
    initializationOptions: init,
    outputChannel: output,
  });
  client.onNotification("tinyshape/missingTinygrad", (p) => missingTinygrad(p, folder));
  clients.set(key, client);
  client.start().catch((e) => output.appendLine(`tinyshape: failed to start: ${e}`));
}

async function stopClient(key) {
  const client = clients.get(key);
  clients.delete(key);
  if (client instanceof LanguageClient) await client.stop().catch(() => {});
}

function onOpen(doc) {
  if (doc.languageId !== "python" || doc.uri.scheme !== "file") return;
  const folder = vscode.workspace.getWorkspaceFolder(doc.uri);
  if (folder) return startClient(folder, false);
  const dir = vscode.Uri.file(path.dirname(doc.uri.fsPath));
  startClient({ uri: dir, name: path.basename(dir.fsPath), index: -1 }, true);
}

async function restart() {
  warned.clear();
  await Promise.all([...clients.keys()].map(stopClient));
  vscode.workspace.textDocuments.forEach(onOpen);
}

async function missingTinygrad({ python, tinygradPath }, folder) {
  if (warned.has(python)) return;
  warned.add(python);
  // the Python extension's selection is ignored while tinyshape.python is set
  const useApi = pythonApi && !vscode.workspace.getConfiguration("tinyshape", folder.uri).get("python");
  const pick = useApi ? "Select Python Interpreter" : "Choose Python...";
  const msg = tinygradPath
    ? `tinyshape: tinygrad wasn't found in ${tinygradPath}. Pick the tinygrad folder you cloned, or a Python with tinygrad installed.`
    : `tinyshape: tinygrad isn't installed for ${python}. Pick the tinygrad folder you cloned, or a Python with tinygrad installed.`;
  const choice = await vscode.window.showWarningMessage(msg, "Set tinygrad Folder...", pick);
  if (choice === pick) await choosePython(folder, useApi);
  else if (choice) await setTinygradFolder();
}

async function choosePython(folder, useApi) {
  if (useApi) return vscode.commands.executeCommand("python.setInterpreter");
  const [file] = await vscode.window.showOpenDialog({ canSelectFiles: true, openLabel: "Use this Python", title: "Python with tinygrad installed" }) ?? [];
  if (!file) return;
  const inFolder = folder.index >= 0;
  await vscode.workspace.getConfiguration("tinyshape", inFolder ? folder.uri : undefined)
    .update("python", file.fsPath, inFolder ? vscode.ConfigurationTarget.WorkspaceFolder : vscode.ConfigurationTarget.Global);
}

// saved globally: there's usually one checkout per machine
async function setTinygradFolder() {
  const [dir] = await vscode.window.showOpenDialog({ canSelectFolders: true, canSelectFiles: false, openLabel: "Use this folder", title: "Your tinygrad clone (the folder with setup.py in it)" }) ?? [];
  if (!dir) return;
  const p = dir.fsPath;
  if (!fs.existsSync(path.join(p, "tinygrad", "__init__.py")) && !fs.existsSync(path.join(p, "__init__.py"))) {
    const again = await vscode.window.showErrorMessage(`tinyshape: ${p} doesn't look like a tinygrad clone (no tinygrad/__init__.py inside).`, "Pick Again");
    return again && setTinygradFolder();
  }
  await vscode.workspace.getConfiguration("tinyshape").update("tinygradPath", p, vscode.ConfigurationTarget.Global);
  vscode.window.showInformationMessage(`tinyshape: using tinygrad from ${p}`);
}

async function activate(ctx) {
  context = ctx;
  output = vscode.window.createOutputChannel("tinyshape");
  pythonApi = await getPythonApi();
  ctx.subscriptions.push(
    output,
    vscode.commands.registerCommand("tinyshape.restart", restart),
    vscode.commands.registerCommand("tinyshape.setTinygradFolder", setTinygradFolder),
    vscode.workspace.onDidOpenTextDocument(onOpen),
    vscode.workspace.onDidChangeConfiguration((e) => e.affectsConfiguration("tinyshape") && restart()),
    vscode.workspace.onDidChangeWorkspaceFolders((e) => e.removed.forEach((f) => stopClient(f.uri.toString()))),
  );
  if (pythonApi) ctx.subscriptions.push(pythonApi.environments.onDidChangeActiveEnvironmentPath(() => restart()));
  vscode.workspace.textDocuments.forEach(onOpen);
}

function deactivate() {
  return Promise.all([...clients.keys()].map(stopClient));
}

module.exports = { activate, deactivate };
