# minimal stdio language server: inlay hints with tinygrad tensor shapes, diagnostics for shape errors.
# the actual work happens in analyze.py, run in a fresh subprocess per edit (user code is executed!).
# only files containing a `# <name>.shape = (...)` or `# tinyshape: run` comment are analyzed.
# results are cached in ~/.cache/tinyshape, keyed on the source and settings, and checked against the files the run imported.
import hashlib, json, os, re, subprocess, sys, threading, time, urllib.parse

ANALYZE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze.py")
TRIGGER = re.compile(r"#\s*(\w+\.shape\s*=|tinyshape:\s*run\b)")
DEBOUNCE, TIMEOUT = 0.3, 20.0
CACHE_DIR = os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "tinyshape")
CACHE_MAX = 500  # entries, least recently used go first
LOG_FILE, LOG_MAX = os.path.join(CACHE_DIR, "server.log"), 1 << 20

out_lock = threading.Lock()
docs: dict[str, str] = {}
results: dict[str, list[dict]] = {}
gens: dict[str, int] = {}
procs: dict[str, subprocess.Popen] = {}
state_lock = threading.Lock()
opts = {"python": None, "tinygradPath": None, "nameDims": True, "chainHints": True, "cache": True}
warned_outside: set[str] = set()
warned_missing = False  # the missing-tinygrad notice is sent once per server
client_refresh = False
root: str|None = None  # zed runs one server per worktree and can send every server the same file, so each only takes its own
next_id = 0

def send(msg:dict):
  body = json.dumps(msg).encode()
  with out_lock:
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
    sys.stdout.buffer.flush()

def notify(method, params): send({"jsonrpc": "2.0", "method": method, "params": params})
def request(method, params):
  global next_id
  next_id += 1
  send({"jsonrpc": "2.0", "id": f"ts{next_id}", "method": method, "params": params})
def log(msg, client:bool=True):
  # also to a file, so it can be read without the editor's lsp log panel
  if client: notify("window/logMessage", {"type": 4, "message": f"tinyshape: {msg}"})
  try:
    with open(LOG_FILE, "a") as f: f.write(f"{time.strftime('%m-%d %H:%M:%S')} [{os.getpid()}] {msg}\n")
  except OSError: pass

# ---- cache ----
def cache_key(path:str, src:str, python:str, env:dict) -> str:
  st = os.stat(ANALYZE)
  envs = {k: v for k, v in env.items() if k in ("DEV", "PYTHONPATH") or k.startswith("TINYSHAPE_")}
  return hashlib.sha256(json.dumps([path, src, python, envs, st.st_mtime_ns, st.st_size]).encode()).hexdigest()

def cache_get(key:str) -> dict|None:
  f = os.path.join(CACHE_DIR, key + ".json")
  try:
    with open(f) as fh: res = json.load(fh)
    for dep, mtime, size in res["deps"]:
      st = os.stat(dep)
      if (st.st_mtime_ns, st.st_size) != (mtime, size): return None
    os.utime(f)  # mtime is the lru clock
    return res
  except (OSError, ValueError, KeyError, TypeError): return None

def cache_put(key:str, res:dict):
  try:
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = os.path.join(CACHE_DIR, f".{key}.{os.getpid()}.tmp")
    with open(tmp, "w") as fh: json.dump(res, fh)
    os.replace(tmp, os.path.join(CACHE_DIR, key + ".json"))
    entries = [e for e in os.scandir(CACHE_DIR) if e.name.endswith(".json")]
    if len(entries) > CACHE_MAX:
      for e in sorted(entries, key=lambda e: e.stat().st_mtime)[:len(entries) - CACHE_MAX]: os.remove(e.path)
  except OSError as e: log(f"cache write failed: {e}")

def uri_path(uri:str) -> str: return urllib.parse.unquote(urllib.parse.urlparse(uri).path)
def owned(uri:str) -> bool:
  if root is None or (uri_path(uri) + os.sep).startswith(root + os.sep): return True
  if uri not in warned_outside: warned_outside.add(uri); log(f"ignoring {uri_path(uri)}, outside root {root}", client=False)
  return False

def publish(uri, hints, diags):
  with state_lock: results[uri] = hints
  notify("textDocument/publishDiagnostics", {"uri": uri, "diagnostics": [
    {"range": {"start": {"line": d["line"], "character": 0}, "end": {"line": d["line"], "character": 10**4}},
     "severity": d["severity"], "source": "tinyshape", "message": d["message"]} for d in diags]})
  if client_refresh: request("workspace/inlayHint/refresh", None)

def analyze(uri:str, gen:int):
  threading.Event().wait(DEBOUNCE)
  with state_lock:
    if gens.get(uri) != gen: return
    src = docs.get(uri)
    if (old := procs.pop(uri, None)) is not None: old.kill()
    if src is None: return
    if not TRIGGER.search(src):
      if results.pop(uri, None) is not None: publish(uri, [], [])
      return
    path, python = uri_path(uri), find_python(uri_path(uri))
    env = os.environ | {"DEV": "NULL", "TINYSHAPE_NAME_DIMS": "1" if opts["nameDims"] else "0",
                         "TINYSHAPE_CHAIN_HINTS": "1" if opts["chainHints"] else "0"}
    if (tg := tinygrad_root()): env["PYTHONPATH"] = os.pathsep.join(filter(None, [tg, env.get("PYTHONPATH")]))
  key = cache_key(path, src, python, env) if opts["cache"] else None
  if key and (res := cache_get(key)) is not None:
    with state_lock:
      if gens.get(uri) != gen: return
    log(f"{path}: {len(res['hints'])} hints, {len(res['diagnostics'])} diagnostics (cached)")
    return publish(uri, res["hints"], res["diagnostics"])
  with state_lock:
    if gens.get(uri) != gen: return
    t0 = time.monotonic()
    p = subprocess.Popen([python, ANALYZE, path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=os.path.dirname(path), env=env)
    procs[uri] = p
  try: stdout, stderr = p.communicate(src.encode(), timeout=TIMEOUT)
  except subprocess.TimeoutExpired:
    p.kill()
    return log(f"analysis of {path} timed out after {TIMEOUT}s")
  with state_lock:
    if gens.get(uri) != gen: return
    procs.pop(uri, None)
  try: res = json.loads(stdout)
  except json.JSONDecodeError: return log(f"analysis failed (exit {p.returncode}):\n{stderr.decode(errors='replace')[-4000:]}")
  if res.get("syntax_error"): return  # keep old hints while typing
  if res.get("missing_tinygrad"): return missing_tinygrad(res["python"])
  if key: cache_put(key, res)
  log(f"{path}: {len(res['hints'])} hints, {len(res['diagnostics'])} diagnostics in {time.monotonic() - t0:.2f}s")
  publish(uri, res["hints"], res["diagnostics"])

def tinygrad_root() -> str|None:
  # `tinygradPath` is a checkout: the repo root, or the tinygrad/ package inside it
  if not (p := opts["tinygradPath"]): return None
  p = os.path.normpath(os.path.expanduser(p))
  return os.path.dirname(p) if os.path.exists(os.path.join(p, "__init__.py")) else p

def missing_tinygrad(python:str):
  global warned_missing
  where = f"in {opts['tinygradPath']}" if opts["tinygradPath"] else f"for {python}"
  log(f"tinygrad not found {where}. set `python` to a python with tinygrad installed, or `tinygradPath` to a tinygrad checkout")
  with state_lock:
    if warned_missing: return
    warned_missing = True
  # the vscode extension turns this into a prompt, other clients ignore it
  notify("tinyshape/missingTinygrad", {"python": python, "tinygradPath": opts["tinygradPath"]})

def find_python(path:str) -> str:
  # explicit option, else the nearest .venv above the file, else whatever runs this server
  if opts["python"]: return opts["python"]
  d = os.path.dirname(path)
  while True:
    if os.path.exists(cand := os.path.join(d, ".venv", "bin", "python")): return cand
    if (up := os.path.dirname(d)) == d: return sys.executable
    d = up

def schedule(uri:str):
  with state_lock: gen = gens[uri] = gens.get(uri, 0) + 1
  threading.Thread(target=analyze, args=(uri, gen), daemon=True).start()

def inlay_hints(params):
  r, uri = params["range"], params["textDocument"]["uri"]
  with state_lock: hints = results.get(uri, [])
  log(f"inlayHint {uri_path(uri)} lines {r['start']['line']}-{r['end']['line']}: {sum(r['start']['line'] <= h['line'] <= r['end']['line'] for h in hints)} of {len(hints)}", client=False)
  return [{"position": {"line": h["line"], "character": h["character"]}, "label": f"{' -> ' if h.get('ret') else ': '}{h['label']}", "kind": 1, "paddingLeft": False} | ({"tooltip": h["tooltip"]} if h.get("tooltip") else {})
          for h in hints if r["start"]["line"] <= h["line"] <= r["end"]["line"]]

def handle(msg:dict):
  global client_refresh, root
  method, params, mid = msg.get("method"), msg.get("params") or {}, msg.get("id")
  if method is None: return  # response to one of our requests
  result = None
  if method == "initialize":
    if (r := params.get("rootUri") or params.get("rootPath")): root = os.path.normpath(uri_path(r) if "://" in r else r)
    client_refresh = bool(params.get("capabilities", {}).get("workspace", {}).get("inlayHint", {}).get("refreshSupport"))
    opts.update({k: v for k, v in (params.get("initializationOptions") or {}).items() if v is not None})
    log(f"server {__file__} on {sys.executable}, root {root}, options {opts}, extension settings: {os.environ.get('TINYSHAPE_SETTINGS', '-')}")
    result = {"capabilities": {"textDocumentSync": {"openClose": True, "change": 1, "save": False}, "inlayHintProvider": True},
              "serverInfo": {"name": "tinyshape"}}
  elif method == "shutdown": result = None
  elif method == "exit": os._exit(0)
  elif method == "textDocument/didOpen":
    td = params["textDocument"]
    if owned(td["uri"]): docs[td["uri"]] = td["text"]; schedule(td["uri"])
  elif method == "textDocument/didChange":
    uri = params["textDocument"]["uri"]
    if owned(uri): docs[uri] = params["contentChanges"][-1]["text"]; schedule(uri)
  elif method == "textDocument/didClose":
    uri = params["textDocument"]["uri"]
    with state_lock: docs.pop(uri, None); results.pop(uri, None); gens[uri] = gens.get(uri, 0) + 1
  elif method == "textDocument/inlayHint": result = inlay_hints(params)
  elif mid is not None:
    return send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"unhandled {method}"}})
  if mid is not None: send({"jsonrpc": "2.0", "id": mid, "result": result})

def main():
  try:
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.getsize(LOG_FILE) > LOG_MAX: os.replace(LOG_FILE, LOG_FILE + ".old")
  except OSError: pass
  stdin = sys.stdin.buffer
  while True:
    headers = {}
    while (line := stdin.readline()):
      if line in (b"\r\n", b"\n"): break
      k, _, v = line.decode().partition(":"); headers[k.strip().lower()] = v.strip()
    if not line: break
    handle(json.loads(stdin.read(int(headers["content-length"]))))

if __name__ == "__main__": main()
