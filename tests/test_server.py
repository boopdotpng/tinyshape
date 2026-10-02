import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


SERVER = Path(__file__).resolve().parents[1] / "tinyshape" / "server.py"


def lsp(cache_home, root, path, text):
  # open one file in a fresh server, wait for its analysis, return (inlay hints, server log lines)
  p = subprocess.Popen([sys.executable, str(SERVER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                       env=os.environ | {"XDG_CACHE_HOME": cache_home})
  def send(msg):
    body = json.dumps({"jsonrpc": "2.0"} | msg).encode()
    p.stdin.write(f"Content-Length: {len(body)}\r\n\r\n".encode() + body); p.stdin.flush()
  def recv():
    n = int(p.stdout.readline().split(b":")[1]); p.stdout.readline()
    return json.loads(p.stdout.read(n))
  uri = Path(path).as_uri()
  send({"id": 1, "method": "initialize", "params": {"rootUri": Path(root).as_uri(), "capabilities": {}, "initializationOptions": {"python": sys.executable}}})
  logs = []
  try:
    send({"method": "textDocument/didOpen", "params": {"textDocument": {"uri": uri, "languageId": "python", "version": 1, "text": text}}})
    while (m := recv()).get("method") != "textDocument/publishDiagnostics":
      if m.get("method") == "window/logMessage": logs.append(m["params"]["message"])
    send({"id": 2, "method": "textDocument/inlayHint", "params": {"textDocument": {"uri": uri}, "range": {"start": {"line": 0, "character": 0}, "end": {"line": 100, "character": 0}}}})
    while (m := recv()).get("id") != 2:
      if m.get("method") == "window/logMessage": logs.append(m["params"]["message"])
    return [h["label"] for h in m["result"]], logs
  finally: p.kill(); p.wait(); p.stdin.close(); p.stdout.close()


class Cache(unittest.TestCase):
  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.cache, self.root = os.path.join(self.tmp.name, "cache"), os.path.join(self.tmp.name, "proj")
    os.makedirs(self.root)
    self.dep = os.path.join(self.root, "dims.py")
    Path(self.dep).write_text("N = 3\n")
    self.path = os.path.join(self.root, "model.py")
    self.src = "from tinygrad import Tensor\nfrom dims import N\ndef f(x):\n  # x.shape = (2, 4)\n  return x.reshape(2, 4, 1).expand(2, 4, N)\n"

  def tearDown(self): self.tmp.cleanup()

  def test_second_open_is_cached_and_same(self):
    hints1, logs1 = lsp(self.cache, self.root, self.path, self.src)
    hints2, logs2 = lsp(self.cache, self.root, self.path, self.src)
    self.assertIn(" -> (2, 4, 3)", hints1)
    self.assertEqual(hints1, hints2)
    self.assertFalse(any("(cached)" in l for l in logs1))
    self.assertTrue(any("(cached)" in l for l in logs2))
    self.assertTrue(os.path.exists(os.path.join(self.cache, "tinyshape", "server.log")))

  def test_changed_source_misses(self):
    lsp(self.cache, self.root, self.path, self.src)
    hints, logs = lsp(self.cache, self.root, self.path, self.src.replace("expand(2, 4, N)", "expand(2, 4, 2 * N)"))
    self.assertIn(" -> (2, 4, 6)", hints)
    self.assertFalse(any("(cached)" in l for l in logs))

  def test_changed_import_misses(self):
    lsp(self.cache, self.root, self.path, self.src)
    time.sleep(0.01)
    Path(self.dep).write_text("N = 17\n")  # new size too: python's .pyc check only sees mtime in whole seconds
    hints, logs = lsp(self.cache, self.root, self.path, self.src)
    self.assertIn(" -> (2, 4, 17)", hints)
    self.assertFalse(any("(cached)" in l for l in logs))


if __name__ == "__main__":
  unittest.main()
