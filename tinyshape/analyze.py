# runs one file under a tracer on tinygrad's NULL device and prints tensor shapes as json.
# usage: python analyze.py <path>   (source on stdin)
#
# entry points are functions whose leading comments declare input shapes:
#   def __call__(self, x: Tensor, start_pos: int) -> Tensor:
#     # x.shape = (BS, T, emb_dim)        names that aren't module globals become symbolic dims
#                                         hints use global names if the shape does, raw numbers if it's all literals
#     # x.dtype = dtypes.int              optional
#     # start_pos = 0                     plain values for non-tensor params
#     # self = Block(3)                   optional, default is Class() or Class(0, 0, ...)
import ast, io, json, os, sys, tokenize, re, inspect, builtins, traceback
os.environ.setdefault("DEV", "NULL")

real_stdout = sys.stdout
sys.stdout = sys.stderr  # user prints must not corrupt the json

from tinygrad import Tensor, dtypes

ANN_RE = re.compile(r"#\s*([A-Za-z_]\w*)(\.shape|\.dtype)?\s*=\s*(.+?)\s*$")

def utf16_col(line:str, byte_col:int) -> int:
  s = line.encode()[:byte_col].decode(errors="ignore")
  return len(s.encode("utf-16-le")) // 2

class Analysis:
  def __init__(self, path:str, src:str):
    self.path, self.src = path, src
    self.lines = src.splitlines()
    self.tree = ast.parse(src, path)
    self.hints: dict[tuple[int,int], list[tuple[str,str]]] = {}  # (line, byte_col) -> (label, other form), 1-indexed lines
    self.use_names = True
    self.diags: list[dict] = []
    self.rets: set[tuple[int,int]] = set()
    self.sym: dict[int, str] = {}  # sentinel value -> dim name
    self.next_sentinel = 10007
    self.linemap: dict[int, ast.stmt] = {}
    for node in ast.walk(self.tree):  # bfs, so inner statements overwrite outer ones
      if not isinstance(node, ast.stmt): continue
      body = getattr(node, "body", None)
      end = body[0].lineno - 1 if isinstance(body, list) and body and not isinstance(node, ast.ClassDef|ast.FunctionDef|ast.AsyncFunctionDef) else node.end_lineno
      if isinstance(node, ast.ClassDef|ast.FunctionDef|ast.AsyncFunctionDef): continue
      for l in range(node.lineno, (end or node.lineno) + 1): self.linemap[l] = node

  # ---- formatting ----
  def dim_names(self) -> dict[int, str]:
    if os.environ.get("TINYSHAPE_NAME_DIMS", "1") == "0": return {}
    seen: dict[int, list[str]] = {}
    for k, v in self.g.items():
      if type(v) is int and v > 1 and not k.startswith("_"): seen.setdefault(v, []).append(k)
    return {v: ks[0] for v, ks in seen.items() if len(ks) == 1}

  def fmt_dim(self, d, named:bool) -> str:
    if not isinstance(d, int): return str(d)
    names, rest = [], d
    for s, n in self.sym.items():
      while rest and rest % s == 0: names.append(n); rest //= s
    if rest != 1 or not names: names.append(self.names.get(rest, str(rest)) if named else str(rest))
    return "*".join(names)

  def fmt(self, v, named:bool) -> str|None:
    if isinstance(v, Tensor):
      s = "(" + ", ".join(self.fmt_dim(d, named) for d in v.shape) + ("," if len(v.shape) == 1 else "") + ")"
      return s if v.dtype == dtypes.default_float else f"{s} {v.dtype.name}"
    if isinstance(v, (list, tuple)) and 0 < len(v) <= 4 and all(isinstance(x, Tensor) for x in v):
      inner = ", ".join(self.fmt(x, named) for x in v)
      return f"[{inner}]" if isinstance(v, list) else f"({inner})"
    return None

  def add_hint(self, line:int, col:int, v, ret:bool=False):
    if (s := self.fmt(v, self.use_names)) is None: return
    s = (s, self.fmt(v, not self.use_names))
    self.rets.add((line, col)) if ret else None
    labels = self.hints.setdefault((line, col), [])
    if s not in labels: labels.append(s)

  def diag(self, line:int, msg:str, severity:int=1):
    if any(d["line"] == line and d["message"] == msg for d in self.diags): return
    self.diags.append({"line": line, "message": msg, "severity": severity})

  # ---- tracing ----
  def record_targets(self, frame, target:ast.expr):
    loc = frame.f_locals
    if isinstance(target, (ast.Tuple, ast.List)):
      for t in target.elts: self.record_targets(frame, t)
      return
    if isinstance(target, ast.Starred): return self.record_targets(frame, target.value)
    try:
      if isinstance(target, ast.Name): v = loc[target.id]
      elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name): v = getattr(loc[target.value.id], target.attr)
      else: return
    except Exception: return
    self.add_hint(target.end_lineno, target.end_col_offset, v)

  def flush(self, frame, st:ast.stmt):
    if isinstance(st, ast.Assign):
      for t in st.targets: self.record_targets(frame, t)
    elif isinstance(st, (ast.AugAssign, ast.AnnAssign, ast.For)): self.record_targets(frame, st.target)
    elif isinstance(st, ast.With):
      for it in st.items:
        if it.optional_vars is not None: self.record_targets(frame, it.optional_vars)

  def trace(self, frame, event, arg):
    if frame.f_code.co_filename != self.path: return None
    return self.local_trace

  def local_trace(self, frame, event, arg):
    if event == "line":
      st = self.linemap.get(frame.f_lineno)
      prev = self.pending.get(frame)
      if prev is not None and prev is not st: self.flush(frame, prev)
      self.pending[frame] = st
    elif event == "exception": self.pending[frame] = None  # the statement didn't finish, its targets hold old values
    elif event == "return":
      prev = self.pending.pop(frame, None)
      if isinstance(prev, ast.Return) and prev.value is not None and not frame.f_code.co_name.startswith("<"):
        self.add_hint(prev.end_lineno, prev.end_col_offset, arg, ret=True)
      elif prev is not None: self.flush(frame, prev)
    return self.local_trace

  def run_traced(self, fn, /, *args, **kwargs):
    self.pending = {}
    sys.settrace(self.trace)
    try: return fn(*args, **kwargs)
    finally: sys.settrace(None)

  def report_exc(self, e:BaseException, fallback_line:int, prefix:str=""):
    line = fallback_line
    for fs in traceback.extract_tb(e.__traceback__):
      if fs.filename == self.path and fs.lineno: line = fs.lineno
    msg = f"{type(e).__name__}: {e}".strip()
    for s, n in self.sym.items(): msg = re.sub(rf"\b{s}\b", n, msg)
    self.diag(line, prefix + msg)

  # ---- entry points ----
  def eval_expr(self, expr:str, extra:dict|None=None):
    g = {"Tensor": Tensor, "dtypes": dtypes} | self.g  # usable in annotations even if the file doesn't import them
    class Syms(dict):
      def __missing__(s, k):
        if k in g or hasattr(builtins, k): raise KeyError(k)
        for v, n in self.sym.items():
          if n == k: return v
        while self.next_sentinel in self.names or any(self.next_sentinel % p == 0 for p in range(2, 101)): self.next_sentinel += 1
        v = self.next_sentinel; self.next_sentinel += 1
        self.sym[v] = k
        return v
    ns = Syms(extra or {})
    return eval(expr, g, ns)

  def comments(self) -> dict[int, str]:
    out = {}
    for tok in tokenize.generate_tokens(io.StringIO(self.src).readline):
      if tok.type == tokenize.COMMENT: out[tok.start[0]] = tok.string
    return out

  def entries(self):
    comments = self.comments()
    def leading(fn):
      body = fn.body
      if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str) and len(body) > 1: body = body[1:]
      stop = body[0].lineno if body else fn.end_lineno
      return [(l, comments[l]) for l in range(fn.lineno, stop + 1) if l in comments]
    for node in self.tree.body:
      fns = [(None, node)] if isinstance(node, ast.FunctionDef) else [(node, n) for n in node.body if isinstance(n, ast.FunctionDef)] if isinstance(node, ast.ClassDef) else []
      for cls, fn in fns:
        anns = [(l, m.groups()) for l, c in leading(fn) if (m := ANN_RE.match(c))]
        params = [a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs]
        anns = [(l, g) for l, g in anns if g[0] in params]
        if any(g[1] == ".shape" for _, g in anns): yield cls, fn, anns

  def run_entry(self, cls:ast.ClassDef|None, fn:ast.FunctionDef, anns):
    shapes, dts, vals = {}, {}, {}
    self.use_names = any(isinstance(n, ast.Name) and type(self.g.get(n.id)) is int
                         for l, (_, kind, expr) in anns if kind == ".shape" for n in ast.walk(ast.parse(expr, mode="eval")))
    for l, (name, kind, expr) in anns:
      try:
        v = self.eval_expr(expr)
        if kind == ".shape": shapes[name] = tuple(v) if isinstance(v, (tuple, list)) else (v,)
        elif kind == ".dtype": dts[name] = v
        else: vals[name] = v
      except Exception as e: return self.diag(l, f"tinyshape: can't evaluate `{expr}`: {e}")
    kwargs = {n: Tensor.empty(*s, dtype=dts.get(n)) for n, s in shapes.items()} | vals
    target = self.g.get(fn.name) if cls is None else None
    if cls is not None:
      C = self.g.get(cls.name)
      raw = C.__dict__.get(fn.name) if isinstance(C, type) else None
      if raw is None: return self.diag(fn.lineno, f"tinyshape: {cls.name}.{fn.name} not found after executing module")
      if isinstance(raw, staticmethod): target = raw.__func__
      elif isinstance(raw, classmethod): target = raw.__func__.__get__(C)
      else:
        self_name = (fn.args.posonlyargs + fn.args.args)[0].arg
        if self_name not in kwargs:
          try: kwargs[self_name] = self.run_traced(self.make_instance, C)
          except Exception as e: return self.report_exc(e, fn.lineno, f"tinyshape: constructing {cls.name} failed, add `# self = {cls.name}(...)`: ")
        target = raw
    if target is None: return self.diag(fn.lineno, f"tinyshape: {fn.name} not found after executing module")
    try: self.run_traced(target, **kwargs)
    except Exception as e: self.report_exc(e, fn.lineno)

  def make_instance(self, C):
    try: return C()
    except TypeError as e:
      sig = inspect.signature(C)
      req = [p for p in sig.parameters.values() if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
      if not req: raise e
      return C(*[0]*len(req))

  def run(self):
    self.g = {"__name__": "__tinyshape__", "__file__": self.path, "__builtins__": builtins}
    self.names = {}
    sys.path.insert(0, os.path.dirname(self.path))
    for st in self.tree.body:
      try: self.run_traced(exec, compile(ast.Module([st], []), self.path, "exec"), self.g)
      except Exception as e: self.report_exc(e, st.lineno, "tinyshape (module level): ")
    self.names = self.dim_names()
    for cls, fn, anns in self.entries(): self.run_entry(cls, fn, anns)
    hints = []
    for (line, col), labels in sorted(self.hints.items()):
      text = self.lines[line-1] if line-1 < len(self.lines) else ""
      join = lambda xs: " | ".join(xs[:3]) + (" | …" if len(xs) > 3 else "")
      main, alt = join([a for a, _ in labels]), join([b for _, b in labels])
      hints.append({"line": line-1, "character": utf16_col(text, col), "label": main, "tooltip": alt if alt != main else None, "ret": (line, col) in self.rets})
    return {"hints": hints, "diagnostics": [d | {"line": d["line"]-1} for d in self.diags]}

if __name__ == "__main__":
  path = os.path.abspath(sys.argv[1])
  src = sys.stdin.read()
  try: out = Analysis(path, src).run()
  except SyntaxError: out = {"hints": [], "diagnostics": [], "syntax_error": True}
  real_stdout.write(json.dumps(out))
  real_stdout.flush()
  os._exit(0)
