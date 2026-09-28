# runs one file under a tracer on tinygrad's NULL device and prints tensor shapes as json.
# usage: python analyze.py <path>   (source on stdin)
#
# entry points are functions whose leading comments declare input shapes:
#   def __call__(self, x: Tensor, start_pos: int) -> Tensor:
#     # x.shape = (BS, T, emb_dim)        names that aren't module globals become symbolic dims
#                                         hints use global names if the shape does, raw numbers if it's all literals
#     # x.dtype = dtypes.int              optional, default float with an int retry if the call fails
#     # start_pos = 0                     plain values for non-tensor params, int/float/bool ones default to 0
#     # self = Block(3)                   optional, default is Class() or Class(0, 0, ...)
#   def rope_table():
#     # tinyshape: run                    run a function that has no shapes to declare
import ast, copy, io, json, os, sys, tokenize, re, inspect, builtins, traceback
os.environ.setdefault("DEV", "NULL")

real_stdout = sys.stdout
sys.stdout = sys.stderr  # user prints must not corrupt the json

from tinygrad import Tensor, dtypes
from tinygrad.helpers import argfix

# random init builds and realizes an rng graph per tensor (most of the time for a big model's __init__), but only
# shapes matter here, so every random constructor makes an empty tensor instead
def _empty_like_rand(default):
  def f(cls, *shape, dtype=None, device=None, **_): return Tensor.empty(*argfix(*shape), dtype=dtype or default, device=device)
  return classmethod(f)
for _n in ["rand", "randn", "uniform", "normal", "scaled_uniform", "glorot_uniform", "kaiming_uniform", "kaiming_normal"]:
  setattr(Tensor, _n, _empty_like_rand(dtypes.default_float))
Tensor.randint = _empty_like_rand(dtypes.int32)

ANN_RE = re.compile(r"#\s*([A-Za-z_]\w*)(\.shape|\.dtype)?\s*=\s*(.+?)\s*$")
RUN_RE = re.compile(r"#\s*tinyshape:\s*run\b")

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
    self.spans: list[tuple[int,int]] = []  # line ranges of entry points
    self.cur: tuple[int,int]|None = None  # span of the entry being run
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
    if not names and (off := self.offset_dim(d)): return off
    if rest != 1 or not names: names.append(self.names.get(rest, str(rest)) if named else str(rest))
    return "*".join(names)

  def offset_dim(self, d:int) -> str|None:
    # k*T + c for small k and c, e.g. after pad or cat. sentinels are spaced so this is unambiguous
    for s, n in self.sym.items():
      for k in range(1, 9):
        if 0 < abs(c := d - k*s) <= 64: return f"{n if k == 1 else f'{k}*{n}'}{c:+d}"
    return None

  def fmt(self, v, named:bool) -> str|None:
    if isinstance(v, Tensor):
      s = "(" + ", ".join(self.fmt_dim(d, named) for d in v.shape) + ("," if len(v.shape) == 1 else "") + ")"
      return s if v.dtype == dtypes.default_float else f"{s} {v.dtype.name}"
    if isinstance(v, (list, tuple)) and 0 < len(v) <= 4 and all(isinstance(x, Tensor) for x in v):
      inner = ", ".join(self.fmt(x, named) for x in v)
      return f"[{inner}]" if isinstance(v, list) else f"({inner})"
    return None

  def foreign(self, line:int) -> bool:
    # inside another entry point: that function's own comment decides its shapes, not whoever calls it
    return any(a <= line <= b for a, b in self.spans if (a, b) != self.cur)

  def add_hint(self, line:int, col:int, v, ret:bool=False):
    if self.names is None:  # module level, names aren't known yet. copy lists, they may be appended to later
      v = list(v) if isinstance(v, list) else v
      return self.deferred.append(lambda: self.add_hint(line, col, v, ret))
    if self.foreign(line) or (s := self.fmt(v, self.use_names)) is None: return
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
        elts = prev.value.elts if isinstance(prev.value, ast.Tuple) else None
        if elts and isinstance(arg, tuple) and len(arg) == len(elts) and not any(isinstance(e, ast.Starred) for e in elts):
          for e, v in zip(elts, arg): self.add_hint(e.end_lineno, e.end_col_offset, v)  # `return a, b` gets a hint on each
        else: self.add_hint(prev.end_lineno, prev.end_col_offset, arg, ret=True)
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
      if fs.filename == self.path and fs.lineno and not self.foreign(fs.lineno): line = fs.lineno
    msg = f"{type(e).__name__}: {e}".strip()
    used = []
    for s, n in self.sym.items():
      msg, k = re.subn(rf"\b{s}\b", n, msg)
      if k: used.append(f"{n}={s}")
    # a symbolic dim is a big stand-in number, which overruns fixed-size tables like rope or a kv cache
    if used: msg += (f" ({used[0]} is a stand-in for an unknown size; if it indexes" if len(used) == 1 else
                     f" ({', '.join(used)} are stand-ins for unknown sizes; if one indexes") + " a fixed-size table, give it a number in the shape comment)"
    self.diag(line, prefix + msg)

  # ---- entry points ----
  def eval_expr(self, expr:str, extra:dict|None=None):
    g = {"Tensor": Tensor, "dtypes": dtypes} | self.g  # usable in annotations even if the file doesn't import them
    class Syms(dict):
      def __missing__(s, k):
        if k in g or hasattr(builtins, k): raise KeyError(k)
        for v, n in self.sym.items():
          if n == k: return v
        def ok(v): return (v not in self.names and all(v % p for p in range(2, 101)) and
                           all(abs(k*v - j*u) > 200 for u in self.sym for k in range(1, 9) for j in range(1, 9)))
        while not ok(self.next_sentinel): self.next_sentinel += 1
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
        lead = leading(fn)
        anns = [(l, m.groups()) for l, c in lead if (m := ANN_RE.match(c))]
        params = [a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs]
        anns = [(l, g) for l, g in anns if g[0] in params]
        if any(g[1] == ".shape" for _, g in anns) or any(RUN_RE.match(c) for _, c in lead): yield cls, fn, anns

  def run_entry(self, cls:ast.ClassDef|None, fn:ast.FunctionDef, anns):
    shapes, dts, vals = {}, {}, {}
    # with no declared shapes there's no style to follow, so use names
    self.use_names = not any(kind == ".shape" for _, (_, kind, _) in anns) or any(isinstance(n, ast.Name) and type(self.g.get(n.id)) is int
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
    # required params with no annotation comment: scalars get a zero, anything else needs a comment
    for p in inspect.signature(target).parameters.values():
      if p.name in kwargs or p.default is not p.empty or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD): continue
      zero = {t: t() for t in (int, float, bool)} | {t.__name__: t() for t in (int, float, bool)}
      if p.annotation in zero: kwargs[p.name] = zero[p.annotation]
      else: return self.diag(fn.lineno, f"tinyshape: no value for `{p.name}`, add `# {p.name}.shape = (...)` or `# {p.name} = ...`")
    snap = ({k: list(v) for k, v in self.hints.items()}, set(self.rets))
    try: self.run_traced(target, **kwargs)
    except Exception as e:
      # inputs without a declared dtype are float; if that fails, retry with one of them as int (e.g. token ids
      # into nn.Embedding, indices into gather), then all of them. if every attempt fails, report the one that got
      # furthest (most hints), so a later bug isn't hidden behind the float input's error
      undeclared = [n for n in shapes if n not in dts]
      best = (self.hints, self.rets, e)
      for ints in [[n] for n in undeclared] + ([undeclared] if len(undeclared) > 1 else []):
        self.hints, self.rets = {k: list(v) for k, v in snap[0].items()}, set(snap[1])
        try: self.run_traced(target, **(kwargs | {n: Tensor.empty(*shapes[n], dtype=dtypes.int) for n in ints}))
        except Exception as e2:
          if sum(map(len, self.hints.values())) > sum(map(len, best[0].values())): best = (self.hints, self.rets, e2)
          continue
        return
      self.hints, self.rets, e = best
      self.report_exc(e, fn.lineno)

  def make_instance(self, C):
    try: return C()
    except TypeError as e:
      sig = inspect.signature(C)
      req = [p for p in sig.parameters.values() if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
      if not req: raise e
      return C(*[0]*len(req))

  # ---- chain hints ----
  def instrument(self) -> ast.Module:
    # wrap each link of a method chain like `a(x).b()[0].c()` so we can see its value, the value of its receiver, and the
    # value of the step after it. every shape in the chain is shown once, after the link that produced it: a link that
    # keeps its receiver's shape gets no hint, and neither does one whose shape carries on to the end of the chain,
    # since that shape is already on the assignment or return
    tree, self.links = copy.deepcopy(self.tree), []
    self.link_vals: dict[int, list] = {}
    self.link_recv: dict[int, list] = {}
    self.link_next: dict[int, list] = {}
    self.link_step: dict[int, int|None] = {}
    links, recvs, nexts = {}, {}, {}  # id(link) -> k, id(receiver) -> [k], id(step after link) -> k
    for node in ast.walk(tree):
      if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute): inner, step = node.func, node
      elif isinstance(node, (ast.Attribute, ast.Subscript)): inner, step = node, node
      else: continue
      link = inner.value
      if not isinstance(link, (ast.Call, ast.Subscript, ast.BinOp)) or id(link) in links: continue
      k = links[id(link)] = len(self.links)
      self.links.append((link.end_lineno, link.end_col_offset))
      nexts[id(step)] = k
      recv = link.func.value if isinstance(link, ast.Call) and isinstance(link.func, ast.Attribute) else link.value if isinstance(link, ast.Subscript) else None
      if recv is not None: recvs.setdefault(id(recv), []).append(k)
    for node in ast.walk(tree):
      if id(node) in nexts: self.link_step[nexts[id(node)]] = links.get(id(node))
    def wrap(fn:str, k:int, node):
      return ast.copy_location(ast.Call(ast.Name(fn, ast.Load()), [ast.Constant(k), node], []), node)
    class Wrap(ast.NodeTransformer):
      def visit(self, node):
        nid = id(node)
        node = super().visit(node)
        for k in recvs.get(nid, []): node = wrap("__tinyshape_recv__", k, node)
        if nid in nexts: node = wrap("__tinyshape_next__", nexts[nid], node)
        if nid in links: node = wrap("__tinyshape_rec__", links[nid], node)
        return node
    return ast.fix_missing_locations(Wrap().visit(tree))

  def recorder(self, store:dict, pairs:bool):
    def rec(k:int, v):
      if self.names is None:
        c = list(v) if isinstance(v, list) else v
        self.deferred.append(lambda: rec(k, c)); return v
      if self.foreign(self.links[k][0]): return v
      s = self.fmt(v, self.use_names)
      item = (s, self.fmt(v, not self.use_names)) if pairs else s
      if (pairs and s is None) or item in (seen := store.setdefault(k, [])): return v
      seen.append(item)
      return v
    return rec

  def chain_hints(self):
    def reaches_end(k:int) -> bool:
      shapes = [a for a, _ in self.link_vals.get(k, [])]
      if shapes != self.link_next.get(k): return False
      return (j := self.link_step.get(k)) is None or reaches_end(j)
    for k, vals in self.link_vals.items():
      if [a for a, _ in vals] == self.link_recv.get(k) or reaches_end(k): continue
      labels = self.hints.setdefault(self.links[k], [])
      labels.extend(p for p in vals if p not in labels)

  def run(self):
    self.g = {"__name__": "__tinyshape__", "__file__": self.path, "__builtins__": builtins}
    self.names, self.deferred = None, []  # module-level hints are formatted once the globals are known
    sys.path.insert(0, os.path.dirname(self.path))
    tree, entries = self.tree, list(self.entries())
    self.spans = [(fn.lineno, fn.end_lineno) for _, fn, _ in entries]
    if os.environ.get("TINYSHAPE_CHAIN_HINTS", "1") != "0":
      tree = self.instrument()
      self.g["__tinyshape_rec__"] = self.recorder(self.link_vals, True)
      self.g["__tinyshape_recv__"] = self.recorder(self.link_recv, False)
      self.g["__tinyshape_next__"] = self.recorder(self.link_next, False)
    for st in tree.body:
      try: self.run_traced(exec, compile(ast.Module([st], []), self.path, "exec"), self.g)
      except Exception as e: self.report_exc(e, st.lineno, "tinyshape (module level): ")
    self.names = self.dim_names()
    for f in self.deferred: f()
    for cls, fn, anns in entries:
      self.cur = (fn.lineno, fn.end_lineno)
      self.run_entry(cls, fn, anns)
    self.cur = None
    if hasattr(self, "link_vals"): self.chain_hints()
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
