import json
from pathlib import Path
import subprocess
import sys
import unittest


ANALYZER = Path(__file__).resolve().parents[1] / "tinyshape" / "analyze.py"


def analyze(source):
  result = subprocess.run([sys.executable, str(ANALYZER), "/tmp/tinyshape_test.py"],
                          input=source, text=True, capture_output=True, check=True)
  return json.loads(result.stdout)


class ConcreteShapes(unittest.TestCase):
  def test_undefined_name_reports_comment_and_other_entries_still_run(self):
    result = analyze("""from tinygrad import Tensor
def invalid(x):
  # x.shape = (B, S)
  return x
def valid(x):
  # x.shape = (2, 3)
  return x
""")
    self.assertEqual(len(result["diagnostics"]), 1)
    self.assertEqual(result["diagnostics"][0]["line"], 2)
    self.assertIn("name 'B' is not defined", result["diagnostics"][0]["message"])
    self.assertEqual([h["label"] for h in result["hints"]], ["(2, 3)"])

  def test_defined_globals_and_expressions_keep_numeric_hints(self):
    result = analyze("""from tinygrad import Tensor, nn
B, S, emb_dim = 2, 3, 5120
lin_conv_dim = 10240
def project(x):
  # x.shape = (B, S, emb_dim)
  qkv = nn.Linear(emb_dim, lin_conv_dim, bias=False)(x)
  return qkv
def reshape(x):
  # x.shape = (B, S * 2)
  return x.reshape(B, 2, S)
""")
    self.assertEqual(result["diagnostics"], [])
    self.assertIn("(2, 3, 10240)", [h["label"] for h in result["hints"]])
    self.assertIn("(2, 2, 3)", [h["label"] for h in result["hints"]])

  def test_invalid_dimensions_report_comment(self):
    for shape in ["(2, 3.5)", "(2, -1)", "(True, 3)", "('S', 3)"]:
      with self.subTest(shape=shape):
        result = analyze(f"from tinygrad import Tensor\ndef f(x):\n  # x.shape = {shape}\n  return x\n")
        self.assertEqual(result["hints"], [])
        self.assertEqual(result["diagnostics"][0]["line"], 2)
        self.assertIn("concrete non-negative integers", result["diagnostics"][0]["message"])

  def test_scalar_and_empty_shapes(self):
    for shape, label in [("3", "(3,)"), ("()", "()"), ("(0, 3)", "(0, 3)")]:
      with self.subTest(shape=shape):
        result = analyze(f"from tinygrad import Tensor\ndef f(x):\n  # x.shape = {shape}\n  return x\n")
        self.assertEqual(result["diagnostics"], [])
        self.assertEqual(result["hints"][0]["label"], label)

  def test_realize_and_assign_are_skipped_but_checked(self):
    result = analyze("""from tinygrad import Tensor, dtypes
def ok(x):
  # x.shape = (2, 3)
  cache = Tensor.zeros(4, 3).contiguous().realize()
  cache[:2].assign(x)
  cache.assign(1.0)
  return cache
def bad_shape(x):
  # x.shape = (2, 3)
  Tensor.zeros(2, 4).realize().assign(x)
def bad_dtype(x):
  # x.shape = (2, 3)
  Tensor.zeros(2, 3, dtype=dtypes.half).assign(x)
""")
    self.assertIn("(4, 3)", [h["label"] for h in result["hints"]])
    self.assertEqual(sorted(d["line"] for d in result["diagnostics"]), [9, 12])
    self.assertIn("dtype mismatch", result["diagnostics"][-1]["message"])

  def test_repeated_calls_keep_per_layer_differences(self):
    # blocks 0-2 repeat, block 3 has a different flag, block 4 different weights; each one's output shape must still be right.
    # the layer list is built with the same Mlp() call 5 times, so __init__ must not be skipped
    result = analyze("""from tinygrad import Tensor, nn
class Mlp:
  def __init__(self, d): self.lin = nn.Linear(4, d)
  def __call__(self, x): return self.lin(x)
class Block:
  def __init__(self, i): self.wide, self.mlp = i == 3, Mlp(6 if i == 4 else 4)
  def __call__(self, x):
    y = self.mlp(x)
    return y.cat(y, dim=-1) if self.wide else y
class Model:
  def __init__(self): self.layers = [Block(i) for i in range(5)]
  def __call__(self, x):
    # x.shape = (2, 4)
    outs = [layer(x) for layer in self.layers]
    return outs[2], outs[3], outs[4]
""")
    self.assertEqual(result["diagnostics"], [])
    self.assertEqual([h["label"] for h in result["hints"] if h["line"] == 14], ["(2, 4)", "(2, 8)", "(2, 6)"])

  def test_repeated_call_with_new_shape_still_errors(self):
    result = analyze("""from tinygrad import Tensor
def f(x): return x.reshape(2, 3)
def g(x, y):
  # x.shape = (6,)
  # y.shape = (6,)
  a, b, c = f(x), f(y), f(x.cat(x))
  return a
""")
    self.assertEqual(len(result["diagnostics"]), 1)
    self.assertEqual(result["diagnostics"][0]["line"], 1)


if __name__ == "__main__":
  unittest.main()
