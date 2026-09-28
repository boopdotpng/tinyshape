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


if __name__ == "__main__":
  unittest.main()
