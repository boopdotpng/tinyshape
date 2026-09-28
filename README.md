# tinyshape

A language server that shows [tinygrad](https://github.com/tinygrad/tinygrad) tensor shapes as inlay hints. Write down the input shape once and every line after it gets its shape.

```python
class Mlp:
  def __call__(self, x: Tensor) -> Tensor:
    # x.shape = (1, 1, emb_dim)
    gate: (1, 1, 17408) = self.gate_proj(x).silu()
    up: (1, 1, 17408) = self.up_proj(x)
    return self.down_proj(gate*up) -> (1, 1, 5120)
```

Chained calls get a hint after each link that changes the shape, so you can follow the transform without assigning intermediates. Each shape appears once, right after the step that produced it; the final shape is on the variable:

```python
    q: (1, 24, 3, 256) = x.reshape(B, T, n_heads, head_dim): (1, 3, 24, 256).transpose(1, 2): (1, 24, 3, 256).contiguous()
```

`return a, b` gets a hint after each element instead of one for the whole tuple.

Shape errors show up as diagnostics on the failing line, e.g. `cannot dot (1, 3, 64) and (1, 4, 3, 16)`.

## How it works

tinyshape doesn't reimplement tinygrad's shape rules. It runs your code:

- The file is executed on tinygrad's `NULL` device, where tensors are lazy. Nothing is allocated and no kernels run.
- Each annotated function is called with `Tensor.empty` inputs of the declared shape.
- `sys.settrace` records the shape of every tensor assigned or returned in your file. That includes lines reached through other functions and classes, e.g. an `Mlp` called from an annotated `Block`.

Shape comments must resolve to concrete non-negative integers. You can use numeric literals, expressions, or defined module globals; undefined names produce a diagnostic on the comment. A small file with a transformer block takes about 0.1s per analysis.

## Annotations

Put these comments at the top of a function body:

| comment | meaning |
|---|---|
| `# x.shape = (1, 3, emb_dim)` | Input shape. Every dimension must resolve to a concrete integer; names must be defined. |
| `# x.dtype = dtypes.int` | Input dtype. Usually not needed: inputs are float, and if the call fails they're retried as int, e.g. token ids into `nn.Embedding`. |
| `# start_pos = 0` | Value for a non-tensor parameter. |
| `# self = Block(3)` | How to build `self`. The default is `Cls()`, then `Cls(0, 0, ...)`. |
| `# tinyshape: run` | Run this function even though it has no input shapes, e.g. a `def rope_table():` that builds tensors from globals. |

Code only gets hints if it runs: in an annotated function, in anything it calls, or at module level. A function that nothing calls needs `# tinyshape: run`.

A `-> Tensor` function whose body is only `pass` or `...` returns its first `Tensor` argument unchanged, so code that calls an unfinished layer still runs and gets hints.

Dims are printed the same way everywhere:

- Dims are numbers, e.g. `(1, 1, 17408)`, since the code already says which globals they come from.
- Hovering a hint shows a dim by name when exactly one module-level int has its value, e.g. `(1, 1, mlp_size)`, plus the dtype if it isn't the default float.
- If a line runs with several shapes (loops, several callers), the hint shows the first and the hover lists the others.

## Caveats

- **It runs your code.** Module-level code runs on every edit, with `__name__ != "__main__"`. Only files containing a `# name.shape =` or `# tinyshape: run` comment are analyzed, and each run has a 20s timeout.
- **Only the branch that ran is seen.** Shapes come from one real run, so a different branch can give different shapes.
- **Tensors only.** Hints are shown for tensors and for small lists or tuples of tensors.

## Zed

1. Clone the repo. In Zed, run `zed: install dev extension` and pick the `zed/` folder. This needs Rust installed through rustup.
2. Enable the server for Python in `settings.json`:

```jsonc
"languages": {
  "Python": { "language_servers": ["basedpyright", "tinyshape", "..."] }
},
"lsp": {
  "tinyshape": {
    "initialization_options": {
      "python": "/path/to/.venv/bin/python",           // optional, python with tinygrad installed
      "tinygradPath": "/path/to/tinygrad",               // optional, a tinygrad git clone, if it isn't pip installed
      "server": "/path/to/tinyshape/tinyshape/server.py", // optional, run the checkout instead of the embedded copy
      "nameDims": true,                                  // optional, false never shows global names
      "chainHints": true                                 // optional, false hides hints inside method chains
    }
  }
}
```

Choosing the Python for the analysis:

- `python` from the options is used if set.
- Otherwise it's the nearest `.venv/bin/python` above the file.
- Otherwise it's the Python running the server.

Notes:

- **Don't set `lsp.tinyshape.binary.path`.** Zed then runs that binary with no arguments and skips the extension, so `server.py` never starts.
- **Updates:** the extension embeds a copy of the Python files. With `server` pointing at a checkout, Python edits only need `editor: restart language server`. Otherwise run `zed: rebuild dev extension`. If the server doesn't come back after a rebuild, restart Zed.
- **basedpyright** shows its own `: Tensor` hint next to tinyshape's. It can't be turned off for tensors only, just for all variables with `basedpyright.analysis.inlayHints.variableTypes: false`.

## VS Code

1. Download `tinyshape-<version>.vsix` from the [releases](https://github.com/boopdotpng/tinyshape/releases) and run `code --install-extension tinyshape-<version>.vsix`, or use `Extensions: Install from VSIX...`.
2. Open a Python file with a `# x.shape = (...)` comment. If tinygrad is installed in the interpreter you picked with `Python: Select Interpreter`, hints show up and you're done.
3. If tinyshape can't find tinygrad, it shows a popup with two fixes:
   - **Set tinygrad Folder...**: pick the folder you cloned tinygrad into, the one with `setup.py` in it. This is saved for every project. It can also be run from the command palette as `tinyshape: Set tinygrad Folder`.
   - **Select Python Interpreter**: pick a Python that has tinygrad installed, e.g. after `pip install tinygrad`.

All settings, for `settings.json`:

```jsonc
"tinyshape.python": "/path/to/.venv/bin/python",           // python with tinygrad installed
"tinyshape.tinygradPath": "/path/to/tinygrad",              // a tinygrad git clone, if it isn't pip installed
"tinyshape.server": "/path/to/tinyshape/tinyshape/server.py", // run the checkout instead of the bundled copy
"tinyshape.nameDims": true,                                  // false never shows global names
"tinyshape.chainHints": true                                 // false hides hints inside method chains
```

The Python for the analysis is `tinyshape.python` if set, else the interpreter selected in the Python extension, else it's chosen the same way as in Zed, with `python3` from `PATH` running the server. Changing a setting or the selected interpreter restarts the server; `tinyshape: Restart Server` does it by hand, e.g. after editing a checkout's `server.py`. Server logs are in the `tinyshape` output channel.

To build the `.vsix` yourself: `cd vscode && npm install && npx vsce package`. This bundles a copy of the Python files from `tinyshape/`.

## Other editors

`tinyshape/server.py` is a plain stdio LSP server with no dependencies beyond the standard library. It provides inlay hints and diagnostics, so any editor that can run a custom language server for Python should work:

```sh
python tinyshape/server.py
```

You can also run the analysis without an editor:

```sh
python tinyshape/analyze.py model.py < model.py   # prints hints + diagnostics as json
```
