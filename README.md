# tinyshape

A language server that shows [tinygrad](https://github.com/tinygrad/tinygrad) tensor shapes as inlay hints. Write down the input shape once and every line after it gets its shape.

```python
class Mlp:
  def __call__(self, x: Tensor) -> Tensor:
    # x.shape = (BS, 1, emb_dim)
    gate: (BS, 1, mlp_size) = self.gate_proj(x).silu()
    up: (BS, 1, mlp_size) = self.up_proj(x)
    return self.down_proj(gate*up) -> (BS, 1, emb_dim)
```

Chained calls get a hint after each link that changes the shape, so you can follow the transform without assigning intermediates. Each shape appears once, right after the step that produced it; the final shape is on the variable:

```python
    q: (BS, n_heads, T, head_dim) = x.reshape(B, T, n_heads, head_dim): (BS, T, n_heads, head_dim).transpose(1, 2): (BS, n_heads, T, head_dim).contiguous()
```

`return a, b` gets a hint after each element instead of one for the whole tuple.

Shape errors show up as diagnostics on the failing line, e.g. `cannot dot (BS, T, 64) and (BS, 4, T, 16)`.

## How it works

tinyshape doesn't reimplement tinygrad's shape rules. It runs your code:

- The file is executed on tinygrad's `NULL` device, where tensors are lazy. Nothing is allocated and no kernels run.
- Each annotated function is called with `Tensor.empty` inputs of the declared shape.
- `sys.settrace` records the shape of every tensor assigned or returned in your file. That includes lines reached through other functions and classes, e.g. an `Mlp` called from an annotated `Block`.

Symbolic dims like `BS` are stand-in prime numbers during the run, and are turned back into names for display. A small file with a transformer block takes about 0.1s per analysis.

## Annotations

Put these comments at the top of a function body:

| comment | meaning |
|---|---|
| `# x.shape = (BS, T, emb_dim)` | Input shape. Names that aren't module globals become symbolic dims. |
| `# x.dtype = dtypes.int` | Input dtype. Usually not needed: inputs are float, and if the call fails they're retried as int, e.g. token ids into `nn.Embedding`. |
| `# start_pos = 0` | Value for a non-tensor parameter. |
| `# self = Block(3)` | How to build `self`. The default is `Cls()`, then `Cls(0, 0, ...)`. |
| `# tinyshape: run` | Run this function even though it has no input shapes, e.g. a `def rope_table():` that builds tensors from globals. |

Code only gets hints if it runs: in an annotated function, in anything it calls, or at module level. A function that nothing calls needs `# tinyshape: run`.

A `-> Tensor` function whose body is only `pass` or `...` returns its first `Tensor` argument unchanged, so code that calls an unfinished layer still runs and gets hints.

Dims are printed the same way everywhere:

- Dims are numbers, e.g. `(1, 1, 17408)`, since the code already says which globals they come from. Symbolic dims keep their names: `(BS, T, 5120)`.
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
2. Optionally set these in `settings.json`:

```jsonc
"tinyshape.python": "/path/to/.venv/bin/python",           // python with tinygrad installed
"tinyshape.server": "/path/to/tinyshape/tinyshape/server.py", // run the checkout instead of the bundled copy
"tinyshape.nameDims": true,                                  // false never shows global names
"tinyshape.chainHints": true                                 // false hides hints inside method chains
```

Python is chosen the same way as in Zed, with `python3` from `PATH` running the server when `tinyshape.python` isn't set. Changing a setting restarts the server; `tinyshape: Restart Server` does it by hand, e.g. after editing a checkout's `server.py`. Server logs are in the `tinyshape` output channel.

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
