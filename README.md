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

Dims are printed the way you wrote the annotation:

- If it names a module-level int, like `(1, 1, emb_dim)`, a dim equal to exactly one module-level int is shown by name: `(1, 1, mlp_size)`.
- If it's all numbers, like `(1, 1, 5120)`, numbers are shown: `(1, 1, 17408)`.
- Hovering a hint shows the other form.
- If a line runs with several shapes (loops, several callers), they're joined with `|`.

## Caveats

- **It runs your code.** Module-level code runs on every edit, with `__name__ != "__main__"`. Only files containing a `# name.shape =` comment are analyzed, and each run has a 20s timeout.
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

## Other editors

`tinyshape/server.py` is a plain stdio LSP server with no dependencies beyond the standard library. It provides inlay hints and diagnostics, so any editor that can run a custom language server for Python should work:

```sh
python tinyshape/server.py
```

You can also run the analysis without an editor:

```sh
python tinyshape/analyze.py model.py < model.py   # prints hints + diagnostics as json
```
