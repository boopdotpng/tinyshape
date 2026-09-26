// the python server is embedded and written to the extension's work dir, then run with
// `lsp.tinyshape.initialization_options.python` (a python that has tinygrad) or python3 from PATH.
use zed_extension_api::{self as zed, settings::LspSettings, LanguageServerId, Result};

const SERVER: &str = include_str!("../../tinyshape/server.py");
const ANALYZE: &str = include_str!("../../tinyshape/analyze.py");

struct TinyShape;

impl zed::Extension for TinyShape {
  fn new() -> Self { TinyShape }

  fn language_server_command(&mut self, id: &LanguageServerId, worktree: &zed::Worktree) -> Result<zed::Command> {
    std::fs::create_dir_all("tinyshape").map_err(|e| e.to_string())?;
    std::fs::write("tinyshape/server.py", SERVER).map_err(|e| e.to_string())?;
    std::fs::write("tinyshape/analyze.py", ANALYZE).map_err(|e| e.to_string())?;
    let server = std::env::current_dir().map_err(|e| e.to_string())?.join("tinyshape/server.py");
    // not `binary.path`: when that is set zed runs it directly and never calls this function
    let settings = LspSettings::for_worktree(id.as_ref(), worktree);
    let debug = format!("{:?}", settings.as_ref().map(|s| &s.initialization_options));
    let opts = settings.ok().and_then(|s| s.initialization_options);
    let python = opts.as_ref().and_then(|o| o.get("python")).and_then(|p| p.as_str()).map(String::from)
      .or_else(|| worktree.which("python3"))
      .ok_or("tinyshape: set lsp.tinyshape.initialization_options.python to a python with tinygrad installed")?;
    // `server` points at a checkout's server.py, so python edits don't need an extension rebuild
    let server = opts.as_ref().and_then(|o| o.get("server")).and_then(|p| p.as_str()).map(String::from)
      .unwrap_or_else(|| server.to_string_lossy().into_owned());
    let args = vec![server];
    let mut env = worktree.shell_env();
    env.push(("TINYSHAPE_SETTINGS".into(), debug));
    Ok(zed::Command { command: python, args, env })
  }

  fn language_server_initialization_options(&mut self, id: &LanguageServerId, worktree: &zed::Worktree) -> Result<Option<zed::serde_json::Value>> {
    Ok(LspSettings::for_worktree(id.as_ref(), worktree).ok().and_then(|s| s.initialization_options))
  }
}

zed::register_extension!(TinyShape);
