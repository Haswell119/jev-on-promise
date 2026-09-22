use sextant_core::{Engine, EngineConfig, Model, Resources};
use std::path::PathBuf;
use std::sync::Arc;

pub fn load_model(model_dir: Option<PathBuf>) -> Result<Model, String> {
    match model_dir {
        Some(dir) => Model::load_dir(&dir),
        None => Ok(Model::embedded()),
    }
}

pub fn build_engine(model_dir: Option<PathBuf>, threads: usize) -> Result<Arc<Engine>, String> {
    let model = load_model(model_dir)?;
    Ok(Arc::new(Engine::new(Resources::embedded(), Arc::new(model), EngineConfig { threads, ..Default::default() })))
}

pub fn read_input(file: Option<PathBuf>) -> Result<String, String> {
    match file {
        Some(p) if p.as_os_str() != "-" => std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display())),
        _ => {
            let mut s = String::new();
            std::io::Read::read_to_string(&mut std::io::stdin(), &mut s).map_err(|e| format!("stdin: {e}"))?;
            Ok(s)
        }
    }
}

/// Expand files / directories into a sorted list of *.jsonl paths.
pub fn expand_jsonl(inputs: &[PathBuf]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for p in inputs {
        if p.is_dir() {
            let mut entries: Vec<PathBuf> =
                std::fs::read_dir(p).map(|rd| rd.flatten().map(|e| e.path()).collect()).unwrap_or_default();
            entries.sort();
            for e in entries {
                if e.is_dir() {
                    out.extend(expand_jsonl(&[e]));
                } else if e.extension().map(|x| x == "jsonl").unwrap_or(false) {
                    out.push(e);
                }
            }
        } else {
            out.push(p.clone());
        }
    }
    out
}
