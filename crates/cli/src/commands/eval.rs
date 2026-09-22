#![allow(clippy::too_many_arguments)]
use std::path::PathBuf;
pub fn run(_model_dir: Option<PathBuf>, _threads: usize, _inputs: Vec<PathBuf>, _raw_out: Option<PathBuf>, _out: Option<PathBuf>, _group_by: &str, _limit: usize, _show_failures: bool, _filter: Option<String>) -> i32 {
    eprintln!("eval: not implemented yet");
    2
}
