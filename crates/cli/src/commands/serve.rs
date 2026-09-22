use super::common::build_engine;
use std::path::PathBuf;

pub fn run(model_dir: Option<PathBuf>, threads: usize, addr: &str, max_body: usize) -> i32 {
    let engine = match build_engine(model_dir, threads) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("error: {e}");
            return 2;
        }
    };
    let addr: std::net::SocketAddr = match addr.parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: invalid address `{addr}`: {e}");
            return 2;
        }
    };
    tracing::info!(model = %engine.model.source, "loaded model artifact");
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().expect("tokio runtime");
    match rt.block_on(sextant_server::serve(engine, addr, max_body)) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("error: {e}");
            1
        }
    }
}
