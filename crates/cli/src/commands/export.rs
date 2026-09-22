use sextant_core::calibration::Calibration;
use sextant_core::scoring::Weights;
use std::path::PathBuf;

pub fn run(out_dir: PathBuf) -> i32 {
    if let Err(e) = std::fs::create_dir_all(&out_dir) {
        eprintln!("error: {e}");
        return 1;
    }
    let w = serde_json::to_string_pretty(&Weights::bootstrap()).unwrap();
    let c = serde_json::to_string_pretty(&Calibration::default()).unwrap();
    std::fs::write(out_dir.join("weights.json"), w + "\n").unwrap();
    std::fs::write(out_dir.join("calibration.json"), c + "\n").unwrap();
    println!("wrote {}/weights.json and calibration.json (bootstrap, untrained)", out_dir.display());
    0
}
