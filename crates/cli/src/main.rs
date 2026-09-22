//! `sextant` command line: serve, decide, validate, bench, eval, train,
//! calibrate, leakage, export-model.

mod commands;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "sextant",
    version,
    about = "Sextant: non-neural, deterministic, calibrated decision engine (Choice / Score / Noul)"
)]
struct Cli {
    /// Directory holding weights.json + calibration.json (default: embedded artifact).
    #[arg(long, global = true, env = "SEXTANT_MODEL_DIR")]
    model_dir: Option<PathBuf>,
    /// Worker threads for question evaluation (0 = all CPUs).
    #[arg(long, global = true, default_value_t = 0, env = "SEXTANT_THREADS")]
    threads: usize,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run the HTTP server.
    Serve {
        #[arg(long, default_value = "0.0.0.0:8080", env = "SEXTANT_ADDR")]
        addr: String,
        /// Maximum request body size in bytes.
        #[arg(long, default_value_t = 4 * 1024 * 1024, env = "SEXTANT_MAX_BODY")]
        max_body: usize,
    },
    /// Evaluate a request JSON file (or stdin when omitted / `-`).
    Decide {
        file: Option<PathBuf>,
        /// Attach explanations.
        #[arg(long)]
        explain: bool,
        /// Pretty-print the JSON output.
        #[arg(long)]
        pretty: bool,
    },
    /// Validate a request JSON file (or stdin) without evaluating it.
    Validate { file: Option<PathBuf> },
    /// Latency / throughput benchmark on synthetic requests.
    Bench {
        /// Iterations per scenario.
        #[arg(long, default_value_t = 200)]
        iters: usize,
        /// Write a JSON report to this path.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Only run scenarios whose name contains this substring.
        #[arg(long)]
        filter: Option<String>,
    },
    /// Evaluate the engine on a JSONL dataset of {request, gold} records.
    Eval {
        /// One or more JSONL files (or directories of *.jsonl).
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
        /// Write per-record outputs (JSONL).
        #[arg(long)]
        raw_out: Option<PathBuf>,
        /// Write a metrics JSON report.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Group metrics by this record field (e.g. `tier`, `family`, `source`).
        #[arg(long, default_value = "tier")]
        group_by: String,
        /// Limit the number of records (0 = all).
        #[arg(long, default_value_t = 0)]
        limit: usize,
        /// Print each failure with evidence.
        #[arg(long)]
        show_failures: bool,
        /// Only records whose id contains this substring.
        #[arg(long)]
        filter: Option<String>,
    },
    /// Fit the fusion weights from allowed training data (writes weights.json).
    Train {
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
        #[arg(long, default_value = "model")]
        out_dir: PathBuf,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// L2 regularization strength.
        #[arg(long, default_value_t = 0.001)]
        l2: f32,
        /// Shrinkage of per-family deltas toward the base expert.
        #[arg(long, default_value_t = 0.001)]
        family_l2: f32,
        #[arg(long, default_value_t = 600)]
        epochs: usize,
        /// Disable per-family experts.
        #[arg(long)]
        no_families: bool,
        /// Zero these features (ablation), comma separated.
        #[arg(long)]
        drop_features: Option<String>,
        /// Loss balancing across (primitive, source) groups: sqrt | full | none.
        #[arg(long, default_value = "full")]
        balance: String,
        /// JSON file with a list of record ids to exclude (from `sextant leakage --exclusions-out`).
        #[arg(long)]
        exclude_ids: Option<PathBuf>,
    },
    /// Fit calibration (temperatures, Platt, confidence map) on a separate split (writes calibration.json).
    Calibrate {
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
        #[arg(long, default_value = "model")]
        out_dir: PathBuf,
        /// Use isotonic regression for Noul instead of Platt (report only; artifact keeps Platt).
        #[arg(long)]
        compare_isotonic: bool,
    },
    /// Contamination scan: training/calibration data vs evaluation data.
    Leakage {
        /// Training / calibration JSONL files or directories.
        #[arg(long, required = true, num_args = 1..)]
        train: Vec<PathBuf>,
        /// Evaluation JSONL files or directories (JevBench / jev-bench / internal eval).
        #[arg(long, required = true, num_args = 1..)]
        eval: Vec<PathBuf>,
        #[arg(long, default_value = "reports/leakage.json")]
        out: PathBuf,
        /// Write the ids of training records whose STATE matches an external eval set (for `train --exclude-ids`).
        #[arg(long)]
        exclusions_out: Option<PathBuf>,
        /// Eval files whose matches do not count toward the verdict or exclusions (internal dev sets), comma separated substrings.
        #[arg(long, default_value = "data/synthetic")]
        internal: String,
    },
    /// Export neural training/inference inputs (question, evidence, candidates, features).
    ExportPairs {
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
        #[arg(long)]
        out: PathBuf,
        /// Word budget for the retrieved evidence block.
        #[arg(long, default_value_t = 140)]
        budget_words: usize,
        /// Limit records per input file (0 = all).
        #[arg(long, default_value_t = 0)]
        limit: usize,
        /// Omit the symbolic feature vectors (smaller files).
        #[arg(long)]
        no_features: bool,
        /// Evidence selection strategy: quota (per-candidate reservations) or pooled.
        #[arg(long, default_value = "quota")]
        strategy: String,
        /// Grow the evidence budget with the state length, up to this cap (0 = fixed).
        #[arg(long, default_value_t = 0)]
        adaptive_cap: usize,
    },
    /// Score exported pair rows with the Rust neural runtime (parity / evaluation).
    NeuralProbe {
        #[arg(long)]
        neural_dir: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 0)]
        limit: usize,
    },
    /// Write the bootstrap (hand-set) weights and default calibration to a directory.
    ExportBootstrap {
        #[arg(long, default_value = "model")]
        out_dir: PathBuf,
    },
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env().add_directive("info".parse().unwrap()))
        .with_target(false)
        .init();
    let cli = Cli::parse();
    let code = match cli.cmd {
        Cmd::Serve { addr, max_body } => commands::serve::run(cli.model_dir, cli.threads, &addr, max_body),
        Cmd::Decide { file, explain, pretty } => {
            commands::decide::run(cli.model_dir, cli.threads, file, explain, pretty)
        }
        Cmd::Validate { file } => commands::decide::validate(file),
        Cmd::Bench { iters, out, filter } => commands::bench::run(cli.model_dir, cli.threads, iters, out, filter),
        Cmd::Eval { inputs, raw_out, out, group_by, limit, show_failures, filter } => commands::eval::run(
            cli.model_dir,
            cli.threads,
            inputs,
            raw_out,
            out,
            &group_by,
            limit,
            show_failures,
            filter,
        ),
        Cmd::Train {
            inputs,
            out_dir,
            seed,
            l2,
            family_l2,
            epochs,
            no_families,
            drop_features,
            balance,
            exclude_ids,
        } => commands::train::run(
            inputs,
            out_dir,
            seed,
            l2,
            family_l2,
            epochs,
            no_families,
            drop_features,
            balance,
            exclude_ids,
        ),
        Cmd::Calibrate { inputs, out_dir, compare_isotonic } => {
            commands::calibrate::run(inputs, out_dir, compare_isotonic)
        }
        Cmd::Leakage { train, eval, out, exclusions_out, internal } => {
            commands::leakage::run(train, eval, out, exclusions_out, internal)
        }
        Cmd::ExportPairs { inputs, out, budget_words, limit, no_features, strategy, adaptive_cap } => {
            commands::export_pairs::run(cli.model_dir, cli.threads, inputs, out, budget_words, limit, no_features, strategy, adaptive_cap)
        }
        Cmd::NeuralProbe { neural_dir, input, out, limit } => commands::neural_probe::run(neural_dir, input, out, limit),
        Cmd::ExportBootstrap { out_dir } => commands::export::run(out_dir),
    };
    std::process::exit(code);
}
