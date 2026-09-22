//! # Sextant core
//!
//! A non-neural, deterministic, calibrated probabilistic decision engine
//! exposing three typed primitives over a shared textual/structured state:
//!
//! * **Noul** – calibrated probability that a proposition is true;
//! * **Choice** – a probability distribution over a closed option set;
//! * **Score** – an ordinal distribution over described levels and its
//!   expected value.
//!
//! Pipeline: validation → state normalization → shared state index →
//! question analysis → evidence retrieval → symbolic resolvers →
//! multi-channel feature extraction → learned linear fusion → probability
//! normalization → calibration → typed answer.
//!
//! No language models, no neural networks, no network access.

#![forbid(unsafe_code)]

pub mod api;
pub mod calibration;
pub mod engine;
pub mod export;
pub mod features;
pub mod lexicon;
pub mod model;
#[cfg(feature = "neural")]
pub mod neural;
pub mod question;
pub mod resolvers;
pub mod scoring;
pub mod state;
pub mod text;

pub use api::{Answer, ApiError, Question, QuestionKind, SystemOneRequest, SystemOneResponse};
pub use engine::{Engine, EngineConfig};
pub use lexicon::Resources;
pub use model::Model;

/// Crate version (also reported by `GET /health`).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
