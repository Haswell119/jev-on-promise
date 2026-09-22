//! Deterministic text pipeline: normalization, tokenization, stemming,
//! sentence segmentation, numbers/dates, negation & modality scope.

pub mod dates;
pub mod negation;
pub mod normalize;
pub mod numbers;
pub mod segment;
pub mod stem;
pub mod tokenize;

pub use normalize::normalize_nfkc;
pub use tokenize::{tokenize, RawToken, TokenKind};
