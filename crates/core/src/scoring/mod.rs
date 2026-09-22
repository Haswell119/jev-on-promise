//! Learned fusion (linear / logistic heads with per-family experts),
//! probability normalization, ordinal smoothing and confidence.

pub mod confidence;
pub mod heads;
pub mod softmax;

pub use confidence::{confidence, ConfidenceParams};
pub use heads::{DenseHead, DenseNoul, DenseWeights, Head, NoulHead, Weights};
pub use softmax::{entropy_normalized, ordinal_smooth, softmax_temp};
