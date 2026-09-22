//! Public request/response contract (TypeSafe-inspired System One shape) and validation.

pub mod error;
pub mod types;
pub mod validate;

pub use error::ApiError;
pub use types::*;
