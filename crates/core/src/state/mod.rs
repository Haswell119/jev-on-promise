//! Shared state representation: JSON flattening, vocabulary interning and
//! the per-request `StateIndex` that every question reuses.

pub mod flatten;
pub mod index;
pub mod vocab;

pub use flatten::{flatten_state, flatten_state_with_arrays, ArrayInfo, FieldKind, FlatField};
pub use index::{Segment, StateIndex, Token, ATTR_CAP, ATTR_FUNC, ATTR_KEY, ATTR_STOP};
pub use vocab::{TermId, Vocab, VocabExt};
