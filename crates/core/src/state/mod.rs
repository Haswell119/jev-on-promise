//! Shared state representation: JSON flattening, vocabulary interning and
//! the per-request `StateIndex` that every question reuses.

pub mod flatten;
pub mod index;
pub mod vocab;

pub use flatten::{flatten_state, flatten_state_with_arrays, ArrayInfo, FlatField, FieldKind};
pub use index::{Segment, StateIndex, Token, ATTR_KEY, ATTR_STOP, ATTR_FUNC, ATTR_CAP};
pub use vocab::{TermId, Vocab, VocabExt};
