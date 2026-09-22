//! Question analysis: instruction flattening, path references, reference
//! data, criterion flattening with field-polarity semantics, numeric range
//! parsing and the cheap question-family detector.

pub mod analysis;
pub mod criteria;
pub mod family;
pub mod numeric;

pub use analysis::{PathRef, QuestionView};
pub use criteria::{Criterion, Phrase, QueryTerm};
pub use family::Family;
pub use numeric::NumericRange;
