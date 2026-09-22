//! Immutable lexical resources loaded exactly once: background IDF table,
//! sentiment valence lexicon and a WordNet-derived lexical graph.
//! All are plain TSV files embedded at compile time (see `assets/lexicon`).
//! No neural components anywhere.

pub mod graph;
pub mod idf;
pub mod sentiment;

use std::sync::{Arc, OnceLock};

pub use graph::LexGraph;
pub use idf::IdfTable;
pub use sentiment::SentimentLexicon;

/// Bundle of all lexical resources.
#[derive(Debug)]
pub struct Resources {
    pub idf_table: IdfTable,
    pub sentiment: SentimentLexicon,
    pub graph: LexGraph,
}

pub const EMBEDDED_WORD_FREQ: &str = include_str!("../../assets/lexicon/word_freq.tsv");
pub const EMBEDDED_SENTIMENT: &str = include_str!("../../assets/lexicon/sentiment.tsv");
pub const EMBEDDED_WN_LEMMAS: &str = include_str!("../../assets/lexicon/wn_lemmas.tsv");
pub const EMBEDDED_WN_SYNSETS: &str = include_str!("../../assets/lexicon/wn_synsets.tsv");
pub const EMBEDDED_WN_ANTONYMS: &str = include_str!("../../assets/lexicon/wn_antonyms.tsv");

impl Resources {
    /// Build from raw TSV contents.
    pub fn from_tsv(
        word_freq: &str,
        sentiment: &str,
        wn_lemmas: &str,
        wn_synsets: &str,
        wn_antonyms: &str,
    ) -> Resources {
        Resources {
            idf_table: IdfTable::parse(word_freq),
            sentiment: SentimentLexicon::parse(sentiment),
            graph: LexGraph::parse(wn_lemmas, wn_synsets, wn_antonyms),
        }
    }

    /// The compiled-in resources (parsed once per process).
    pub fn embedded() -> Arc<Resources> {
        static R: OnceLock<Arc<Resources>> = OnceLock::new();
        R.get_or_init(|| {
            Arc::new(Resources::from_tsv(
                EMBEDDED_WORD_FREQ,
                EMBEDDED_SENTIMENT,
                EMBEDDED_WN_LEMMAS,
                EMBEDDED_WN_SYNSETS,
                EMBEDDED_WN_ANTONYMS,
            ))
        })
        .clone()
    }

    /// Empty resources (tests / ablations).
    pub fn empty() -> Resources {
        Resources::from_tsv("", "", "", "", "")
    }

    /// Background IDF in [0, 1] for a stem (falls back to the surface form).
    #[inline]
    pub fn idf(&self, stem: &str, surface: &str) -> f32 {
        self.idf_table.idf(stem, surface)
    }
}
