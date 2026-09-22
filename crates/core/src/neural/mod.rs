//! Local neural decision scorer.
//!
//! A compact transformer encoder scores one `(question, candidate, evidence)`
//! sequence into a scalar compatibility logit; the logits of a question's
//! candidates are normalised into a distribution by the engine. Everything
//! runs in-process through candle — no Python, no service, no network.
//!
//! The artifact directory holds:
//! * `config.json`      — encoder config (HuggingFace BERT layout),
//! * `model.safetensors`— encoder weights,
//! * `tokenizer.json`   — the exact tokenizer used at training time,
//! * `head.safetensors` — pooling head weights,
//! * `scorer.json`      — Sextant-side settings (max length, pooling,
//!   whether symbolic features are consumed, fusion weights).

use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScorerConfig {
    /// Maximum sequence length (tokens) handed to the encoder.
    pub max_len: usize,
    /// `cls` or `mean`.
    pub pooling: String,
    /// Word budget used when selecting evidence (must match training).
    pub evidence_words: usize,
    /// Number of symbolic features consumed by the head (0 = none).
    #[serde(default)]
    pub n_features: usize,
    /// Whether the head consumes the symbolic fusion logit.
    #[serde(default)]
    pub use_symbolic_logit: bool,
    /// Encoder id and revision, for provenance.
    #[serde(default)]
    pub encoder: String,
    #[serde(default)]
    pub revision: String,
    #[serde(default)]
    pub version: String,
    /// Separator inserted between the question and the candidate text.
    #[serde(default = "default_sep")]
    pub pair_sep: String,
}

fn default_sep() -> String {
    " | ".to_string()
}

impl Default for ScorerConfig {
    fn default() -> Self {
        ScorerConfig {
            max_len: 192,
            pooling: "mean".into(),
            evidence_words: 140,
            n_features: 0,
            use_symbolic_logit: false,
            encoder: String::new(),
            revision: String::new(),
            version: String::new(),
            pair_sep: default_sep(),
        }
    }
}

/// Head: Linear(h + extra → h) → GELU → Linear(h → 1), optionally preceded
/// by a LayerNorm over the extra (symbolic) inputs.
struct Head {
    l1: Linear,
    l2: Linear,
    feat_norm: Option<candle_nn::LayerNorm>,
}

pub struct NeuralScorer {
    tokenizer: Tokenizer,
    model: BertModel,
    head: Head,
    device: Device,
    pub config: ScorerConfig,
    pub hidden_size: usize,
    pub n_params: usize,
}

impl std::fmt::Debug for NeuralScorer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NeuralScorer(encoder={}, hidden={}, max_len={}, pooling={})", self.config.encoder, self.hidden_size, self.config.max_len, self.config.pooling)
    }
}

fn err<E: std::fmt::Display>(ctx: &str) -> impl Fn(E) -> String + '_ {
    move |e| format!("{ctx}: {e}")
}

impl NeuralScorer {
    pub fn load(dir: &Path) -> Result<NeuralScorer, String> {
        let device = Device::Cpu;
        let cfg_text = std::fs::read_to_string(dir.join("config.json")).map_err(err("config.json"))?;
        let bert_cfg: BertConfig = serde_json::from_str(&cfg_text).map_err(err("config.json"))?;
        let scorer_cfg: ScorerConfig = match std::fs::read_to_string(dir.join("scorer.json")) {
            Ok(t) => serde_json::from_str(&t).map_err(err("scorer.json"))?,
            Err(_) => ScorerConfig::default(),
        };
        let mut tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| format!("tokenizer.json: {e}"))?;
        // Mirror the training-time tokenizer exactly: truncate the evidence
        // side only, pad to the longest sequence in the batch.
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: scorer_cfg.max_len,
                strategy: tokenizers::TruncationStrategy::OnlySecond,
                direction: tokenizers::TruncationDirection::Right,
                stride: 0,
            }))
            .map_err(|e| format!("truncation: {e}"))?;
        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            direction: tokenizers::PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id: bert_cfg.pad_token_id as u32,
            pad_type_id: 0,
            pad_token: "[PAD]".to_string(),
        }));
        // Safe (non-mmap) load: the crate keeps `#![forbid(unsafe_code)]`.
        let enc_tensors = candle_core::safetensors::load(dir.join("model.safetensors"), &device).map_err(err("model.safetensors"))?;
        let n_params_enc: usize = enc_tensors.values().map(|t| t.elem_count()).sum();
        let vb = VarBuilder::from_tensors(enc_tensors, DType::F32, &device);
        let model = BertModel::load(vb, &bert_cfg).map_err(err("encoder"))?;
        let head_tensors = candle_core::safetensors::load(dir.join("head.safetensors"), &device).map_err(err("head.safetensors"))?;
        let n_params_head: usize = head_tensors.values().map(|t| t.elem_count()).sum();
        let hvb = VarBuilder::from_tensors(head_tensors, DType::F32, &device);
        let h = bert_cfg.hidden_size;
        let extra = scorer_cfg.n_features + usize::from(scorer_cfg.use_symbolic_logit);
        let l1 = candle_nn::linear(h + extra, h, hvb.pp("l1")).map_err(err("head.l1"))?;
        let l2 = candle_nn::linear(h, 1, hvb.pp("l2")).map_err(err("head.l2"))?;
        let feat_norm = if extra > 0 { Some(candle_nn::layer_norm(extra, 1e-5, hvb.pp("feat_norm")).map_err(err("head.feat_norm"))?) } else { None };
        let n_params = n_params_enc + n_params_head;
        Ok(NeuralScorer { tokenizer, model, head: Head { l1, l2, feat_norm }, device, config: scorer_cfg, hidden_size: h, n_params })
    }

    /// Score every candidate of one question. Returns one logit per candidate.
    pub fn score(&self, question: &str, evidence: &str, candidates: &[String], features: Option<&[Vec<f32>]>, symbolic_logits: Option<&[f32]>) -> Result<Vec<f32>, String> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        let pairs: Vec<(String, String)> = candidates.iter().map(|c| (format!("{}{}{}", question, self.config.pair_sep, c), evidence.to_string())).collect();
        let encodings = self
            .tokenizer
            .encode_batch(pairs.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect::<Vec<_>>(), true)
            .map_err(|e| format!("tokenize: {e}"))?;
        // The tokenizer applied truncation and batch padding itself.
        let max_len = encodings.first().map(|e| e.len()).unwrap_or(1).max(1);
        let n = encodings.len();
        let mut ids = Vec::with_capacity(n * max_len);
        let mut types = Vec::with_capacity(n * max_len);
        let mut mask = Vec::with_capacity(n * max_len);
        for e in &encodings {
            if e.len() != max_len {
                return Err(format!("ragged batch: {} vs {}", e.len(), max_len));
            }
            ids.extend_from_slice(e.get_ids());
            types.extend_from_slice(e.get_type_ids());
            mask.extend_from_slice(e.get_attention_mask());
        }
        let ids = Tensor::from_vec(ids, (n, max_len), &self.device).map_err(err("ids"))?;
        let types = Tensor::from_vec(types, (n, max_len), &self.device).map_err(err("types"))?;
        let mask_u = Tensor::from_vec(mask, (n, max_len), &self.device).map_err(err("mask"))?;
        let hidden = self.model.forward(&ids, &types, Some(&mask_u)).map_err(err("forward"))?;
        let maskf = mask_u.to_dtype(DType::F32).map_err(err("mask f32"))?;
        let pooled = if self.config.pooling == "cls" {
            hidden.i((.., 0, ..)).map_err(err("cls"))?
        } else {
            let m = maskf.unsqueeze(2).map_err(err("unsqueeze"))?;
            let summed = hidden.broadcast_mul(&m).map_err(err("mul"))?.sum(1).map_err(err("sum"))?;
            let counts = m.sum(1).map_err(err("count"))?.clamp(1e-6, f32::MAX).map_err(err("clamp"))?;
            summed.broadcast_div(&counts).map_err(err("div"))?
        };
        let mut input = pooled;
        let extra_dim = self.config.n_features + usize::from(self.config.use_symbolic_logit);
        if extra_dim > 0 {
            let mut flat: Vec<f32> = Vec::with_capacity(n * extra_dim);
            for i in 0..n {
                if self.config.n_features > 0 {
                    let f = features.and_then(|f| f.get(i)).ok_or("neural head expects symbolic features")?;
                    if f.len() != self.config.n_features {
                        return Err(format!("expected {} features, got {}", self.config.n_features, f.len()));
                    }
                    flat.extend_from_slice(f);
                }
                if self.config.use_symbolic_logit {
                    flat.push(symbolic_logits.and_then(|s| s.get(i)).copied().unwrap_or(0.0));
                }
            }
            let e = Tensor::from_vec(flat, (n, extra_dim), &self.device).map_err(err("extra"))?;
            let e = match &self.head.feat_norm {
                Some(ln) => ln.forward(&e).map_err(err("feat_norm"))?,
                None => e,
            };
            input = Tensor::cat(&[&input, &e], 1).map_err(err("cat"))?;
        }
        let x = self.head.l1.forward(&input).map_err(err("l1"))?;
        let x = x.gelu_erf().map_err(err("gelu"))?;
        let out = self.head.l2.forward(&x).map_err(err("l2"))?;
        let v = out.squeeze(1).map_err(err("squeeze"))?.to_vec1::<f32>().map_err(err("to_vec"))?;
        Ok(v)
    }
}

use candle_core::IndexOp;
