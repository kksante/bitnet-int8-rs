//! BitNet b1.58 2B4T forward pass (W1.58A8), a faithful port of the verified
//! numpy reference. Float-faithful: ternary weights are dequantized, the model's
//! own per-token int8 absmax activation quantization is applied at every
//! BitLinear (this is part of the trained model), and the matmul accumulation is
//! integer-exact. Norms, RoPE and softmax are computed in f32.
//!
//! See reference/FINDINGS.md for the integer-path characterization; porting the
//! integer attention/softmax variants studied there is a mechanical change to
//! `attention` and `BitLinear::forward`.

use crate::gguf::Gguf;
use rayon::prelude::*;
use std::io;

pub struct Config {
    pub n_layer: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub rope_base: f32,
    pub rms_eps: f32,
    pub vocab: usize,
    pub bos: u32,
    pub eos: u32,
}

/// A ternary BitLinear projection. `codes` is row-major [out, in] in {-1,0,1};
/// the real weight is `codes * scale`. Weights are stored unpacked (int8) for
/// speed: on a CPU with spare RAM this decode is memory-stalled, and unpacking is
/// fast. A 2-bit-packed store (4x smaller footprint, ~2x slower with a scalar
/// unpack) was measured and discussed in the paper (the memory--compute boundary);
/// recovering both needs a SIMD shuffle-unpack kernel.
pub struct BitLinear {
    pub codes: Vec<i8>,
    pub scale: f32,
    pub n_in: usize,
    pub n_out: usize,
}

impl BitLinear {
    /// y = (quant_int8(x) · ternary) * scale / act_scale, per the W1.58A8 model.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.n_in);
        // per-token absmax int8 activation quant
        let mut amax = 1e-5f32;
        for &v in x { amax = amax.max(v.abs()); }
        let s = 127.0 / amax;
        // int8 activations so the inner loop is an int8*int8 dot (NEON-friendly
        // with `-C target-cpu=native`).
        let mut xq = vec![0i8; self.n_in];
        for i in 0..self.n_in {
            xq[i] = (x[i] * s).round().clamp(-128.0, 127.0) as i8;
        }
        let mut out = vec![0f32; self.n_out];
        let inv = self.scale / s;
        let n_in = self.n_in;
        let codes = &self.codes;
        let xq = &xq;
        // parallel over output channels; branchless int8 multiply-accumulate.
        out.par_iter_mut().enumerate().for_each(|(o, y)| {
            let row = &codes[o * n_in..(o + 1) * n_in];
            let mut acc: i32 = 0;             // |acc| <= n_in*127 < 2^31
            for i in 0..n_in {
                acc += (row[i] as i32) * (xq[i] as i32);
            }
            *y = acc as f32 * inv;
        });
        out
    }
}

pub struct Layer {
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub attn_sub: Vec<f32>,
    pub ffn_sub: Vec<f32>,
    pub q: BitLinear,
    pub k: BitLinear,
    pub v: BitLinear,
    pub o: BitLinear,
    pub gate: BitLinear,
    pub up: BitLinear,
    pub down: BitLinear,
    // KV cache: each entry is [n_kv * head_dim]
    pub kc: Vec<Vec<f32>>,
    pub vc: Vec<Vec<f32>>,
}

pub struct Model {
    pub cfg: Config,
    pub embed: Vec<f32>, // [vocab * d]
    pub out_norm: Vec<f32>,
    pub layers: Vec<Layer>,
    pub pos: usize,
}

impl Model {
    pub fn load(path: &str) -> io::Result<Self> {
        let g = Gguf::open(path)?;
        // arch-agnostic: read the metadata key prefix from general.architecture
        let a = g.meta_str("general.architecture").unwrap_or("bitnet-b1.58").to_string();
        let a = a.as_str();
        let cfg = Config {
            n_layer: g.meta_u64(&format!("{a}.block_count")).unwrap() as usize,
            d_model: g.meta_u64(&format!("{a}.embedding_length")).unwrap() as usize,
            d_ff: g.meta_u64(&format!("{a}.feed_forward_length")).unwrap() as usize,
            n_head: g.meta_u64(&format!("{a}.attention.head_count")).unwrap() as usize,
            n_kv: g.meta_u64(&format!("{a}.attention.head_count_kv")).unwrap() as usize,
            head_dim: g.meta_u64(&format!("{a}.rope.dimension_count")).unwrap() as usize,
            rope_base: g.meta_f32(&format!("{a}.rope.freq_base")).unwrap(),
            rms_eps: g.meta_f32(&format!("{a}.attention.layer_norm_rms_epsilon")).unwrap(),
            vocab: g.meta_u64(&format!("{a}.vocab_size")).unwrap() as usize,
            bos: g.meta_u32("tokenizer.ggml.bos_token_id").unwrap_or(128000),
            eos: g.meta_u32("tokenizer.ggml.eos_token_id").unwrap_or(128001),
        };
        let embed = g.load_f16("token_embd.weight")?;
        let out_norm = g.load_f32("output_norm.weight")?;
        let mk = |g: &Gguf, name: &str| -> io::Result<BitLinear> {
            let (codes, scale, n_in, n_out) = g.load_i2s(name)?;
            Ok(BitLinear { codes, scale, n_in, n_out })
        };
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let p = format!("blk.{i}.");
            layers.push(Layer {
                attn_norm: g.load_f32(&format!("{p}attn_norm.weight"))?,
                ffn_norm: g.load_f32(&format!("{p}ffn_norm.weight"))?,
                attn_sub: g.load_f32(&format!("{p}attn_sub_norm.weight"))?,
                ffn_sub: g.load_f32(&format!("{p}ffn_sub_norm.weight"))?,
                q: mk(&g, &format!("{p}attn_q.weight"))?,
                k: mk(&g, &format!("{p}attn_k.weight"))?,
                v: mk(&g, &format!("{p}attn_v.weight"))?,
                o: mk(&g, &format!("{p}attn_output.weight"))?,
                gate: mk(&g, &format!("{p}ffn_gate.weight"))?,
                up: mk(&g, &format!("{p}ffn_up.weight"))?,
                down: mk(&g, &format!("{p}ffn_down.weight"))?,
                kc: Vec::new(),
                vc: Vec::new(),
            });
        }
        Ok(Self { cfg, embed, out_norm, layers, pos: 0 })
    }

    pub fn reset(&mut self) {
        for l in &mut self.layers { l.kc.clear(); l.vc.clear(); }
        self.pos = 0;
    }

    /// One decode step. Returns logits over the vocabulary.
    pub fn forward(&mut self, token: usize) -> Vec<f32> {
        let c = &self.cfg;
        let (d, hd, nh, nkv) = (c.d_model, c.head_dim, c.n_head, c.n_kv);
        let rep = nh / nkv;
        let eps = c.rms_eps;
        let pos = self.pos;

        // rope tables for this position
        let mut cosv = vec![0f32; hd / 2];
        let mut sinv = vec![0f32; hd / 2];
        for i in 0..hd / 2 {
            let freq = (c.rope_base).powf(-((2 * i) as f32) / hd as f32);
            let ang = pos as f32 * freq;
            cosv[i] = ang.cos();
            sinv[i] = ang.sin();
        }

        let mut x = self.embed[token * d..(token + 1) * d].to_vec();

        for li in 0..self.layers.len() {
            // ---- attention ----
            let h = rmsnorm(&x, &self.layers[li].attn_norm, eps);
            let mut q = self.layers[li].q.forward(&h);
            let mut k = self.layers[li].k.forward(&h);
            let v = self.layers[li].v.forward(&h);
            for head in 0..nh { rope_inplace(&mut q[head * hd..(head + 1) * hd], &cosv, &sinv); }
            for head in 0..nkv { rope_inplace(&mut k[head * hd..(head + 1) * hd], &cosv, &sinv); }

            self.layers[li].kc.push(k);
            self.layers[li].vc.push(v);
            let tcur = self.layers[li].kc.len();

            let mut attn = vec![0f32; d];
            let scl = 1.0 / (hd as f32).sqrt();
            let mut scores = vec![0f32; tcur];
            for head in 0..nh {
                let g = head / rep;
                let qh = &q[head * hd..(head + 1) * hd];
                // scores
                let mut maxs = f32::NEG_INFINITY;
                for t in 0..tcur {
                    let kt = &self.layers[li].kc[t][g * hd..(g + 1) * hd];
                    let mut dot = 0f32;
                    for i in 0..hd { dot += qh[i] * kt[i]; }
                    scores[t] = dot * scl;
                    if scores[t] > maxs { maxs = scores[t]; }
                }
                // softmax
                let mut sum = 0f32;
                for t in 0..tcur { scores[t] = (scores[t] - maxs).exp(); sum += scores[t]; }
                let inv = 1.0 / sum;
                // weighted sum of V
                let oh = &mut attn[head * hd..(head + 1) * hd];
                for t in 0..tcur {
                    let p = scores[t] * inv;
                    let vt = &self.layers[li].vc[t][g * hd..(g + 1) * hd];
                    for i in 0..hd { oh[i] += p * vt[i]; }
                }
            }
            let attn = rmsnorm(&attn, &self.layers[li].attn_sub, eps);
            let o = self.layers[li].o.forward(&attn);
            for i in 0..d { x[i] += o[i]; }

            // ---- ffn (squared ReLU) ----
            let h2 = rmsnorm(&x, &self.layers[li].ffn_norm, eps);
            let gate = self.layers[li].gate.forward(&h2);
            let up = self.layers[li].up.forward(&h2);
            let mut act = vec![0f32; self.cfg.d_ff];
            for i in 0..self.cfg.d_ff {
                let r = gate[i].max(0.0);
                act[i] = r * r * up[i];
            }
            let act = rmsnorm(&act, &self.layers[li].ffn_sub, eps);
            let down = self.layers[li].down.forward(&act);
            for i in 0..d { x[i] += down[i]; }
        }

        self.pos += 1;
        let hn = rmsnorm(&x, &self.out_norm, eps);
        // tied lm_head: logits[v] = hn · embed[v]; parallel over the vocabulary.
        let embed = &self.embed;
        let hn = &hn;
        let mut logits = vec![0f32; self.cfg.vocab];
        logits.par_iter_mut().enumerate().for_each(|(vtok, l)| {
            let row = &embed[vtok * d..(vtok + 1) * d];
            let mut s = 0f32;
            for i in 0..d { s += hn[i] * row[i]; }
            *l = s;
        });
        logits
    }
}

fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut ss = 0f64;
    for &v in x { ss += (v as f64) * (v as f64); }
    let denom = ((ss / n as f64).sqrt() + eps as f64) as f32;
    let mut out = vec![0f32; n];
    for i in 0..n { out[i] = x[i] / denom * w[i]; }
    out
}

/// HF-style rotate_half RoPE on one head vector (length hd), in place.
fn rope_inplace(x: &mut [f32], cosv: &[f32], sinv: &[f32]) {
    let half = x.len() / 2;
    for i in 0..half {
        let a = x[i];
        let b = x[i + half];
        x[i] = a * cosv[i] - b * sinv[i];
        x[i + half] = b * cosv[i] + a * sinv[i];
    }
}
