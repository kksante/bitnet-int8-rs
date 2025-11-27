// src/model/attention.rs
use ndarray::{Array2, ArrayView2};
use crate::model::softmax::IntSoftmax;

/// Pure integer multi-head attention
/// 
/// Computes: softmax(Q @ K.T / sqrt(d_k)) @ V
/// All operations in int8/u16, zero floating point
pub struct Attention {
    n_heads: usize,
    head_dim: usize,
    softmax: IntSoftmax,
}

impl Attention {
    pub fn new(n_heads: usize, head_dim: usize) -> Self {
        Self {
            n_heads,
            head_dim,
            softmax: IntSoftmax::new(head_dim),
        }
    }

    /// Forward pass
    /// Q, K, V: [seq, dim] where dim = n_heads * head_dim
    /// Returns: [seq, dim]
    pub fn forward(
        &self,
        q: ArrayView2<i8>,
        k: ArrayView2<i8>,
        v: ArrayView2<i8>,
    ) -> Array2<i8> {
        let seq = q.nrows();
        let dim = q.ncols();
        let mut output = Array2::<i8>::zeros((seq, dim));

        // Process each head
        for h in 0..self.n_heads {
            let start = h * self.head_dim;
            let end = start + self.head_dim;

            // Extract head slices [seq, head_dim]
            let q_h = self.extract_head(q, start, end);
            let k_h = self.extract_head(k, start, end);
            let v_h = self.extract_head(v, start, end);

            // Q @ K.T → [seq, seq] scores in i8
            let scores = self.compute_scores(q_h.view(), k_h.view());

            // Softmax → [seq, seq] probs in u16
            let probs = self.softmax.forward(scores.view());

            // probs @ V → [seq, head_dim] output in i8
            let out_h = self.apply_values(probs.view(), v_h.view());

            // Write head output back
            self.write_head(&mut output, &out_h, start);
        }

        output
    }

    /// Extract a single head's slice from [seq, dim] → [seq, head_dim]
    fn extract_head(&self, x: ArrayView2<i8>, start: usize, _end: usize) -> Array2<i8> {
        let seq = x.nrows();
        let mut head = Array2::<i8>::zeros((seq, self.head_dim));
        for i in 0..seq {
            for j in 0..self.head_dim {
                head[[i, j]] = x[[i, start + j]];
            }
        }
        head
    }

    /// Write head output back to full output tensor
    fn write_head(&self, output: &mut Array2<i8>, head: &Array2<i8>, start: usize) {
        let seq = head.nrows();
        for i in 0..seq {
            for j in 0..self.head_dim {
                output[[i, start + j]] = head[[i, j]];
            }
        }
    }

    /// Compute attention scores: Q @ K.T
    /// Scales to fit i8 output range
    fn compute_scores(&self, q: ArrayView2<i8>, k: ArrayView2<i8>) -> Array2<i8> {
        let seq = q.nrows();
        let mut scores = Array2::<i8>::zeros((seq, seq));

        // Scale factor: brings dot product into i8 range
        // Max dot product: 127 * 127 * head_dim
        // log2(127 * 127) ≈ 14, plus log2(head_dim)/2 for sqrt scaling
        let scale_shift = 7 + (31 - (self.head_dim as u32).leading_zeros()) / 2;

        for i in 0..seq {
            for j in 0..seq {
                let mut sum: i32 = 0;
                for d in 0..self.head_dim {
                    sum += q[[i, d]] as i32 * k[[j, d]] as i32;
                }
                let scaled = sum >> scale_shift;
                scores[[i, j]] = scaled.clamp(-128, 127) as i8;
            }
        }
        scores
    }

    /// Apply attention weights to values: probs @ V
    /// probs: u16 [seq, seq] where 65535 = 1.0
    /// v: i8 [seq, head_dim]
    /// output: i8 [seq, head_dim]
    fn apply_values(&self, probs: ArrayView2<u16>, v: ArrayView2<i8>) -> Array2<i8> {
        let seq = probs.nrows();
        let mut output = Array2::<i8>::zeros((seq, self.head_dim));

        for i in 0..seq {
            for d in 0..self.head_dim {
                let mut sum: i64 = 0;
                for j in 0..seq {
                    sum += probs[[i, j]] as i64 * v[[j, d]] as i64;
                }
                // Divide by 65535 (probs sum to 65535)
                let result = sum / 65535;
                output[[i, d]] = result.clamp(-128, 127) as i8;
            }
        }
        output
    }
}