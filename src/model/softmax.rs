// src/model/softmax.rs
use ndarray::{Array2, ArrayView2};
use crate::model::exp_lut::EXP_LUT_Q4;


const OUTPUT_MAX: u64 = 65535; // u16 max, represents probability 1.0

/// Pure integer softmax with configurable scaling
///
/// Improvements over basic version:
/// - Q4 fractional LUT (16x resolution)
/// - Configurable scale_shift for sqrt(d_k) handling
/// - u16 output for better probability resolution
/// - Exact sum correction
pub struct IntSoftmax {
    /// Right-shift applied to input differences
    /// For attention: sqrt(d_k) as power of 2
    /// d_k=64 → scale_shift=3 (divide by 8)
    scale_shift: u32,
}

#[allow(dead_code)]
impl IntSoftmax {
    /// Create with automatic scaling for given head dimension
    /// Computes scale_shift ≈ log2(sqrt(head_dim)) using pure integer math
    pub fn new(head_dim: usize) -> Self {
        // log2(sqrt(x)) = log2(x) / 2
        // log2(x) = 63 - leading_zeros(x) for non-zero x
        let scale_shift = if head_dim <= 1 {
            0
        } else {
            let log2_head = 63 - (head_dim as u64).leading_zeros();
            log2_head / 2
        };
        Self { scale_shift }
    }

    /// Create with explicit scale shift
    pub fn with_scale_shift(scale_shift: u32) -> Self {
        Self { scale_shift }
    }

    /// Create with no scaling (scale_shift = 0)
    pub fn unscaled() -> Self {
        Self { scale_shift: 0 }
    }

    /// Forward pass
    /// Input: i8 attention scores
    /// Output: u16 probabilities where 65535 ≈ 1.0
    pub fn forward(&self, x: ArrayView2<i8>) -> Array2<u16> {
        let (batch, seq) = x.dim();
        let mut out = Array2::<u16>::zeros((batch, seq));

        for b in 0..batch {
            let row = x.row(b);

            // Find max for numerical stability
            let max_val = row.iter().copied().max().unwrap_or(0) as i32;

            // Compute exp values via Q4 LUT
            let mut exp_vals: Vec<u32> = Vec::with_capacity(seq);
            let mut sum: u64 = 0;

            for &val in row.iter() {
                // Q4 scaled difference: (val - max) * 16 >> scale_shift
                let diff_q4 = ((val as i32 - max_val) << 4) >> self.scale_shift;
                let idx = (diff_q4 + 255).clamp(0, 255) as usize;

                let exp_val = EXP_LUT_Q4[idx];
                exp_vals.push(exp_val);
                sum += exp_val as u64;
            }

            // Handle zero sum (all inputs very negative)
            if sum == 0 {
                let uniform = (OUTPUT_MAX / seq as u64) as u16;
                for i in 0..seq {
                    out[[b, i]] = uniform;
                }
                continue;
            }

            // Normalize to u16 probabilities
            let mut prob_sum: u64 = 0;
            let mut max_idx: usize = 0;
            let mut max_exp: u32 = 0;

            for i in 0..seq {
                let prob = (exp_vals[i] as u64 * OUTPUT_MAX / sum) as u16;
                out[[b, i]] = prob;
                prob_sum += prob as u64;

                if exp_vals[i] > max_exp {
                    max_exp = exp_vals[i];
                    max_idx = i;
                }
            }

            // Exact sum correction: adjust largest element
            if prob_sum != OUTPUT_MAX {
                let correction = OUTPUT_MAX as i64 - prob_sum as i64;
                let adjusted = out[[b, max_idx]] as i64 + correction;
                out[[b, max_idx]] = adjusted.clamp(0, OUTPUT_MAX as i64) as u16;
            }
        }
        out
    }
}