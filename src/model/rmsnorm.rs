// src/model/rmsnorm.rs
use ndarray::{Array1, Array2, ArrayView2};

// Fixed-point format constants
const Q12_SHIFT: i32 = 12;
const Q16_SHIFT: i32 = 16;
const Q24_SHIFT: i32 = 24;
const Q32_TO_Q24_SHIFT: i32 = 8;

// Output scale: 64 represents 1.0 (leaves headroom for values up to ±2)
const OUTPUT_SCALE: i64 = 64;

/// Pure integer RMSNorm — zero floating-point operations
///
/// Normalizes to unit variance without changing scale.
/// Output uses 64 = 1.0 convention, giving headroom for |values| up to ~2.
///
/// Fixed-point formats:
/// - Weights: Q16.16 (1.0 = 65536)
/// - Epsilon: Q32 (1e-5 ≈ 429)
/// - Intermediate: Q24/Q12 for precision during normalization
pub struct RMSNorm {
    weight: Array1<i32>,
    eps: i64,
}

impl RMSNorm {
    /// Create from pre-quantized weights and epsilon
    pub fn from_quantized(weight_q16: Array1<i32>, eps_q32: i64) -> Self {
        Self {
            weight: weight_q16,
            eps: eps_q32,
        }
    }

    /// Forward pass using pure integer arithmetic
    pub fn forward(&self, x: ArrayView2<i8>) -> Array2<i8> {
        let (batch, dim) = x.dim();
        let mut out = Array2::<i8>::zeros((batch, dim));

        for b in 0..batch {
            let row = x.row(b);

            // Sum of squares (i64 accumulator)
            let sum_sq: i64 = row.iter().map(|&v| (v as i64).pow(2)).sum();

            // Mean squared in Q24, plus epsilon (Q32 → Q24)
            let mean_sq_q24 = (sum_sq << Q24_SHIFT) / dim as i64;
            let rms_sq_q24 = mean_sq_q24 + (self.eps >> Q32_TO_Q24_SHIFT);

            // Integer square root: Q24 → Q12
            let rms_q12 = isqrt(rms_sq_q24 as u64);
            if rms_q12 == 0 {
                continue;
            }

            // Normalize and apply weights
            for i in 0..dim {
                let x_i = row[i] as i64;

                // Normalize: Q24 / Q12 = Q12
                let norm_q12 = (x_i << Q24_SHIFT) / rms_q12 as i64;

                // Apply weight: Q12 × Q16 = Q28 → Q16
                let weighted_q16 = (norm_q12 * self.weight[i] as i64) >> Q12_SHIFT;

                // Scale to output range (64 = 1.0)
                let result = (weighted_q16 * OUTPUT_SCALE) >> Q16_SHIFT;

                out[[b, i]] = result.clamp(-128, 127) as i8;
            }
        }
        out
    }
}

/// Integer square root via Newton-Raphson
/// Input: Q24, Output: Q12
#[inline]
fn isqrt(n: u64) -> u64 {
    if n < 2 {
        return n;
    }

    let mut x = n;
    let mut y = (x + 1) / 2;

    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}