// src/model/ffn.rs
use ndarray::{Array2, ArrayView2};
use crate::lut::SIGMOID_LUT;

/// Swish activation: x * sigmoid(x)
#[inline]
fn swish(x: i8) -> i8 {
    let idx = (x as i16 + 128) as usize;
    let sig = SIGMOID_LUT[idx] as i16;  // 0-255
    // x * sig / 256, but preserve more precision
    let result = (x as i16 * sig + 128) >> 8;  // +128 for rounding
    result.clamp(-128, 127) as i8
}

/// Pure integer SwiGLU FFN
pub struct FFN {
    w_gate: Array2<i8>,
    w_up: Array2<i8>,
    w_down: Array2<i8>,
}

impl FFN {
    pub fn new(w_gate: Array2<i8>, w_up: Array2<i8>, w_down: Array2<i8>) -> Self {
        Self { w_gate, w_up, w_down }
    }

    pub fn forward(&self, x: ArrayView2<i8>) -> Array2<i8> {
        let gate = self.ternary_matmul_gate(x, self.w_gate.view());
        let up = self.ternary_matmul_gate(x, self.w_up.view());
        let hidden = self.swiglu_combine(gate.view(), up.view());
        self.ternary_matmul_down(hidden.view(), self.w_down.view())
    }

    /// Ternary matmul for gate/up projections (less aggressive scaling)
    fn ternary_matmul_gate(&self, x: ArrayView2<i8>, w: ArrayView2<i8>) -> Array2<i8> {
        let m = x.nrows();
        let k = x.ncols();
        let n = w.ncols();
        let mut out = Array2::<i8>::zeros((m, n));

        // Lighter scaling — just enough to avoid overflow
        // Max sum: 127 * k (all weights +1, all inputs max)
        // For k=4: max = 508, fits in i16, shift by 2
        let scale_shift = (31 - (k as u32).leading_zeros()).saturating_sub(6).max(1);

        for i in 0..m {
            for j in 0..n {
                let mut sum: i32 = 0;
                for p in 0..k {
                    match w[[p, j]] {
                        1 => sum += x[[i, p]] as i32,
                        -1 => sum -= x[[i, p]] as i32,
                        _ => {}
                    }
                }
                let scaled = sum >> scale_shift;
                out[[i, j]] = scaled.clamp(-128, 127) as i8;
            }
        }
        out
    }

    /// Ternary matmul for down projection
    fn ternary_matmul_down(&self, x: ArrayView2<i8>, w: ArrayView2<i8>) -> Array2<i8> {
        let m = x.nrows();
        let k = x.ncols();
        let n = w.ncols();
        let mut out = Array2::<i8>::zeros((m, n));

        let scale_shift = (31 - (k as u32).leading_zeros()).saturating_sub(6).max(1);

        for i in 0..m {
            for j in 0..n {
                let mut sum: i32 = 0;
                for p in 0..k {
                    match w[[p, j]] {
                        1 => sum += x[[i, p]] as i32,
                        -1 => sum -= x[[i, p]] as i32,
                        _ => {}
                    }
                }
                let scaled = sum >> scale_shift;
                out[[i, j]] = scaled.clamp(-128, 127) as i8;
            }
        }
        out
    }

    /// Combine gate and up: swish(gate) ⊙ up
    fn swiglu_combine(&self, gate: ArrayView2<i8>, up: ArrayView2<i8>) -> Array2<i8> {
        let (rows, cols) = gate.dim();
        let mut out = Array2::<i8>::zeros((rows, cols));

        for i in 0..rows {
            for j in 0..cols {
                let swished = swish(gate[[i, j]]) as i16;
                let u = up[[i, j]] as i16;
                // Scale to preserve signal: multiply then shift by 6 (not 7)
                let result = (swished * u + 32) >> 6;  // +32 for rounding
                out[[i, j]] = result.clamp(-128, 127) as i8;
            }
        }
        out
    }
}