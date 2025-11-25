use ndarray::{Array2, ArrayView2};

/// True BitNet b1.58 Linear Layer
/// - Ternary weights: -1, 0, +1
/// - Int8 activations
/// - Zero multiplications
/// - Zero floats
/// - i32 accumulator (correct & safe)
/// - Power-of-2 scaling via bit shift

pub struct BitLinearInt8 {
    weights: Array2<i8>,     // Shape: [out_features, in_features], values in {-1, 0, 1}
    scale_shift: u8,         // Right-shift amount for final scaling (power-of-2)
}




impl BitLinearInt8 {
    /// Create from raw ternary weights and integer scale
    pub fn new(weights: Array2<i8>, scale_shift: u8) -> Self {
        assert!(weights.iter().all(|&w| w >= -1 && w <= 1 && w != 0 || w == 0));
        Self { weights, scale_shift }
    }

    /// Public getter for weights
    pub fn weights(&self) -> &Array2<i8> {
        &self.weights
    }

    /// Forward pass: true 1.58-bit matmul — NO MULTIPLICATIONS
    pub fn forward(&self, x: ArrayView2<i8>) -> Array2<i8> {
        let (batch_size, in_features) = x.dim();
        let out_features = self.weights.nrows();

        let mut output = Array2::<i8>::zeros((batch_size, out_features));

        for b in 0..batch_size {
            for o in 0..out_features {
                let mut acc: i32 = 0;

                // THE REAL BitNet b1.58 kernel — only add/subtract
                for i in 0..in_features {
                    let w = self.weights[[o, i]];           // -1, 0, or 1
                    let x_val = x[[b, i]] as i32;

                    match w {
                        1 => acc += x_val,
                        -1 => acc -= x_val,
                        0 => {}, // do nothing
                        _ => unreachable!(),
                    }
                }

                // Apply power-of-2 scaling and clamp to int8
                let scaled = acc >> self.scale_shift;
                output[[b, o]] = scaled.clamp(-128, 127) as i8;
            }
        }

        output
    }
}