mod model;  
use model::bitlinear_int8::BitLinearInt8;
use model::rmsnorm::RMSNorm;
use ndarray::{Array1, Array2};


fn main() {
    println!("BitNet b1.58 — True Zero-Multiply Kernel");

    let weights = Array2::from_shape_vec(
        (4, 4),
        vec![
            -1, 0, 1, -1,
             1, -1, 0,  1,
             0,  1, -1, 0,
             1,  0, -1, 1,
        ],
    ).unwrap();

    let layer = BitLinearInt8::new(weights, 0); // no shift = scale 1.0

    let x = Array2::from_shape_vec((1, 4), vec![127, 127, 127, 127]).unwrap();

    let out = layer.forward(x.view());

    println!("Weights:\n{}", layer.weights());
    println!("Input (int8):\n{}", x);
    println!("Output (int8):\n{}", out);

    println!("\nTrue zero-float RMSNorm test:");

    // Pre-quantized weight (1.0, 1.0, 1.0, 1.0) in Q0.16 → 65536
    let weight_q16 = Array1::from_vec(vec![65536i32; 4]);

    // eps = 1e-5 → Q32 = 429
    let norm = RMSNorm::from_quantized(weight_q16, 429);

    let input = Array2::from_shape_vec((1, 4), vec![-127i8, -50i8, 0i8, 127i8]).unwrap();
    let output = norm.forward(input.view());

    println!("Input:  {:?}", input);
    println!("RMSNorm output: {:?}", output);
}