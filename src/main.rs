mod model;  
use model::bitlinear_int8::BitLinearInt8;
use ndarray::Array2;

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
}