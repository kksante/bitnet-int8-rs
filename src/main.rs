use bitnet_int8_rs::model::{
    attention::Attention,
    bitlinear_int8::BitLinearInt8,
    ffn::FFN,
    rmsnorm::RMSNorm,
    softmax::IntSoftmax,
};
use ndarray::{array, Array1, Array2};


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

    // Test softmax
    println!("\nTrue zero-float softmax test:");
    let scores = array![[-10i8, 0, 5, 5]];
    let softmax = IntSoftmax::unscaled();
    let probs = softmax.forward(scores.view());
    println!("Softmax input:  {:?}", scores);
    println!("Softmax output: {:?}", probs);
    println!("Sum: {}", probs.row(0).iter().map(|&x| x as u64).sum::<u64>());
    // Expected: most weight on the two 5s, little on 0, almost none on -10

    // Test attention
    println!("\nAttention test:");
    let n_heads = 2;
    let head_dim = 4;
    let _seq = 3;
    let _dim = n_heads * head_dim; // 8

    // Q, K, V: [seq, dim]
    let q = array![
        [64i8, 64, 0, 0,   0, 0, 64, 64],
        [0, 0, 64, 64,     64, 64, 0, 0],
        [32, 32, 32, 32,   32, 32, 32, 32]
    ];
    let k = array![
        [64i8, 64, 0, 0,   0, 0, 64, 64],
        [0, 0, 64, 64,     64, 64, 0, 0],
        [32, 32, 32, 32,   32, 32, 32, 32]
    ];
    let v = array![
        [100i8, 0, 0, 0,   0, 100, 0, 0],
        [0, 100, 0, 0,     0, 0, 100, 0],
        [0, 0, 100, 0,     0, 0, 0, 100]
    ];

    let attn = Attention::new(n_heads, head_dim);
    let output = attn.forward(q.view(), k.view(), v.view());
    println!("Q:\n{:?}", q);
    println!("K:\n{:?}", k);
    println!("V:\n{:?}", v);
    println!("Attention output:\n{:?}", output);

    // Test FFN (SwiGLU)
    println!("\nFFN (SwiGLU) test:");
    let _dim = 4;
    let _hidden_dim = 8;

    // Ternary weights
    let w_gate = array![
        [ 1i8,  0, -1,  1,  0, -1,  1,  0],
        [-1,  1,  0, -1,  1,  0, -1,  1],
        [ 0, -1,  1,  0, -1,  1,  0, -1],
        [ 1,  1, -1, -1,  0,  0,  1,  1]
    ];
    let w_up = array![
        [ 1i8,  1,  1,  1, -1, -1, -1, -1],
        [ 0,  1,  0,  1,  0,  1,  0,  1],
        [-1,  0,  1,  0, -1,  0,  1,  0],
        [ 1, -1,  1, -1,  1, -1,  1, -1]
    ];
    let w_down = array![
        [ 1i8, -1,  0,  1],
        [ 0,  1, -1,  0],
        [-1,  0,  1, -1],
        [ 1,  1,  0,  0],
        [ 0, -1,  1,  1],
        [-1,  1,  0, -1],
        [ 1,  0, -1,  1],
        [ 0,  0,  1,  0]
    ];

    let ffn = FFN::new(w_gate, w_up, w_down);

    let x = array![
        [64i8, 32, -32, -64],
        [100, 50, 0, -50]
    ];
    let output = ffn.forward(x.view());
    println!("Input:\n{:?}", x);
    println!("FFN output:\n{:?}", output);
}