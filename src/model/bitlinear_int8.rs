use candle_core::{Tensor, Result, Device, DType};

/// This will become the first true end-to-end int8 BitLinear on GPU
pub struct BitLinearInt8 {
    weight_packed: Tensor,    // u32-packed ternary weights
    scale_w: Tensor,          // group-wise scales
    scale_act_ema: Tensor,    // running EMA scale for activations
    alpha: f32,
}

impl BitLinearInt8 {
    pub fn new(weight_packed: Tensor, scale_w: Tensor, hidden_size: usize, device: &Device) -> Result<Self> {
        let scale_act_ema = Tensor::zeros((hidden_size,), DType::F32, device)?;
        Ok(Self { weight_packed, scale_w, scale_act_ema, alpha: 0.01 })
    }

    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        // Week 2–6: this becomes the fused Triton int8×ternary kernel call
        todo!("true int8×ternary → int8 kernel goes here")
    }
}
