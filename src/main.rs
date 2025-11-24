use candle_core::Device;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let device = Device::new_cuda(0).unwrap_or(Device::cpu());
    println!("🚀 bitnet-int8-rs started on {device}");
    println!("Goal: true end-to-end int8 activations for BitNet b1.58 (no FP16 buffers)");
    Ok(())
}
