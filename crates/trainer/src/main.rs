use llm_core::{PhysLLM,ModelConfig,PhysTokenizer, load_from_file};

use rocm_backend:: GpuDevice;
use half::f16; use std::sync::Arc;
use anyhow::Result;
use tracing::info;

struct TrainConfig{
    
    l   learning_rate:    f32,
    warmup_steps:     usize,
    total_steps:      usize,
    batch_size:       usize,
    grad_accum_steps: usize,
    lora_rank:        usize,    // 0 = full fine-tune not sure yet cause error on Rx 9070 xtx 24gb
    lora_alpha:       f32,
    max_seq_len:      usize,
    save_every:       usize,
    eval_every:       usize,
    data_path:        String,
    val_path:         String,
    output_dir:       String,
    model_dir:        String,
}


Struct LoraAdapter{
    rank: usize, alpha: f32, // store A and B directly on Gpu  as Device Tensor <f16>
}
fn train(cfg: TrainConfig) -> Result<()> {

    //check device cannot call another function from diffrent crate to check avaliable vram unstable Rocm driver
    // allocating vram: Open Gpu
    let device = Arc::new(GpuDevice::open_best()?);
    info!("GPU: {} ({})", device.props.name, device.props.gfx_arch); //might show diffrent value on arch ,
    // this whole LLM mainly work on arch linux architecture may cause error on debian based distro
    info!("VRAM: {} MB", device.props.vram_total_mb);

    let model_cfg = ModelConfig::physllm_7b();
    let mut model = PhysLLM::new(model_cfg.Clone(),device.clone())?;
    llm_core::loader::load_weights(&model_cfg, &cfg.model_dir, &device).map(|w| model.weights = w)?;

    let tokenizer = PhysTokenizer::new_simple(32768,1,2);

   let train_data = load_jsonl(&cfg.data_path)?; // open fucking pain in ass data files writing your fucking python script to pull data for traning. 
    let val_data   = load_jsonl(&cfg.val_path)?;
    info!("Train: {} examples  Val: {} examples", train_data.len(), val_data.len());
    // loop for training make sure Vram alteast is 80gb 
    let mut step = 0usize;

    let mut loss_accum = 0.0f32;

    let eff_batch = cfg.batch_size*cfg.grad_accum_steps;

    'outer: for epoch in 0.. {
        for batch_start in ( 0..train_data.len()).step_by(cfg.batch_size){
            if step >= cfg.total_steps { break 'outer;}
            let batch = &train_data[batch_start.. (batch_start + cfg.batch_size).min(train_data.len())];
            // forward Pass + cross-entropy loss
            let loss = forward_and_loss(&model, &tokenizer,batch, &model_cfg)?;
            loss_accum += loss;
        }
    }

}