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

            // backward gradient 
            //manual gradient computation & autograde crate

            backward_pass_stub(loss, learning_rate_schedule(step,&cfg));

            if(step + 1)% cfg.grad_accum_steps == 0 {
                apply_gradients_stub(&mut model, learning_rate_schedule(step, &cfg));
                let avg_loss = loss_accum / cfg.grad_accum_steps as f32;
                info!("step={step} loss={avg_loss:.4} lr={:.2e}", learning_rate_schedule(step,&cfg));
                loss_accum = 0.0;

            }
            if step % cfg.eval_every == 0{
                let val_loss = evaluate(&model,&tokenizer,&val_data,&model_cfg)?;
                info!("Eval step={step} val_loss={val_loss:.4} ppl={:.2}" , val_loss.exp());
            }
            if step % cfg.save_every == 0{
                save_checkpoint(&model, &cfg.output_dir, step)?;
                info!("Checkpoint saved: {}/step_{step}", cfg.output_dir);
            }
            step +=1;
        }
    }
    save_checkpoint(&model,&cfg.output_dir, step)?;
    info!("Traning completed. Finial checkpoint at step {step}.");
    Ok(())

}

fn learning_rate_schedule(step:usize, cfg: &TrainConfig) -> f32{
    // linear and cosine(h) decay
    if step < cfg.warmup_steps{
        cfg.learning_rate * step as f32 / cfg.warmup_steps as f32

    } else {
        let progress = (step - cfg.warmup_steps) as f32 / (cfg.total_steps - cfg.warmup_steps) as f32;
        cfg.learning_rate * (0.5 *(1.0 +(std::f32::consts::PI * progress.cos())))
    }
}

fn forward_and_loss(
    model: &PhysLLMm,
    tokenizer: &PhysTokenizer,
    batch: &[serde_json::value],
    cfg:&ModelConfig,
) -> Result<f32> {
    let mut total_loss = 0.0f32;
    let mut total_tokens = 0usize;

    for example in batch{
        // extracting messages and format to string
        let text = format_messages(example);
        let tokens = tokenizer.encode(&text)?;

        if tokens.len() < 2 {continue;}
        let input = &tokens[..tokens.len() -1];
        let target = &tokens[1..];

        //forward psss ->>>> logits[ seg\q, vocab]
        let logits = model.forward(input,0)?;

        // cross entropy loss

     let loss = cross_entropy(&logits,target,cfg.vocab_size);
     total_loss += loss;
     total_tokens =+ target.len();
    }
    Ok(if total_tokens > 0 {total_loss / total_tokens as f32} else{ 0.0})
}

fn cross_entropy(logits: &[f32], targets: &[u32], vocab_size: usize) -> f32 {
    let mut loss = 0.0f32;
    for (i, &target) in targets.iter().enumerate() {
        let offset = i * vocab_size;
        let row    = &logits[offset..offset + vocab_size];
        let max    = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let log_sum_exp = row.iter().map(|&x| (x - max).exp()).sum::<f32>().ln() + max;
        loss += log_sum_exp - row[target as usize];
    }
    loss
}
