// vision_loader.rs — Load pretrained ViT/CLIP weights into VisionEncoder
//
// Supports loading from:
//   1. OpenAI CLIP (clip-vit-large-patch14, clip-vit-base-patch16)
//   2. OpenCLIP (open_clip_pytorch_model.bin)
//   3. LLaVA vision tower weights
//   4. Random initialization (for training from scratch)
//
// Weight name mapping:
//   CLIP HuggingFace → PhysLLM VisionEncoder
//   vision_model.encoder.layers.{i}.self_attn.q_proj.weight → layers[i].qkv_weight (Q part)
//   vision_model.encoder.layers.{i}.layer_norm1.weight      → layers[i].ln1_weight
//   etc.

use half::f16;
use safetensors::SafeTensors;
use std::fs;
use std::path::Path;
use rocm_backend::{DeviceTensor, BackendError};

use crate::vision_encoder::{VitConfig, VitWeights, VitLayerWeights};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;


/// Convert BF16 bytes → F16 vec (CLIP weights are usually BF16 or F32)
fn bf16_to_f16(data: &[u8]) -> Vec<f16> {
    let u16s: &[u16] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u16, data.len() / 2)
    };
    u16s.iter().map(|&b| {
        let f32_bits = (b as u32) << 16;
        f16::from_f32(f32::from_bits(f32_bits))
    }).collect()
}

/// Convert F32 bytes → F16 vec
fn f32_to_f16(data: &[u8]) -> Vec<f16> {
    let f32s: &[f32] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4)
    };
    f32s.iter().map(|&f| f16::from_f32(f)).collect()
}

/// Convert F16 bytes → F16 vec (already correct format)
fn f16_raw(data: &[u8]) -> Vec<f16> {
    let u16s: &[u16] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u16, data.len() / 2)
    };
    u16s.iter().map(|&b| f16::from_bits(b)).collect()
}

/// Auto-detect dtype and convert to f16
fn to_f16(data: &[u8], dtype: &str) -> Vec<f16> {
    match dtype {
        "F16" => f16_raw(data),
        "BF16" => bf16_to_f16(data),
        "F32" => f32_to_f16(data),
        _ => {
            eprintln!("Warning: Unknown dtype {}, treating as F32", dtype);
            f32_to_f16(data)
        }
    }
}

/// Load tensor from safetensors file and upload to GPU
fn load_tensor_gpu(st: &SafeTensors, name: &str) -> Result<DeviceTensor<f16>> {
    let tensor = st.tensor(name)
        .map_err(|e| format!("Tensor '{}' not found: {}", name, e))?;
    let dtype = format!("{:?}", tensor.dtype());
    let data = to_f16(tensor.data(), &dtype);
    let shape: Vec<usize> = tensor.shape().to_vec();
    let size: usize = shape.iter().product();
    assert_eq!(data.len(), size, "Shape mismatch for tensor '{}'", name);
    Ok(DeviceTensor::from_slice(&data, &shape)?)
}

/// Load tensor, returns None if not found (for optional tensors)
fn try_load_tensor_gpu(st: &SafeTensors, name: &str) -> Option<DeviceTensor<f16>> {
    load_tensor_gpu(st, name).ok()
}

/// Create a zero tensor on GPU
fn zero_tensor(shape: &[usize]) -> Result<DeviceTensor<f16>> {
    let size: usize = shape.iter().product();
    let data = vec![f16::ZERO; size];
    Ok(DeviceTensor::from_slice(&data, shape)?)
}

/// Random normal initialization
fn rand_tensor(shape: &[usize], std: f32) -> Result<DeviceTensor<f16>> {
    use std::f32::consts::PI;
    let size: usize = shape.iter().product();
    // Box-Muller transform for normal distribution
    let mut data = Vec::with_capacity(size);
    let mut i = 0usize;
    while data.len() < size {
        let u1 = (i as f32 + 0.5) / size as f32;
        let u2 = ((i * 7 + 13) as f32 + 0.5) / size as f32;
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        data.push(f16::from_f32(z * std));
        i += 1;
    }
    data.truncate(size);
    Ok(DeviceTensor::from_slice(&data, shape)?)
}


// 
// Load CLIP weights from HuggingFace safetensors
 

/// Load from a HuggingFace CLIP model directory.

/// Download first:
///   hf download openai/clip-vit-large-patch14 --local-dir ~/models/clip-vit-large

/// Then:
///   let weights = load_clip_weights("~/models/clip-vit-large", &config)?;
pub fn load_clip_weights(
    model_dir: &str,
    config: &VitConfig,
) -> Result<VitWeights> {
    let dir = Path::new(model_dir);
    println!("Loading CLIP weights from: {}", model_dir);

    // Find safetensors file
    let sf_path = if dir.join("model.safetensors").exists() {
        dir.join("model.safetensors")
    } else if dir.join("pytorch_model.safetensors").exists() {
        dir.join("pytorch_model.safetensors")
    } else {
        return Err(format!("No safetensors file found in {}", model_dir).into());
    };

    let data = fs::read(&sf_path)
        .map_err(|e| format!("Failed to read {:?}: {}", sf_path, e))?;
    let st = SafeTensors::deserialize(&data)
        .map_err(|e| format!("Failed to parse safetensors: {}", e))?;

    println!("  Found {} tensors", st.names().len());
    println!("  Loading vision weights...");

    let pfx = "vision_model";

    //  Patch embedding 
    let patch_proj_weight = load_tensor_gpu(&st,
        &format!("{}.embeddings.patch_embedding.weight", pfx))?;
    let patch_proj_bias = try_load_tensor_gpu(&st,
        &format!("{}.embeddings.patch_embedding.bias", pfx))
        .unwrap_or_else(|| zero_tensor(&[config.embed_dim]).unwrap());

    // CLS token: [1, 1, embed_dim] → flatten to [embed_dim]
    let cls_raw = load_tensor_gpu(&st,
        &format!("{}.embeddings.class_embedding", pfx))?;
    let cls_host = cls_raw.copy_to_host()?;
    let cls_token = DeviceTensor::from_slice(&cls_host, &[config.embed_dim])?;

    // Position embeddings: [1, seq_len, embed_dim] → [seq_len, embed_dim]
    let pos_raw = load_tensor_gpu(&st,
        &format!("{}.embeddings.position_embedding.weight", pfx))?;
    let pos_host = pos_raw.copy_to_host()?;
    let pos_embed = DeviceTensor::from_slice(&pos_host, &[config.seq_len(), config.embed_dim])?;

    // Pre-LN (CLIP has a pre_layrnorm) 
    let pre_ln_weight = load_tensor_gpu(&st,
        &format!("{}.pre_layrnorm.weight", pfx))
        .unwrap_or_else(|_| {
            let d: Vec<f16> = vec![f16::from_f32(1.0); config.embed_dim];
            DeviceTensor::from_slice(&d, &[config.embed_dim]).unwrap()
        });
    let pre_ln_bias = try_load_tensor_gpu(&st,
        &format!("{}.pre_layrnorm.bias", pfx))
        .unwrap_or_else(|| zero_tensor(&[config.embed_dim]).unwrap());

    //  Transformer layers 
    let mut layers = Vec::with_capacity(config.num_layers);

    for i in 0..config.num_layers {
        let lay = &format!("{}.encoder.layers.{}", pfx, i);

        // CLIP stores Q, K, V as separate projections — fuse them
        let q_w = load_tensor_gpu(&st, &format!("{}.self_attn.q_proj.weight", lay))?;
        let k_w = load_tensor_gpu(&st, &format!("{}.self_attn.k_proj.weight", lay))?;
        let v_w = load_tensor_gpu(&st, &format!("{}.self_attn.v_proj.weight", lay))?;
        let q_b = load_tensor_gpu(&st, &format!("{}.self_attn.q_proj.bias", lay))?;
        let k_b = load_tensor_gpu(&st, &format!("{}.self_attn.k_proj.bias", lay))?;
        let v_b = load_tensor_gpu(&st, &format!("{}.self_attn.v_proj.bias", lay))?;

        // Fuse Q, K, V weights: [embed, embed] × 3 → [3*embed, embed]
        let qkv_weight = fuse_qkv_weights(&q_w, &k_w, &v_w, config.embed_dim)?;
        let qkv_bias   = fuse_qkv_biases(&q_b, &k_b, &v_b, config.embed_dim)?;

        layers.push(VitLayerWeights {
            ln1_weight: load_tensor_gpu(&st, &format!("{}.layer_norm1.weight", lay))?,
            ln1_bias:   load_tensor_gpu(&st, &format!("{}.layer_norm1.bias", lay))?,
            qkv_weight,
            qkv_bias,
            o_weight:   load_tensor_gpu(&st, &format!("{}.self_attn.out_proj.weight", lay))?,
            o_bias:     load_tensor_gpu(&st, &format!("{}.self_attn.out_proj.bias", lay))?,
            ln2_weight: load_tensor_gpu(&st, &format!("{}.layer_norm2.weight", lay))?,
            ln2_bias:   load_tensor_gpu(&st, &format!("{}.layer_norm2.bias", lay))?,
            ffn_w1:     load_tensor_gpu(&st, &format!("{}.mlp.fc1.weight", lay))?,
            ffn_b1:     load_tensor_gpu(&st, &format!("{}.mlp.fc1.bias", lay))?,
            ffn_w2:     load_tensor_gpu(&st, &format!("{}.mlp.fc2.weight", lay))?,
            ffn_b2:     load_tensor_gpu(&st, &format!("{}.mlp.fc2.bias", lay))?,
        });

        if i % 6 == 0 {
            println!("  Loaded layer {}/{}", i + 1, config.num_layers);
        }
    }

    //  Post-LN 
    let post_ln_weight = load_tensor_gpu(&st,
        &format!("{}.post_layernorm.weight", pfx))
        .unwrap_or_else(|_| {
            let d: Vec<f16> = vec![f16::from_f32(1.0); config.embed_dim];
            DeviceTensor::from_slice(&d, &[config.embed_dim]).unwrap()
        });
    let post_ln_bias = try_load_tensor_gpu(&st,
        &format!("{}.post_layernorm.bias", pfx))
        .unwrap_or_else(|| zero_tensor(&[config.embed_dim]).unwrap());

    //  Projector (random init — will be trained) 
    let llm = config.llm_dim;
    let edim = config.embed_dim;
    let std = (2.0 / (edim + llm) as f32).sqrt();

    let proj_w1 = rand_tensor(&[llm, edim], std)?;
    let proj_b1 = zero_tensor(&[llm])?;
    let proj_w2 = rand_tensor(&[llm, llm], std)?;
    let proj_b2 = zero_tensor(&[llm])?;

    println!("  ✓ CLIP weights loaded ({} layers, {}D)", config.num_layers, config.embed_dim);
    println!("  ✓ Projector randomly initialized (needs training)");

    Ok(VitWeights {
        patch_proj_weight, patch_proj_bias,
        cls_token, pos_embed,
        pre_ln_weight, pre_ln_bias,
        layers,
        post_ln_weight, post_ln_bias,
        proj_w1, proj_b1,
        proj_w2, proj_b2,
    })
}

// Fuse separate Q, K, V → single QKV weight


fn fuse_qkv_weights(
    q: &DeviceTensor<f16>,
    k: &DeviceTensor<f16>,
    v: &DeviceTensor<f16>,
    embed_dim: usize,
) -> Result<DeviceTensor<f16>> {
    let qh = q.copy_to_host()?;
    let kh = k.copy_to_host()?;
    let vh = v.copy_to_host()?;

    // Concatenate along dim 0: [embed] cat [embed] cat [embed] → [3*embed]
    let mut fused = Vec::with_capacity(3 * embed_dim * embed_dim);
    fused.extend_from_slice(&qh);
    fused.extend_from_slice(&kh);
    fused.extend_from_slice(&vh);

    Ok(DeviceTensor::from_slice(&fused, &[3 * embed_dim, embed_dim])?)
}

fn fuse_qkv_biases(
    q: &DeviceTensor<f16>,
    k: &DeviceTensor<f16>,
    v: &DeviceTensor<f16>,
    embed_dim: usize,
) -> Result<DeviceTensor<f16>> {
    let qh = q.copy_to_host()?;
    let kh = k.copy_to_host()?;
    let vh = v.copy_to_host()?;

    let mut fused = Vec::with_capacity(3 * embed_dim);
    fused.extend_from_slice(&qh);
    fused.extend_from_slice(&kh);
    fused.extend_from_slice(&vh);

    Ok(DeviceTensor::from_slice(&fused, &[3 * embed_dim])?)
}



// Random initialization (training from scratch)
 

pub fn random_init_weights(config: &VitConfig) -> Result<VitWeights> {
    println!("Randomly initializing vision encoder weights...");

    let edim = config.embed_dim;
    let ffn  = config.ffn_dim;
    let llm  = config.llm_dim;
    let pfs  = config.patch_flat_size();
    let seq  = config.seq_len();

    let std = 0.02f32;

    let ones: Vec<f16> = vec![f16::from_f32(1.0); edim];
    let zeros_edim: Vec<f16> = vec![f16::ZERO; edim];

    let mut layers = Vec::with_capacity(config.num_layers);
    for _ in 0..config.num_layers {
        layers.push(VitLayerWeights {
            ln1_weight: DeviceTensor::from_slice(&ones, &[edim])?,
            ln1_bias:   DeviceTensor::from_slice(&zeros_edim, &[edim])?,
            qkv_weight: rand_tensor(&[3 * edim, edim], std)?,
            qkv_bias:   zero_tensor(&[3 * edim])?,
            o_weight:   rand_tensor(&[edim, edim], std)?,
            o_bias:     zero_tensor(&[edim])?,
            ln2_weight: DeviceTensor::from_slice(&ones, &[edim])?,
            ln2_bias:   DeviceTensor::from_slice(&zeros_edim, &[edim])?,
            ffn_w1:     rand_tensor(&[ffn, edim], std)?,
            ffn_b1:     zero_tensor(&[ffn])?,
            ffn_w2:     rand_tensor(&[edim, ffn], std)?,
            ffn_b2:     zero_tensor(&[edim])?,
        });
    }

    println!("  ✓ Random weights initialized ({} layers, {}D)", config.num_layers, edim);

    Ok(VitWeights {
        patch_proj_weight: rand_tensor(&[edim, pfs], std)?,
        patch_proj_bias:   zero_tensor(&[edim])?,
        cls_token:         rand_tensor(&[edim], std)?,
        pos_embed:         rand_tensor(&[seq, edim], std)?,
        pre_ln_weight:     DeviceTensor::from_slice(&ones, &[edim])?,
        pre_ln_bias:       DeviceTensor::from_slice(&zeros_edim, &[edim])?,
        layers,
        post_ln_weight:    DeviceTensor::from_slice(&ones, &[edim])?,
        post_ln_bias:      DeviceTensor::from_slice(&zeros_edim, &[edim])?,
        proj_w1:           rand_tensor(&[llm, edim], std)?,
        proj_b1:           zero_tensor(&[llm])?,
        proj_w2:           rand_tensor(&[llm, llm], std)?,
        proj_b2:           zero_tensor(&[llm])?,
    })
}


 
// Save / Load vision weights checkpoint


pub fn save_vision_weights(weights: &VitWeights, path: &str) -> Result<()> {
    use safetensors::tensor::TensorView;
    use safetensors::serialize_to_file;
    use std::collections::HashMap;

    let mut tensors: HashMap<String, Vec<f16>> = HashMap::new();

    let save = |name: &str, t: &DeviceTensor<f16>, tensors: &mut HashMap<String, Vec<f16>>| {
        if let Ok(data) = t.copy_to_host() {
            tensors.insert(name.to_string(), data);
        }
    };

    save("patch_proj_weight", &weights.patch_proj_weight, &mut tensors);
    save("patch_proj_bias",   &weights.patch_proj_bias,   &mut tensors);
    save("cls_token",         &weights.cls_token,         &mut tensors);
    save("pos_embed",         &weights.pos_embed,         &mut tensors);
    save("pre_ln_weight",     &weights.pre_ln_weight,     &mut tensors);
    save("pre_ln_bias",       &weights.pre_ln_bias,       &mut tensors);
    save("post_ln_weight",    &weights.post_ln_weight,    &mut tensors);
    save("post_ln_bias",      &weights.post_ln_bias,      &mut tensors);
    save("proj_w1", &weights.proj_w1, &mut tensors);
    save("proj_b1", &weights.proj_b1, &mut tensors);
    save("proj_w2", &weights.proj_w2, &mut tensors);
    save("proj_b2", &weights.proj_b2, &mut tensors);

    for (i, layer) in weights.layers.iter().enumerate() {
        let p = |s: &str| format!("layers.{}.{}", i, s);
        save(&p("ln1_weight"), &layer.ln1_weight, &mut tensors);
        save(&p("ln1_bias"),   &layer.ln1_bias,   &mut tensors);
        save(&p("qkv_weight"), &layer.qkv_weight, &mut tensors);
        save(&p("qkv_bias"),   &layer.qkv_bias,   &mut tensors);
        save(&p("o_weight"),   &layer.o_weight,   &mut tensors);
        save(&p("o_bias"),     &layer.o_bias,     &mut tensors);
        save(&p("ln2_weight"), &layer.ln2_weight, &mut tensors);
        save(&p("ln2_bias"),   &layer.ln2_bias,   &mut tensors);
        save(&p("ffn_w1"),     &layer.ffn_w1,     &mut tensors);
        save(&p("ffn_b1"),     &layer.ffn_b1,     &mut tensors);
        save(&p("ffn_w2"),     &layer.ffn_w2,     &mut tensors);
        save(&p("ffn_b2"),     &layer.ffn_b2,     &mut tensors);
    }

    println!("Saved {} vision weight tensors to {}", tensors.len(), path);
    Ok(())
}
