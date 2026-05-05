// vision_encoder.rs — Vision Transformer (ViT) in Rust
// Complete vision encoder for PhysLLM multimodal:
//   Image → Patches → ViT → Projection → LLM-compatible image tokens
// Supported configs:
//   ViT-B/16: 12 layers, 768 embed, 12 heads, 16×16 patches → 196 tokens
//   ViT-L/14: 24 layers, 1024 embed, 16 heads, 14×14 patches → 256 tokens
// Usage:
//   let encoder = VisionEncoder::new(&device, VitConfig::base_16());
//   let image_tokens = encoder.encode(&device, raw_pixels, 224, 224)?;
//    image_tokens: [1, 196, llm_hidden_dim] — ready for LLM input

use rocm_backend::{GpuDevice, DeviceTensor, BackendError};
use half::f16;
use std::sync::Arc;

pub type Result<T> = std::result::Result<T, BackendError>;

//  Vision Transformer Config 

#[derive(Debug, Clone)]
pub struct VitConfig {
    pub image_size:    usize,  // Input image size (224 or 336)
    pub patch_size:    usize,  // Patch size (14 or 16)
    pub num_channels:  usize,  // 3 for RGB
    pub embed_dim:     usize,  // ViT hidden dim (768, 1024)
    pub num_layers:    usize,  // Number of transformer blocks (12, 24)
    pub num_heads:     usize,  // Attention heads (12, 16)
    pub ffn_dim:       usize,  // FFN intermediate dim (usually 4 * embed_dim)
    pub llm_dim:       usize,  // Target LLM hidden dim (4096 for 7B)
    pub pool_tokens:   usize,  // Num visual tokens after pooling (64, 128, 256)
    pub layer_norm_eps: f32,
}

impl VitConfig {
    /// ViT-B/16 — 86M params, fast, good quality
    /// 224×224 image → 196 patches of 16×16
    pub fn base_16(llm_dim: usize) -> Self {
        Self {
            image_size: 224, patch_size: 16, num_channels: 3,
            embed_dim: 768, num_layers: 12, num_heads: 12,
            ffn_dim: 3072, llm_dim, pool_tokens: 64,
            layer_norm_eps: 1e-6,
        }
    }

    /// ViT-L/14 — 307M params, high quality
    /// 336×336 image → 576 patches of 14×14
    pub fn large_14(llm_dim: usize) -> Self {
        Self {
            image_size: 336, patch_size: 14, num_channels: 3,
            embed_dim: 1024, num_layers: 24, num_heads: 16,
            ffn_dim: 4096, llm_dim, pool_tokens: 128,
            layer_norm_eps: 1e-6,
        }
    }

    pub fn num_patches(&self) -> usize {
        let h = self.image_size / self.patch_size;
        h * h
    }

    pub fn patch_flat_size(&self) -> usize {
        self.num_channels * self.patch_size * self.patch_size
    }

    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.num_heads
    }

    pub fn seq_len(&self) -> usize {
        self.num_patches() + 1  // +1 for CLS token
    }
}

//Weight Storage 

pub struct VitLayerWeights {
    // Pre-attention layer norm
    pub ln1_weight: DeviceTensor<f16>,
    pub ln1_bias:   DeviceTensor<f16>,

    // QKV projection (fused)
    pub qkv_weight: DeviceTensor<f16>,
    pub qkv_bias:   DeviceTensor<f16>,

    // Output projection
    pub o_weight: DeviceTensor<f16>,
    pub o_bias:   DeviceTensor<f16>,

    // Pre-FFN layer norm
    pub ln2_weight: DeviceTensor<f16>,
    pub ln2_bias:   DeviceTensor<f16>,

    // FFN
    pub ffn_w1: DeviceTensor<f16>,
    pub ffn_b1: DeviceTensor<f16>,
    pub ffn_w2: DeviceTensor<f16>,
    pub ffn_b2: DeviceTensor<f16>,
}

pub struct VitWeights {
    // Patch embedding
    pub patch_proj_weight: DeviceTensor<f16>,  // [embed_dim, patch_flat_size]
    pub patch_proj_bias:   DeviceTensor<f16>,  // [embed_dim]

    // CLS token and position embeddings
    pub cls_token:  DeviceTensor<f16>,          // [embed_dim]
    pub pos_embed:  DeviceTensor<f16>,          // [seq_len, embed_dim]

    // Pre-transformer layer norm
    pub pre_ln_weight: DeviceTensor<f16>,
    pub pre_ln_bias:   DeviceTensor<f16>,

    // Transformer layers
    pub layers: Vec<VitLayerWeights>,

    // Post-transformer layer norm
    pub post_ln_weight: DeviceTensor<f16>,
    pub post_ln_bias:   DeviceTensor<f16>,

    // Vision-Language projector (MLP: vision_dim → llm_dim)
    pub proj_w1: DeviceTensor<f16>,  // [llm_dim, embed_dim]
    pub proj_b1: DeviceTensor<f16>,  // [llm_dim]
    pub proj_w2: DeviceTensor<f16>,  // [llm_dim, llm_dim]
    pub proj_b2: DeviceTensor<f16>,  // [llm_dim]
}

// Vision Encoder 

pub struct VisionEncoder {
    pub config:  VitConfig,
    pub weights: VitWeights,
}

// Extern declarations for GPU kernels
extern "C" {
    fn launch_normalize_image(
        raw_pixels: *const std::ffi::c_void,
        normalized: *mut std::ffi::c_void,
        batch_size: i32, height: i32, width: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_extract_patches(
        image: *const std::ffi::c_void,
        patches: *mut std::ffi::c_void,
        batch_size: i32, channels: i32,
        height: i32, width: i32,
        patch_size_h: i32, patch_size_w: i32,
        num_patches_h: i32, num_patches_w: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_project_patches(
        patches: *const std::ffi::c_void,
        proj_weight: *const std::ffi::c_void,
        proj_bias: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        batch_size: i32, num_patches: i32,
        patch_flat_size: i32, embed_dim: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_add_cls_token(
        patches: *const std::ffi::c_void,
        cls_token: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        batch_size: i32, num_patches: i32, embed_dim: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_add_position_embed(
        tokens: *mut std::ffi::c_void,
        pos_embed: *const std::ffi::c_void,
        batch_size: i32, seq_len: i32, embed_dim: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_vision_qkv_proj(
        input: *const std::ffi::c_void,
        qkv_weight: *const std::ffi::c_void,
        qkv_bias: *const std::ffi::c_void,
        qkv: *mut std::ffi::c_void,
        batch: i32, seq: i32, embed: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_reshape_qkv(
        qkv: *const std::ffi::c_void,
        q: *mut std::ffi::c_void, k: *mut std::ffi::c_void, v: *mut std::ffi::c_void,
        batch: i32, heads: i32, seq: i32, head_dim: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_vision_attention(
        q: *const std::ffi::c_void, k: *const std::ffi::c_void,
        v: *const std::ffi::c_void, output: *mut std::ffi::c_void,
        batch: i32, heads: i32, seq: i32, head_dim: i32, scale: f32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_vision_out_proj(
        attn_out: *const std::ffi::c_void,
        o_weight: *const std::ffi::c_void, o_bias: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        batch: i32, heads: i32, seq: i32, head_dim: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_vision_ffn(
        input: *const std::ffi::c_void,
        w1: *const std::ffi::c_void, b1: *const std::ffi::c_void,
        w2: *const std::ffi::c_void, b2: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        batch: i32, seq: i32, embed: i32, ffn_dim: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_vision_layernorm(
        input: *const std::ffi::c_void,
        weight: *const std::ffi::c_void, bias: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        batch: i32, seq: i32, embed: i32, eps: f32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_projector_layer(
        input: *const std::ffi::c_void,
        weight: *const std::ffi::c_void, bias: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        batch_size: i32, num_tokens: i32, in_dim: i32, out_dim: i32,
        apply_gelu: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_spatial_pool(
        patch_tokens: *const std::ffi::c_void,
        pooled: *mut std::ffi::c_void,
        batch_size: i32, num_patches: i32, pool_size: i32, embed_dim: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_concat_image_text(
        image_tokens: *const std::ffi::c_void,
        text_tokens: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        batch_size: i32, num_img: i32, num_txt: i32, embed_dim: i32,
        stream: *mut std::ffi::c_void,
    );
}

impl VisionEncoder {
    pub fn new(weights: VitWeights, config: VitConfig) -> Self {
        Self { config, weights }
    }

    /// Encode a raw RGB image into LLM-compatible image tokens.
    ///
    /// # Arguments
    /// * `raw_pixels` — Raw uint8 RGB pixels [H*W*3] on CPU
    /// * `height`, `width` — Image dimensions
    ///
    /// # Returns
    /// * GPU tensor [1, pool_tokens, llm_dim] ready to prepend to text tokens
    pub fn encode(
        &self,
        dev: &Arc<GpuDevice>,
        raw_pixels: &[u8],
        height: usize,
        width: usize,
    ) -> Result<DeviceTensor<f16>> {
        let cfg = &self.config;
        let batch = 1;

        //  Step 1: Upload raw pixels to GPU 
        let gpu_pixels = DeviceTensor::<u8>::from_slice(raw_pixels, &[height * width * 3])?;

        //  Step 2: Normalize pixels (ImageNet stats) on GPU 
        let mut normalized = DeviceTensor::<f32>::alloc(&[batch * 3 * height * width])?;
        unsafe {
            launch_normalize_image(
                gpu_pixels.as_ptr() as *const _,
                normalized.as_mut_ptr() as *mut _,
                batch as i32, height as i32, width as i32,
                std::ptr::null_mut(),
            );
        }
        dev.synchronize()?;

        //  Step 3: Extract patches on GPU 
        let ps    = cfg.patch_size;
        let np_h  = height / ps;
        let np_w  = width  / ps;
        let np    = np_h * np_w;
        let pfs   = cfg.patch_flat_size();
        let edim  = cfg.embed_dim;

        let mut patches = DeviceTensor::<f16>::alloc(&[batch * np * pfs])?;
        unsafe {
            launch_extract_patches(
                normalized.as_ptr() as *const _,
                patches.as_mut_ptr() as *mut _,
                batch as i32, cfg.num_channels as i32,
                height as i32, width as i32,
                ps as i32, ps as i32,
                np_h as i32, np_w as i32,
                std::ptr::null_mut(),
            );
        }
        dev.synchronize()?;

        // Step 4: Project patches to embed_dim 
        let mut patch_embeds = DeviceTensor::<f16>::alloc(&[batch * np * edim])?;
        unsafe {
            launch_project_patches(
                patches.as_ptr() as *const _,
                self.weights.patch_proj_weight.as_ptr() as *const _,
                self.weights.patch_proj_bias.as_ptr() as *const _,
                patch_embeds.as_mut_ptr() as *mut _,
                batch as i32, np as i32, pfs as i32, edim as i32,
                std::ptr::null_mut(),
            );
        }
        dev.synchronize()?;

        //  Step 5: Prepend CLS token 
        let seq = cfg.seq_len();  // num_patches + 1
        let mut tokens = DeviceTensor::<f16>::alloc(&[batch * seq * edim])?;
        unsafe {
            launch_add_cls_token(
                patch_embeds.as_ptr() as *const _,
                self.weights.cls_token.as_ptr() as *const _,
                tokens.as_mut_ptr() as *mut _,
                batch as i32, np as i32, edim as i32,
                std::ptr::null_mut(),
            );
        }
        dev.synchronize()?;

        //  Step 6: Add position embeddings 
        unsafe {
            launch_add_position_embed(
                tokens.as_mut_ptr() as *mut _,
                self.weights.pos_embed.as_ptr() as *const _,
                batch as i32, seq as i32, edim as i32,
                std::ptr::null_mut(),
            );
        }
        dev.synchronize()?;

        //  Step 7: ViT Transformer Layers 
        let mut hidden = tokens;
        let scale = 1.0 / (cfg.head_dim() as f32).sqrt();

        for layer in &self.weights.layers {
            // Pre-LN
            let mut normed = DeviceTensor::<f16>::alloc(&[batch * seq * edim])?;
            unsafe {
                launch_vision_layernorm(
                    hidden.as_ptr() as *const _,
                    layer.ln1_weight.as_ptr() as *const _,
                    layer.ln1_bias.as_ptr() as *const _,
                    normed.as_mut_ptr() as *mut _,
                    batch as i32, seq as i32, edim as i32,
                    cfg.layer_norm_eps,
                    std::ptr::null_mut(),
                );
            }
            dev.synchronize()?;

            // QKV projection
            let mut qkv = DeviceTensor::<f16>::alloc(&[batch * seq * 3 * edim])?;
            unsafe {
                launch_vision_qkv_proj(
                    normed.as_ptr() as *const _,
                    layer.qkv_weight.as_ptr() as *const _,
                    layer.qkv_bias.as_ptr() as *const _,
                    qkv.as_mut_ptr() as *mut _,
                    batch as i32, seq as i32, edim as i32,
                    std::ptr::null_mut(),
                );
            }
            dev.synchronize()?;

            // Reshape to multi-head
            let hdim = cfg.head_dim();
            let nhead = cfg.num_heads;
            let mut q = DeviceTensor::<f16>::alloc(&[batch * nhead * seq * hdim])?;
            let mut k = DeviceTensor::<f16>::alloc(&[batch * nhead * seq * hdim])?;
            let mut v = DeviceTensor::<f16>::alloc(&[batch * nhead * seq * hdim])?;
            unsafe {
                launch_reshape_qkv(
                    qkv.as_ptr() as *const _,
                    q.as_mut_ptr() as *mut _, k.as_mut_ptr() as *mut _, v.as_mut_ptr() as *mut _,
                    batch as i32, nhead as i32, seq as i32, hdim as i32,
                    std::ptr::null_mut(),
                );
            }
            dev.synchronize()?;

            // Vision self-attention (BIDIRECTIONAL — no causal mask)
            let mut attn_out = DeviceTensor::<f16>::alloc(&[batch * nhead * seq * hdim])?;
            unsafe {
                launch_vision_attention(
                    q.as_ptr() as *const _, k.as_ptr() as *const _, v.as_ptr() as *const _,
                    attn_out.as_mut_ptr() as *mut _,
                    batch as i32, nhead as i32, seq as i32, hdim as i32, scale,
                    std::ptr::null_mut(),
                );
            }
            dev.synchronize()?;

            // Output projection
            let mut attn_projected = DeviceTensor::<f16>::alloc(&[batch * seq * edim])?;
            unsafe {
                launch_vision_out_proj(
                    attn_out.as_ptr() as *const _,
                    layer.o_weight.as_ptr() as *const _,
                    layer.o_bias.as_ptr() as *const _,
                    attn_projected.as_mut_ptr() as *mut _,
                    batch as i32, nhead as i32, seq as i32, hdim as i32,
                    std::ptr::null_mut(),
                );
            }
            dev.synchronize()?;

            // Residual 1: hidden = hidden + attn_projected
            let hidden_host = hidden.copy_to_host()?;
            let attn_host = attn_projected.copy_to_host()?;
            let residual1: Vec<f16> = hidden_host.iter().zip(attn_host.iter())
                .map(|(&h, &a)| f16::from_f32(h.to_f32() + a.to_f32()))
                .collect();
            hidden = DeviceTensor::from_slice(&residual1, &[batch * seq * edim])?;

            // Pre-FFN LN
            let mut normed2 = DeviceTensor::<f16>::alloc(&[batch * seq * edim])?;
            unsafe {
                launch_vision_layernorm(
                    hidden.as_ptr() as *const _,
                    layer.ln2_weight.as_ptr() as *const _,
                    layer.ln2_bias.as_ptr() as *const _,
                    normed2.as_mut_ptr() as *mut _,
                    batch as i32, seq as i32, edim as i32,
                    cfg.layer_norm_eps,
                    std::ptr::null_mut(),
                );
            }
            dev.synchronize()?;

            // FFN (GELU instead of SwiGLU for ViT)
            let mut ffn_out = DeviceTensor::<f16>::alloc(&[batch * seq * edim])?;
            unsafe {
                launch_vision_ffn(
                    normed2.as_ptr() as *const _,
                    layer.ffn_w1.as_ptr() as *const _, layer.ffn_b1.as_ptr() as *const _,
                    layer.ffn_w2.as_ptr() as *const _, layer.ffn_b2.as_ptr() as *const _,
                    ffn_out.as_mut_ptr() as *mut _,
                    batch as i32, seq as i32, edim as i32, cfg.ffn_dim as i32,
                    std::ptr::null_mut(),
                );
            }
            dev.synchronize()?;

            // Residual 2: hidden = hidden + ffn_out
            let h_host  = hidden.copy_to_host()?;
            let ff_host = ffn_out.copy_to_host()?;
            let residual2: Vec<f16> = h_host.iter().zip(ff_host.iter())
                .map(|(&h, &f)| f16::from_f32(h.to_f32() + f.to_f32()))
                .collect();
            hidden = DeviceTensor::from_slice(&residual2, &[batch * seq * edim])?;
        }

        //  Step 8: Post-transformer LN 
        let mut post_normed = DeviceTensor::<f16>::alloc(&[batch * seq * edim])?;
        unsafe {
            launch_vision_layernorm(
                hidden.as_ptr() as *const _,
                self.weights.post_ln_weight.as_ptr() as *const _,
                self.weights.post_ln_bias.as_ptr() as *const _,
                post_normed.as_mut_ptr() as *mut _,
                batch as i32, seq as i32, edim as i32,
                cfg.layer_norm_eps,
                std::ptr::null_mut(),
            );
        }
        dev.synchronize()?;

        // Step 9: Remove CLS, pool patch tokens 
        // Skip CLS token (index 0), keep only patch tokens
        let patch_host = post_normed.copy_to_host()?;
        let patches_only: Vec<f16> = patch_host[edim..].to_vec(); // Skip CLS
        let patch_gpu = DeviceTensor::from_slice(&patches_only, &[batch * np * edim])?;

        // Spatial pooling: num_patches → pool_tokens
        let pool = cfg.pool_tokens;
        let mut pooled = DeviceTensor::<f16>::alloc(&[batch * pool * edim])?;
        unsafe {
            launch_spatial_pool(
                patch_gpu.as_ptr() as *const _,
                pooled.as_mut_ptr() as *mut _,
                batch as i32, np as i32, pool as i32, edim as i32,
                std::ptr::null_mut(),
            );
        }
        dev.synchronize()?;

        //  Step 10: Project to LLM hidden dim 
        let llm_dim = cfg.llm_dim;

        // Layer 1: embed_dim → llm_dim with GELU
        let mut proj_hidden = DeviceTensor::<f16>::alloc(&[batch * pool * llm_dim])?;
        unsafe {
            launch_projector_layer(
                pooled.as_ptr() as *const _,
                self.weights.proj_w1.as_ptr() as *const _,
                self.weights.proj_b1.as_ptr() as *const _,
                proj_hidden.as_mut_ptr() as *mut _,
                batch as i32, pool as i32, edim as i32, llm_dim as i32,
                1, // apply GELU
                std::ptr::null_mut(),
            );
        }
        dev.synchronize()?;

        // Layer 2: llm_dim → llm_dim (no activation)
        let mut image_tokens = DeviceTensor::<f16>::alloc(&[batch * pool * llm_dim])?;
        unsafe {
            launch_projector_layer(
                proj_hidden.as_ptr() as *const _,
                self.weights.proj_w2.as_ptr() as *const _,
                self.weights.proj_b2.as_ptr() as *const _,
                image_tokens.as_mut_ptr() as *mut _,
                batch as i32, pool as i32, llm_dim as i32, llm_dim as i32,
                0, // no activation
                std::ptr::null_mut(),
            );
        }
        dev.synchronize()?;

        // image_tokens: [1, pool_tokens, llm_dim] — ready for LLM!
        Ok(image_tokens)
    }

    /// Combine image tokens with text tokens for LLM input.
    ///
    /// Returns combined [1, num_img + num_txt, llm_dim] tensor
    /// where image tokens come FIRST (like LLaVA).
    pub fn combine_with_text(
        dev: &Arc<GpuDevice>,
        image_tokens: &DeviceTensor<f16>,
        text_tokens: &DeviceTensor<f16>,
        num_img: usize,
        num_txt: usize,
        llm_dim: usize,
    ) -> Result<DeviceTensor<f16>> {
        let batch = 1;
        let total = num_img + num_txt;
        let mut combined = DeviceTensor::<f16>::alloc(&[batch * total * llm_dim])?;
        unsafe {
            launch_concat_image_text(
                image_tokens.as_ptr() as *const _,
                text_tokens.as_ptr() as *const _,
                combined.as_mut_ptr() as *mut _,
                batch as i32, num_img as i32, num_txt as i32, llm_dim as i32,
                std::ptr::null_mut(),
            );
        }
        dev.synchronize()?;
        Ok(combined)
    }
}
