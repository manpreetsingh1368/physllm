// vision_inference.rs — Multimodal inference pipeline

// Connects the vision encoder to your existing LLM.

// Flow:
//   1. Decode image (JPEG/PNG → raw pixels)
//   2. Resize to model's expected size (224×224 or 336×336)
//   3. Vision encoder → image tokens [1, 64, llm_dim]
//   4. Tokenize text prompt
//   5. Combine: [image_tokens] + [text_tokens]
//   6. LLM forward pass → text output

// Usage (from your server):
//   let pipeline = MultimodalPipeline::new(device, vit_config, llm_config)?;
//   let response = pipeline.generate_from_image(
//       &image_bytes, "Describe this image in detail"
//   )?;

use std::sync::Arc;
use half::f16;
use rocm_backend::{GpuDevice, DeviceTensor, BackendError};
use crate::vision_encoder::{VisionEncoder, VitConfig};
use crate::vision_loader::{load_clip_weights, random_init_weights};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

//  Image Preprocessing 

/// Decode JPEG/PNG bytes to raw RGB pixels
/// Returns (pixels: Vec<u8>, width: usize, height: usize)
pub fn decode_image(image_bytes: &[u8]) -> Result<(Vec<u8>, usize, usize)> {
    // Use the `image` crate for decoding
    use image::io::Reader as ImageReader;
    use image::GenericImageView;
    use std::io::Cursor;

    let img = ImageReader::new(Cursor::new(image_bytes))
        .with_guessed_format()?
        .decode()?;

    let (w, h) = img.dimensions();
    let rgb = img.to_rgb8();
    let pixels = rgb.into_raw(); // Vec<u8> in RGB order, [H*W*3]

    Ok((pixels, w as usize, h as usize))
}

/// Resize image to target size using bilinear interpolation (CPU)
pub fn resize_image(
    pixels: &[u8],
    src_w: usize, src_h: usize,
    dst_w: usize, dst_h: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; dst_h * dst_w * 3];

    let scale_x = src_w as f32 / dst_w as f32;
    let scale_y = src_h as f32 / dst_h as f32;

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let sx = (dx as f32 + 0.5) * scale_x - 0.5;
            let sy = (dy as f32 + 0.5) * scale_y - 0.5;

            let x0 = (sx.floor() as i32).clamp(0, src_w as i32 - 1) as usize;
            let y0 = (sy.floor() as i32).clamp(0, src_h as i32 - 1) as usize;
            let x1 = (x0 + 1).min(src_w - 1);
            let y1 = (y0 + 1).min(src_h - 1);

            let wx = sx - sx.floor();
            let wy = sy - sy.floor();

            for c in 0..3 {
                let p00 = pixels[(y0 * src_w + x0) * 3 + c] as f32;
                let p10 = pixels[(y0 * src_w + x1) * 3 + c] as f32;
                let p01 = pixels[(y1 * src_w + x0) * 3 + c] as f32;
                let p11 = pixels[(y1 * src_w + x1) * 3 + c] as f32;

                let val = p00 * (1.0 - wx) * (1.0 - wy)
                        + p10 * wx * (1.0 - wy)
                        + p01 * (1.0 - wx) * wy
                        + p11 * wx * wy;

                out[(dy * dst_w + dx) * 3 + c] = val.round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    out
}

//  Multimodal Pipeline 

pub struct MultimodalPipeline {
    pub device:       Arc<GpuDevice>,
    pub vit_config:   VitConfig,
    pub vision_enc:   VisionEncoder,
    // Add your LLM inference engine here:
    // pub llm:       InferenceEngine,
}

impl MultimodalPipeline {
    /// Initialize pipeline with CLIP weights.

    /// # Arguments
    /// * `clip_model_dir` — Path to CLIP safetensors (e.g., ~/models/clip-vit-large-patch14)
    /// * `llm_hidden_dim` — Your LLM's hidden dimension (4096 for 7B)
    pub fn new(
        device: Arc<GpuDevice>,
        clip_model_dir: Option<&str>,
        llm_hidden_dim: usize,
    ) -> Result<Self> {
        let vit_config = VitConfig::large_14(llm_hidden_dim);

        let weights = if let Some(dir) = clip_model_dir {
            println!("Loading CLIP vision encoder from: {}", dir);
            load_clip_weights(dir, &vit_config)?
        } else {
            println!("No CLIP weights provided — using random init (for training)");
            random_init_weights(&vit_config)?
        };

        let vision_enc = VisionEncoder::new(weights, vit_config.clone());

        Ok(Self { device, vit_config, vision_enc })
    }

    /// Encode an image to GPU tokens ready for the LLM.
  
    /// # Returns
    /// * [1, pool_tokens, llm_dim] tensor on GPU
    pub fn encode_image(&self, image_bytes: &[u8]) -> Result<DeviceTensor<f16>> {
        let (pixels, w, h) = decode_image(image_bytes)?;
        let target = self.vit_config.image_size;

        // Resize to model's expected input size
        let resized = if w != target || h != target {
            println!("  Resizing image {}×{} → {}×{}", w, h, target, target);
            resize_image(&pixels, w, h, target, target)
        } else {
            pixels
        };

        println!("  Running vision encoder ({} layers, {}D)...",
            self.vit_config.num_layers, self.vit_config.embed_dim);

        let image_tokens = self.vision_enc.encode(
            &self.device,
            &resized,
            target, target,
        ).map_err(|e| format!("Vision encoder error: {:?}", e))?;

        println!("  ✓ Image encoded → [{} visual tokens × {}D]",
            self.vit_config.pool_tokens, self.vit_config.llm_dim);

        Ok(image_tokens)
    }

    /// Build the multimodal prompt for the LLM.
    /// Image tokens go first, then the text prompt.

    /// Returns a description string for use with vLLM or your Rust LLM.
    pub fn build_image_prompt(&self, text_prompt: &str) -> String {
        // Special image token placeholder (matches LLaVA convention)
        let img_placeholder = "<image>".repeat(self.vit_config.pool_tokens);
        format!("{}\n{}", img_placeholder, text_prompt)
    }

    /// Full pipeline: image + text → LLM-ready combined tokens

    /// In your inference loop:
    ///   1. Call this to get image tokens
    ///   2. Tokenize text prompt
    ///   3. Call VisionEncoder::combine_with_text()
    ///   4. Pass combined tokens to your LLM
    pub fn prepare_multimodal_input(
        &self,
        image_bytes: &[u8],
        text_prompt: &str,
        text_token_ids: &[i64],
    ) -> Result<(DeviceTensor<f16>, String)> {
        // Encode image
        let image_tokens = self.encode_image(image_bytes)?;

        // Build the combined prompt
        let full_prompt = self.build_image_prompt(text_prompt);

        // Image tokens are ready to be prepended to text embeddings
        // The LLM's embedding layer needs to convert text_token_ids to embeddings,
        // then concat: [image_tokens | text_embeddings]
        println!("  Multimodal input ready:");
        println!("    Image tokens: {} × {}D", self.vit_config.pool_tokens, self.vit_config.llm_dim);
        println!("    Text tokens:  {}", text_token_ids.len());
        println!("    Total seq:    {}", self.vit_config.pool_tokens + text_token_ids.len());

        Ok((image_tokens, full_prompt))
    }
}

//  HTTP API for Image + Text Chat 
// Add this endpoint to your FastAPI backend to accept image uploads

pub const IMAGE_API_EXAMPLE: &str = r#"
# Add to your FastAPI backend (app/chat/routes.py):

from fastapi import File, UploadFile, Form
import base64

@router.post("/vision")
async def vision_chat(
    image: UploadFile = File(...),
    message: str = Form(...),
    session_id: Optional[str] = Form(None),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Read image bytes
    image_bytes = await image.read()
    image_b64 = base64.b64encode(image_bytes).decode()

    # Send to vLLM with image (if using multimodal model)
    # OR send to your Rust vision encoder + LLM pipeline
    payload = {
        "model": "/models/gpt-oss-20b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                    },
                    {
                        "type": "text",
                        "text": message
                    }
                ]
            }
        ],
        "max_tokens": 2000,
    }

    # Stream response back to user
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(VLLM_URL, json=payload)
        return resp.json()
"#;
