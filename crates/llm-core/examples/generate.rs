use rocm_backend::{GpuDevice, DeviceTensor};
use std::sync::Arc;
use half::f16;
use tokenizers::Tokenizer;

fn bf16_to_f16_vec(bf16_data: &[u8]) -> Vec<f16> {
    let bf16_slice: &[u16] = unsafe {
        std::slice::from_raw_parts(bf16_data.as_ptr() as *const u16, bf16_data.len() / 2)
    };
    bf16_slice.iter().map(|&bf16| {
        f16::from_f32(f32::from_bits((bf16 as u32) << 16))
    }).collect()
}

fn main() {
    println!("Mistral 7B - Text Generation Demo\n");
    
    let dev = Arc::new(GpuDevice::open_best().unwrap());
    
    // Load tokenizer
    println!("Loading tokenizer...");
    let tokenizer = Tokenizer::from_file("/root/models/mistral-7b/tokenizer.json")
        .expect("Failed to load tokenizer");
    println!("✓ Tokenizer loaded\n");
    
    // Load model (embeddings + lm_head only for now)
    println!("Loading model weights...");
    use safetensors::SafeTensors;
    use std::fs;
    
    let buf1 = fs::read("/root/models/mistral-7b/model-00001-of-00002.safetensors").unwrap();
    let buf2 = fs::read("/root/models/mistral-7b/model-00002-of-00002.safetensors").unwrap();
    let shard1 = SafeTensors::deserialize(&buf1).unwrap();
    let shard2 = SafeTensors::deserialize(&buf2).unwrap();
    
    let get_tensor = |name: &str| -> Vec<f16> {
        shard1.tensor(name).or_else(|_| shard2.tensor(name))
            .map(|t| bf16_to_f16_vec(t.data()))
            .expect(&format!("Missing: {}", name))
    };
    
    let embed_weights = get_tensor("model.embed_tokens.weight");
    let lm_head_weights = get_tensor("lm_head.weight");
    
    println!("✓ Weights loaded\n");
    
    // Test prompt
    let prompt = "The fundamental laws of physics";
    println!("Prompt: \"{}\"\n", prompt);
    
    // Tokenize
    let encoding = tokenizer.encode(prompt, false).unwrap();
    let input_ids = encoding.get_ids();
    println!("Tokens: {:?}", input_ids);
    println!("Token count: {}\n", input_ids.len());
    
    // For MVP: just show what WOULD happen
    println!("Generation process (simplified):");
    println!("  1. Embed tokens → [batch=1, seq={}, hidden=4096]", input_ids.len());
    println!("  2. Run through 32 transformer layers");
    println!("  3. Apply LM head → logits [vocab_size=32000]");
    println!("  4. Sample next token from logits");
    println!("  5. Decode token → text");
    println!("  6. Repeat for N tokens\n");
    
    // Simulate output (for demo purposes)
    let mock_output_ids = vec![526, 847, 369, 3309, 28725]; // " are fundamental to"
    let decoded = tokenizer.decode(&mock_output_ids, false).unwrap();
    
    println!("Demo output (simulated):");
    println!("  \"{}{}\"", prompt, decoded);
    println!("\n(This is mock data - real generation requires implementing forward pass)");
    
    println!("\n✓ Tokenizer working!");
    println!("✓ Model loaded!");
    println!("✓ Next: Implement forward() in model.rs");
}
