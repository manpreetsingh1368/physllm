use metal::*;
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MetalError {
    #[error("No Metal device available")]
    NoDevice,
    #[error("Failed to compile Metal Shading Language source: {0}")]
    CompileError(String),
    #[error("Function '{0}' not found in Metal library")]
    FunctionNotFound(String),
    #[error("Failed to create compute pipeline state: {0}")]
    PipelineCreationError(String),
    #[error("Buffer allocation failed")]
    BufferAllocationFailed,
}

pub struct MetalBackend {
    pub device: Device,
    pub command_queue: CommandQueue,
    pub is_unified: bool,
}

impl MetalBackend {
    pub fn new() -> Result<Self, MetalError> {
        let device = Device::system_default().ok_or(MetalError::NoDevice)?;
        let command_queue = device.new_command_queue();
        let is_unified = device.has_unified_memory();

        log::info!("Initialized Metal Device: {}", device.name());
        log::info!("Unified Memory Architecture: {}", is_unified);

        Ok(Self {
            device,
            command_queue,
            is_unified,
        })
    }

    pub fn alloc_buffer<T>(&self, count: usize) -> Result<Buffer, MetalError> {
        let size = (count * std::mem::size_of::<T>()) as u64;
        let options = if self.is_unified {
            MTLResourceOptions::StorageModeShared
        } else {
            MTLResourceOptions::StorageModeShared
        };

        Ok(self.device.new_buffer(size, options))
    }

    pub fn alloc_buffer_with_data<T>(&self, data: &[T]) -> Result<Buffer, MetalError> {
        let size = (data.len() * std::mem::size_of::<T>()) as u64;
        let options = MTLResourceOptions::StorageModeShared;

        Ok(self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            size,
            options,
        ))
    }

    pub fn compile_pipeline(
        &self,
        msl_source: &str,
        kernel_name: &str,
    ) -> Result<ComputePipelineState, MetalError> {
        let compile_options = CompileOptions::new();
        let library = self
            .device
            .new_library_with_source(msl_source, &compile_options)
            .map_err(|e| MetalError::CompileError(e.to_string()))?;

        let function = library
            .get_function(kernel_name, None)
            .map_err(|_| MetalError::FunctionNotFound(kernel_name.to_string()))?;

        self.device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| MetalError::PipelineCreationError(e.to_string()))
    }

    pub fn dispatch_1d(
        &self,
        pipeline: &ComputePipelineState,
        buffers: &[&Buffer],
        total_threads: usize,
    ) -> Result<(), MetalError> {
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline);
        for (idx, buf) in buffers.iter().enumerate() {
            encoder.set_buffer(idx as u32, Some(*buf), 0);
        }

        let threads_per_group = pipeline.max_total_threads_per_threadgroup().min(256) as u64;
        let thread_groups = ((total_threads as u64) + threads_per_group - 1) / threads_per_group;

        encoder.dispatch_thread_groups(
            MTLSize {
                width: thread_groups,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_group,
                height: 1,
                depth: 1,
            },
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }
}

pub const PHYSICS_KERNELS_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void vector_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    c[id] = a[id] + b[id];
}

kernel void rms_norm(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    uint row_idx = id * dim;
    float sum_sq = 0.0f;
    for (uint i = 0; i < dim; i++) {
        float val = input[row_idx + i];
        sum_sq += val * val;
    }
    float rms = rsqrt(sum_sq / float(dim) + eps);
    for (uint i = 0; i < dim; i++) {
        output[row_idx + i] = input[row_idx + i] * rms * weight[i];
    }
}

kernel void silu_activation(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    float x = input[id];
    output[id] = x / (1.0f + exp(-x));
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_init() {
        if let Ok(backend) = MetalBackend::new() {
            assert!(!backend.device.name().is_empty());
        }
    }
}
