//! # rage-quant
//!
//! High-performance quantized GEMV kernels for CPU-only LLM inference.
//!
//! This crate provides direct dot product operations on GGML quantized
//! tensor blocks (Q8_0, Q6_K, Q4_K) without requiring dequantization
//! to dense f32 tensors first. This approach:
//!
//! - Reduces DRAM bandwidth by 3.76x (1.06 bytes/elem vs 4 bytes/elem)
//! - Eliminates dense f32 cache entirely (78.8% RAM savings)
//! - Achieves 3.0x decode speedup on CPU inference
//!
//! ## Key functions
//!
//! - [`dot_q8_0_f32`] — Direct dot product on Q8_0 blocks (auto-detects AVX2)
//! - [`dot_q6_k_f32`] — Direct dot product on Q6_K blocks
//! - [`dot_q4_k_f32`] — Direct dot product on Q4_K blocks
//! - [`dequantize_q8_0_block`] — Dequantize Q8_0 block to f32
//! - [`dequantize_q4_k_block`] — Dequantize Q4_K block to f32
//! - [`dequantize_q6_k_block`] — Dequantize Q6_K block to f32
//!
//! ## GEMM/GEMV utilities
//!
//! - [`gemm_kernel::dot_f32`] — AVX2+FMA vectorized f32 dot product
//! - [`gemm_kernel::gemv_par`] — Rayon-parallelized GEMV
//! - [`gemm_kernel::gemm_par`] — Rayon-parallelized GEMM
//!
//! ## Example
//!
//! ```ignore
//! use rage_quant::{dot_q8_0_f32, dequantize_q8_0_block};
//!
//! // Direct quantized dot product (no dequantization needed)
//! let result = dot_q8_0_f32(&quantized_data, &input_vector, num_elements);
//!
//! // Or dequantize a single block if needed
//! let f32_values = dequantize_q8_0_block(&block_bytes).unwrap();
//! ```

pub mod ggml_quant;
pub mod gemm_kernel;

// Re-export primary quantized dot product functions
pub use ggml_quant::{
    dot_q8_0_f32,
    dot_q6_k_f32,
    dot_q4_k_f32,
    dequantize_q8_0_block,
    dequantize_q4_k_block,
    dequantize_q6_k_block,
    decode_f16,
};

// Re-export GEMM/GEMV utilities
pub use gemm_kernel::{dot_f32, gemv_rows_f32, gemm_f32_row_major};
