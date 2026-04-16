# rage-quant

**Run LLMs 3x faster on your CPU — no GPU required.**

Pure Rust quantized GEMV kernels that perform matrix-vector multiplication directly on GGML quantized data (Q8_0, Q6_K, Q4_K) with AVX2+FMA SIMD. No dequantization. No wasted bandwidth. No GPU needed.

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-pure-orange.svg)]()

---

## Why use rage-quant?

If you are building a Rust LLM inference pipeline and want CPU performance that approaches GPU speed on small-to-medium models, this crate gives you the core computation kernels.

### The problem with standard inference

Every existing Rust framework (candle, burn) does this:

```
GGUF quantized weights -> dequantize to f32 -> f32 GEMV -> result
               4x DRAM bandwidth wasted ^    ^ 3.2 GB RAM for dense cache
```

### What rage-quant does differently

```
GGUF quantized weights -> quantized GEMV -> result
         reads 1.06 bytes/element instead of 4 bytes = 3.76x less DRAM traffic
```

**No dequantization step. No f32 cache. 57% less RAM. 3x faster decode.**

## Real benchmarks (not theoretical)

Tested on **Qwen3-0.6B-Q8_0.gguf** | CPU-only | AMD Ryzen 9 9900X | 12 threads

| What we measured                      | Before       | After       | Improvement    |
|---------------------------------------|-------------|-------------|----------------|
| Decode latency per token              | 42 ms       | 14 ms       | **3.0x faster**|
| From naive Rust implementation        | 120,000 ms  | 466 ms      | **257x faster**|
| From sgemm baseline (standard BLAS)   | 74,758 ms   | 466 ms      | **160x faster**|
| Peak RAM usage                        | 3.2 GB      | 1.38 GB     | **57% less**   |
| Throughput                            | ~24 tok/s   | 67-71 tok/s | **~3x more**   |

These numbers are real, measured, reproducible. See [docs/cpu-optimizations.md](docs/cpu-optimizations.md) for methodology.

### Why is it faster? One insight.

On modern CPUs, LLM decode (batch=1) is **DRAM bandwidth-limited**, not compute-limited. By reading 1 byte (quantized) instead of 4 bytes (f32), you move 3.76x less data through the memory bus. The speedup follows directly.

Additionally: **LLVM cannot auto-vectorize the i8-to-f32 widening path.** It tries i8->i16->i32->f32, wasting registers. Manual `vpmovsxbd` (i8->i32 direct) via `_mm256_cvtepi8_epi32` is required. This is why hand-written AVX2 intrinsics beat the compiler here.

## Quick start

### Install

```toml
[dependencies]
rage-quant = "0.1"
```

### Quantized dot product (the main feature)

```rust
use rage_quant::dot_q8_0_f32;

// quantized_weights: &[u8] -- raw Q8_0 blocks straight from a GGUF file
// input_vector: &[f32] -- your activation vector
// num_elements: total number of f32 elements represented

let result = dot_q8_0_f32(&quantized_weights, &input_vector, num_elements);
// Auto-detects AVX2+FMA at runtime; falls back to scalar on older CPUs
```

### Other quantization formats

```rust
use rage_quant::{dot_q6_k_f32, dot_q4_k_f32};

let result_q6 = dot_q6_k_f32(&q6k_data, &input, num_elements);
let result_q4 = dot_q4_k_f32(&q4k_data, &input, num_elements);
```

### Dequantization (when you need raw f32)

```rust
use rage_quant::{dequantize_q8_0_block, dequantize_q4_k_block, dequantize_q6_k_block};

let f32_values = dequantize_q8_0_block(&block_bytes).unwrap();  // -> 32 f32 values
let f32_q4k = dequantize_q4_k_block(&block_bytes).unwrap();     // -> 256 f32 values
let f32_q6k = dequantize_q6_k_block(&block_bytes).unwrap();     // -> 256 f32 values
```

### Parallel GEMV and GEMM (f32, rayon-accelerated)

```rust
use rage_quant::{gemv_rows_f32, gemm_f32_row_major, dot_f32};

// Matrix-vector multiply (uses all cores via rayon)
let output = gemv_rows_f32(&matrix_data, num_cols, &vector);

// Matrix-matrix multiply (rayon + gemm crate backend)
let output = gemm_f32_row_major(m, n, k, &a, &b);

// Single dot product with AVX2+FMA
let dot = dot_f32(&a, &b);
```

## Supported quantization formats

| Format | Block size | Bits/weight | Function             | SIMD status     |
|--------|-----------|-------------|----------------------|-----------------|
| Q8_0   | 32        | 8           | `dot_q8_0_f32()`     | **AVX2+FMA**    |
| Q6_K   | 256       | 6           | `dot_q6_k_f32()`     | Scalar (AVX2 planned) |
| Q4_K   | 256       | 4           | `dot_q4_k_f32()`     | Scalar (AVX2 planned) |

## How it compares

| Feature                        | rage-quant     | llama.cpp (ggml)  | candle (HuggingFace) | burn       |
|-------------------------------|----------------|-------------------|----------------------|------------|
| Language                       | **Pure Rust**  | C/C++             | Rust                 | Rust       |
| Quantized GEMV (skip deq.)    | **Yes**        | Yes (in C)        | No                   | No         |
| AVX2 SIMD on quantized data   | **Yes**        | Yes (in C)        | Partial              | No         |
| Q8_0 + Q6_K + Q4_K            | **Yes**        | Yes               | Q8_0 only            | No         |
| GGUF-native blocks             | **Yes**        | Yes               | Limited              | No         |
| Standalone reusable crate      | **Yes**        | No (monolithic)   | No (framework)       | No (framework) |
| Zero C/C++ dependency          | **Yes**        | No (is C/C++)     | Has some C deps      | Yes        |
| Measured 3x decode speedup     | **Yes**        | Baseline          | N/A                  | N/A        |

### Why not just use llama.cpp?

llama.cpp is excellent, but:
1. **It is C/C++** -- integrating into a Rust project requires unsafe FFI bindings
2. **It is monolithic** -- you cannot extract just the quantized dot product without pulling the entire engine
3. **rage-quant is a standalone Rust crate** -- `cargo add rage-quant` and you have the kernels

### Why not candle or burn?

Neither implements quantized GEMV on GGUF blocks. They dequantize to f32 first, losing the bandwidth advantage that gives rage-quant its 3x speedup.

## CPU optimization findings (T1-T9)

This crate embodies 9 validated CPU inference optimizations discovered during development. Full details with formulas, measurements, and methodology in [docs/cpu-optimizations.md](docs/cpu-optimizations.md).

| ID | What was optimized                      | Measured result                          |
|----|----------------------------------------|------------------------------------------|
| T1 | GEMV on quantized data (skip f32)      | decode 42ms -> 18ms = **2.3x**          |
| T2 | Eliminate dense f32 weight caches      | RSS 3.2GB -> 1.38GB = **-57% RAM**      |
| T3 | AVX2 widening i8->f32 intrinsics      | **+18.8%** on top of T1                  |
| T4 | Memory-bound diagnosis                 | Proved DRAM is the bottleneck            |
| T5 | In-place residual addition             | Marginal on small models                 |
| T6 | Software prefetch hints                | ~10-20% estimated on 14B+ models         |
| T7 | GEMV vs sgemm for m=1 decode           | sgemm 180ms vs GEMV 18ms = **10x**      |
| T8 | QKV fusion (decode-only path)          | **1.8x** per-layer QKV compute           |
| T9 | Column-tiling for GEMM prefill         | 5091ms -> 3057ms = **1.67x**             |

## Hardware requirements

- **Minimum**: Any x86_64 CPU (scalar fallback works everywhere)
- **Recommended**: AVX2+FMA support (Intel Haswell 2013+ / AMD Zen 2017+)
- **Tested on**: AMD Ryzen 9 9900X (Zen 5), DDR5, 12 threads

ARM NEON and AVX-512 support are planned.

## License

Dual-licensed:

- **Open Source**: [AGPL-3.0](LICENSE) -- free for open-source, personal, and academic use
- **Commercial**: [Commercial License](LICENSE-COMMERCIAL.md) -- for proprietary/closed-source use

For commercial licensing: the@angriestboy.com

## Author

**Carlos Enrique Castro Lazaro**
- GitHub: [@OnCeUponTry](https://github.com/OnCeUponTry)
- Website: [angriestboy.com](https://www.angriestboy.com)

## Contributing

Contributions welcome under AGPL-3.0. By submitting a PR, you agree to license your contribution under the same terms.

**Priority areas:**
- AVX2 SIMD for Q6_K and Q4_K dot products
- ARM NEON implementation
- AVX-512 implementation
- Benchmarks on different hardware (Intel, older AMD, server CPUs)
- Additional quantization formats (Q5_K, Q2_K, IQ formats)
