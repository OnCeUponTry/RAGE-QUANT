# rage-quant

**High-performance quantized GEMV kernels for CPU-only LLM inference in pure Rust.**

Direct dot product on GGML quantized tensor blocks (Q8_0, Q6_K, Q4_K) with AVX2+FMA SIMD acceleration — achieving **3.0x decode speedup** without GPU.

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-pure-orange.svg)]()

---

## What is this?

`rage-quant` provides optimized kernels that perform matrix-vector multiplication (GEMV) **directly on quantized data** — without dequantizing to dense f32 tensors first.

### The Problem

Traditional LLM inference on CPU follows this path:
```
quantized weights (Q8_0) → dequantize to f32 → f32 GEMV → result
```
This wastes:
- **4x DRAM bandwidth** (reading 4 bytes/element instead of ~1 byte)
- **RAM for dense cache** (full f32 copy of every weight tensor)
- **CPU cycles** on the dequantization step itself

### The Solution

rage-quant skips dequantization entirely:
```
quantized weights (Q8_0) → quantized GEMV → result
```
- Reads **1.06 bytes/element** instead of 4 bytes (3.76x bandwidth savings)
- **Zero dense f32 cache** — weights stay quantized in memory
- AVX2+FMA SIMD processes 8 quantized values per cycle

## Benchmarks

Validated on **Qwen3-0.6B-Q8_0.gguf** | CPU-only | Ryzen 9 9900X | 12 threads (rayon)

| Metric                           | Before    | After     | Improvement |
|----------------------------------|-----------|-----------|-------------|
| Decode per token                 | 42 ms     | 14 ms     | **3.0x**    |
| From naive implementation        | 120,000ms | 466 ms    | **257x**    |
| From sgemm baseline              | 74,758 ms | 466 ms    | **160x**    |
| From stable P5 baseline          | 788 ms    | 466 ms    | **40.9%**   |
| Peak RSS (dense cache eliminated)| 3.2 GB    | 1.38 GB   | **-57%**    |
| Throughput (decode)              | ~24 tok/s | 67-71 tok/s| **~3x**    |

### Key insight: memory-bound, not compute-bound

On modern CPUs, LLM inference decode (batch=1) is **DRAM bandwidth-limited**, not compute-limited. By reading quantized data directly (1 byte vs 4 bytes per element), we reduce the bottleneck by 3.76x — which translates directly to the 3.0x speedup observed.

## Supported Formats

| Format | Block size | Bits/weight | Dot product function | SIMD |
|--------|-----------|------------|---------------------|------|
| Q8_0   | 32        | 8          | `dot_q8_0_f32()`    | AVX2+FMA |
| Q6_K   | 256       | 6          | `dot_q6_k_f32()`    | Scalar (AVX2 planned) |
| Q4_K   | 256       | 4          | `dot_q4_k_f32()`    | Scalar (AVX2 planned) |

## Installation

Add to your `Cargo.toml`:
```toml
[dependencies]
rage-quant = "0.1"
```

Or clone and build:
```bash
git clone https://github.com/OnCeUponTry/rage-quant.git
cd rage-quant
cargo build --release
cargo test
```

## Usage

### Direct quantized dot product (the main feature)

```rust
use rage_quant::dot_q8_0_f32;

// quantized_weights: &[u8] — raw Q8_0 blocks from GGUF file
// input_vector: &[f32] — your activation vector
// num_elements: number of f32 elements represented

let result = dot_q8_0_f32(&quantized_weights, &input_vector, num_elements);
// Automatically uses AVX2+FMA if available, falls back to scalar
```

### Other quantization formats

```rust
use rage_quant::{dot_q6_k_f32, dot_q4_k_f32};

let result_q6 = dot_q6_k_f32(&q6k_data, &input_vector, num_elements);
let result_q4 = dot_q4_k_f32(&q4k_data, &input_vector, num_elements);
```

### Dequantization (if you need f32 output)

```rust
use rage_quant::{dequantize_q8_0_block, dequantize_q4_k_block, dequantize_q6_k_block};

let f32_values = dequantize_q8_0_block(&block_bytes)?;  // 32 f32 values
let f32_q4k = dequantize_q4_k_block(&block_bytes)?;     // 256 f32 values
let f32_q6k = dequantize_q6_k_block(&block_bytes)?;     // 256 f32 values
```

### Parallel GEMV/GEMM

```rust
use rage_quant::{gemv_par, gemm_par, dot_f32};

// Parallel matrix-vector multiply (rayon)
let output = gemv_par(&matrix, &vector, rows, cols);

// Parallel matrix-matrix multiply
let output = gemm_par(&a, &b, m, n, k);

// Single dot product with AVX2
let dot = dot_f32(&a, &b);
```

## How it works

### Q8_0 quantized dot product

Each Q8_0 block contains:
- 2 bytes: f16 scale factor `d`
- 32 bytes: 32 signed 8-bit quantized values

Traditional approach: `dequantize(q) = d * q_i` for each element, then dot product on f32.

rage-quant approach: accumulate `d * sum(q_i * x_i)` per block, avoiding f32 materialization.

### AVX2+FMA implementation

The AVX2 kernel processes 32 quantized values per block in 4 groups of 8:

```
For each group of 8 values:
  1. _mm_loadl_epi64     — load 8 bytes (i8 quantized values)
  2. _mm256_cvtepi8_epi32 — sign-extend i8 → i32 (vpmovsxbd)
  3. _mm256_cvtepi32_ps   — convert i32 → f32 (vcvtdq2ps)
  4. _mm256_mul_ps         — multiply by scale factor d
  5. _mm256_fmadd_ps       — fused multiply-add with input vector
```

This pattern is critical because **LLVM cannot auto-vectorize the i8→f32 widening path** — manual intrinsics are required.

Final horizontal reduction uses the standard SSE `movehdup + movehl + add_ss` pattern.

## Comparison with existing tools

| Feature                        | rage-quant | llama.cpp (ggml) | candle (HF) | burn   |
|-------------------------------|------------|-------------------|-------------|--------|
| Language                       | Pure Rust  | C/C++             | Rust        | Rust   |
| Quantized GEMV (no deq.)      | **Yes**    | Yes (C)           | No          | No     |
| AVX2 SIMD for quant dot       | **Yes**    | Yes (C)           | Partial     | No     |
| Q8_0/Q6_K/Q4_K support        | **Yes**    | Yes               | Q8_0 only   | No     |
| GGUF native                   | **Yes**    | Yes               | Limited     | No     |
| Standalone crate               | **Yes**    | No (monolith)     | Framework   | Framework |
| Zero C/C++ dependencies        | **Yes**    | No (is C++)       | Has some    | Yes    |
| Measured decode speedup        | **3.0x**   | Baseline          | N/A         | N/A    |

### Why not just use llama.cpp?

llama.cpp is excellent but:
1. It is C/C++ — if you are building a Rust inference pipeline, you need FFI bindings
2. It is monolithic — you cannot use just the quantized GEMV without the entire engine
3. rage-quant is a **standalone Rust crate** you can `cargo add` to any project

## CPU Optimization Findings (T1-T9)

This crate embodies 9 validated CPU inference optimizations. See [docs/cpu-optimizations.md](docs/cpu-optimizations.md) for the complete findings with formulas and evidence.

| ID | Optimization | Measured Result |
|----|-------------|-----------------|
| T1 | GEMV on quantized data (skip f32) | decode 42ms → 18ms = 2.3x |
| T2 | Eliminate dense f32 caches | RSS 3.2GB → 1.38GB = -57% |
| T3 | AVX2 widening i8→f32 intrinsics | +18.8% on top of T1 |
| T4 | Memory-bound diagnosis | Proved DRAM bottleneck |
| T5 | In-place residual addition | Marginal on 0.6B |
| T6 | Software prefetch hints | ~10-20% on 14B+ (estimated) |
| T7 | GEMV vs sgemm for m=1 decode | sgemm 180ms vs GEMV 18ms = 10x |
| T8 | QKV fusion (decode-only) | 1.8x per-layer QKV |
| T9 | Column-tiling for GEMM prefill | 5091ms → 3057ms = 1.67x |

## Hardware Requirements

- **Minimum**: Any x86_64 CPU (scalar fallback)
- **Recommended**: CPU with AVX2+FMA (Intel Haswell+ / AMD Zen+)
- **Tested on**: AMD Ryzen 9 9900X (Zen 5), 12 threads

ARM/NEON support is planned but not yet implemented.

## License

This project is dual-licensed:

- **Open Source**: [AGPL-3.0](LICENSE) — free for open-source projects and personal/academic use
- **Commercial**: [Commercial License](LICENSE-COMMERCIAL.md) — for proprietary/closed-source use

For commercial licensing inquiries, contact: the@angriestboy.com

## Author

**Carlos Enrique Castro Lazaro**
- GitHub: [@OnCeUponTry](https://github.com/OnCeUponTry)
- Website: [angriestboy.com](https://www.angriestboy.com)

## Contributing

Contributions are welcome under the terms of the AGPL-3.0 license.
By submitting a pull request, you agree to license your contribution under AGPL-3.0.

Priority areas for contribution:
- AVX2 SIMD for Q6_K and Q4_K dot products
- ARM NEON implementation
- AVX-512 implementation
- Benchmarks on different hardware
- Additional quantization format support (Q5_K, Q2_K, IQ formats)
