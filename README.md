---
title: RAGE-QUANT
emoji: ⚡
colorFrom: orange
colorTo: blue
sdk: rust
app_file: Cargo.toml
pytorch: false
tensorflow: false
---

# RAGE-QUANT

**Run LLMs 3x faster on your CPU — no GPU required.**

Pure Rust quantized GEMV kernels for GGUF inference. Skip dequantization, save 57% RAM, get 3x faster decode.

## Features

- **Pure Rust** — zero C/C++ dependencies, cargo add rage-quant
- **AVX2+FMA SIMD** — hand-written intrinsics for i8→f32 widening (LLVM can't auto-vectorize this)
- **Q8_0, Q6_K, Q4_K** — GGUF-native block formats
- **No dequantization** — reads 1.06 bytes/element instead of 4 bytes = 3.76x less DRAM traffic
- **57% less RAM** — no dense f32 weight cache

## Benchmarks

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Decode latency | 42ms | 14ms | **3.0x faster** |
| Naive Rust | 120,000ms | 466ms | **257x faster** |
| sgemm BLAS | 74,758ms | 466ms | **160x faster** |
| Peak RAM | 3.2GB | 1.38GB | **57% less** |
| Throughput | ~24 tok/s | 67-71 tok/s | **~3x more** |

Tested on: Qwen3-0.6B-Q8_0.gguf | AMD Ryzen 9 9900X | 12 threads | CPU-only

## Usage

```toml
[dependencies]
rage-quant = "0.1"
```

```rust
use rage_quant::dot_q8_0_f32;

let result = dot_q8_0_f32(&quantized_weights, &input_vector, num_elements);
```

## License

- **AGPL-3.0** — open source / personal / academic use
- **Commercial** — proprietary use (contact: the@angriestboy.com)

## Author

Carlos Enrique Castro Lazaro — [@OnCeUponTry](https://github.com/OnCeUponTry) | [angriestboy.com](https://www.angriestboy.com)