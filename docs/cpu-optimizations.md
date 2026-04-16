# CPU Inference Optimization Findings (T1-T9)

> Author: Carlos Enrique Castro Lazaro (OnCeUponTry)
> Validated: April 2026
> Hardware: AMD Ryzen 9 9900X, 12 threads, DDR5
> Model: Qwen3-0.6B-Q8_0.gguf (CPU-only inference)

## Overview

These 9 optimizations were discovered and validated during the development of a
pure Rust LLM inference engine. Each finding includes the measured impact, the
technique used, and the domain where it applies.

The cumulative effect: **decode 120,000ms (naive) to 466ms (optimized) = 257x improvement.**

---

## T1 — GEMV on Quantized Data (Skip f32 Materialization)

**Impact:** decode 42ms to 18ms = **2.3x speedup**

**Technique:** Instead of dequantizing weight tensors to f32 before GEMV, perform
the dot product directly on quantized blocks. For Q8_0:

```
Traditional: for each block { dequantize(block) → f32[32] } → dot(f32_weights, f32_input)
Optimized:   for each block { acc += d * sum(q_i * x_i) }
```

**Why it works:** Decode (batch=1) is memory-bandwidth-bound. Q8_0 reads 1.0625 bytes
per element (2 bytes scale + 32 bytes quants = 34 bytes for 32 elements) versus 4 bytes
for f32. This reduces DRAM traffic by 3.76x.

**Code:** `ggml_quant.rs` — `dot_q8_0_f32()`, `dot_q6_k_f32()`, `dot_q4_k_f32()`

---

## T2 — Eliminate Dense f32 Caches

**Impact:** RSS 3.2GB to 1.38GB = **78.8% RAM savings** (57% reduction)

**Technique:** Keep all weight tensors in their quantized GGUF format in memory.
Never allocate a parallel f32 tensor for any weight matrix.

**Why it works:** Dense f32 caches were originally needed because the GEMV code
required f32 inputs. With T1 (quantized GEMV), the dense cache is unnecessary.

---

## T3 — AVX2 Widening i8 to f32 Intrinsics

**Impact:** +18.8% additional speedup on top of T1

**Technique:** Manual AVX2 intrinsics for the i8 to f32 conversion path:

```
_mm_loadl_epi64     → load 8 i8 values
_mm256_cvtepi8_epi32 → sign-extend i8 → i32 (vpmovsxbd)
_mm256_cvtepi32_ps   → convert i32 → f32 (vcvtdq2ps)
_mm256_fmadd_ps      → fused multiply-add
```

**Why manual intrinsics are needed:** LLVM auto-vectorizer cannot figure out the
i8 → f32 widening pattern. It tries to widen to i16 first, then i32, wasting
registers. Direct vpmovsxbd skips the intermediate widening.

**Code:** `ggml_quant.rs` — `dot_q8_0_f32_avx2()`

---

## T4 — Memory-Bound Diagnosis

**Impact:** Conceptual — informed all subsequent optimizations

**Technique:** Analyzed the arithmetic intensity of decode GEMV:

```
Arithmetic intensity = FLOPs / bytes_loaded
For Q8_0 GEMV: 2 * N FLOPs / (1.0625 * N bytes) = 1.88 FLOP/byte
DDR5 bandwidth ~50 GB/s → theoretical max ~94 GFLOP/s for this kernel
Actual FP32 peak ~500 GFLOP/s
```

**Conclusion:** The kernel is 5x below compute peak but close to memory bandwidth
ceiling. Optimizing compute (e.g., more SIMD) yields diminishing returns.
The path forward is reducing bytes loaded (quantization) or improving data reuse
(batching, tiling).

---

## T5 — In-Place Residual Addition

**Impact:** Marginal on 0.6B model, estimated significant on 14B+

**Technique:** Replace `residual = vec_add(residual, layer_output)` with
`add_inplace(&mut residual, &layer_output)` — avoids allocating a new vector
per layer for the residual connection.

**Why it matters at scale:** With 40+ layers (14B model), this eliminates 40
allocations per token of size `hidden_dim * sizeof(f32)`.

---

## T6 — Software Prefetch Hints

**Impact:** Marginal on 0.6B, estimated 10-20% on 14B+ models

**Technique:** Insert `_mm_prefetch` hints N blocks ahead in the GEMV inner loop
to bring weight data into L1/L2 cache before it is needed.

**Why it helps at scale:** Larger models have larger weight matrices. The stride
between accessed blocks exceeds L2 cache line prefetch distance. Explicit prefetch
masks the DRAM latency.

---

## T7 — GEMV Dedicated vs sgemm for Decode (m=1)

**Impact:** sgemm 180ms vs GEMV 18ms = **10x speedup**

**Technique:** Use a dedicated GEMV kernel (matrix-vector) instead of calling
sgemm (matrix-matrix) when the batch size is 1 (autoregressive decode).

**Why it works:** sgemm is optimized for large matrix-matrix products with tiling,
packing, and cache blocking. For m=1 (single vector), all that overhead is wasted.
A simple row-parallel GEMV with rayon is 10x faster because it has zero setup cost
and accesses memory sequentially.

**Code:** `gemm_kernel.rs` — `gemv_par()`

---

## T8 — QKV Fusion (Decode-Only)

**Impact:** 1.8x speedup on QKV projection per layer

**Technique:** Fuse Q, K, V projections into a single GEMV call when the weight
matrices are stored contiguously:

```
Traditional: q = gemv(W_q, x); k = gemv(W_k, x); v = gemv(W_v, x);  // 3 calls
Fused:       qkv = gemv(W_qkv, x);  // 1 call, W_qkv is [W_q | W_k | W_v]
```

**Status:** Validated then reverted — requires weight layout reorganization during
model loading. Will be re-implemented with proper weight packing.

---

## T9 — Column-Tiling for GEMM Prefill

**Impact:** 5091ms to 3057ms = **1.67x speedup**

**Technique:** During prefill (processing the entire prompt), the operation is
matrix-matrix (GEMM). Tile the computation along the column dimension to improve
L2 cache utilization:

```
Instead of: C = A * B  (full columns at once)
Do:         for tile in column_tiles { C[:, tile] = A * B[:, tile] }
```

**Why it works:** By processing a subset of columns at a time, the working set of
matrix B fits in L2 cache, reducing DRAM traffic during the multiply-accumulate.

---

## Summary Table

| ID | Optimization                 | Measured                   | Domain       | Status          |
|----|------------------------------|---------------------------|--------------|-----------------|
| T1 | Quantized GEMV (no deq.)     | 42ms → 18ms = 2.3x       | Decode GEMV  | validated-CPU   |
| T2 | Eliminate dense caches        | 3.2GB → 1.38GB = -57%    | Memory       | validated-CPU   |
| T3 | AVX2 i8→f32 intrinsics       | +18.8% over T1            | SIMD         | validated-CPU   |
| T4 | Memory-bound diagnosis        | Proved DRAM bottleneck    | Analysis     | validated       |
| T5 | In-place residuals            | Marginal on 0.6B          | Memory       | implemented     |
| T6 | Software prefetch             | ~10-20% est. on 14B+     | Cache        | implemented     |
| T7 | GEMV vs sgemm for m=1        | 180ms → 18ms = 10x       | Decode       | validated-CPU   |
| T8 | QKV fusion                   | 1.8x per-layer QKV       | Attention    | validated-reverted |
| T9 | Column-tiling GEMM           | 5091ms → 3057ms = 1.67x  | Prefill      | validated-CPU   |

## Reproduce

All measurements taken with:
- Model: Qwen3-0.6B-Q8_0.gguf (GGUF format, 8-bit quantization)
- CPU: AMD Ryzen 9 9900X (Zen 5, 12 cores)
- RAM: DDR5 (dual channel)
- OS: Debian Linux (kernel 6.x)
- Threads: 12 (rayon thread pool)
- Measurement: wall-clock time per decode step, averaged over 500 tokens
