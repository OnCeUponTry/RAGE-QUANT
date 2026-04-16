use rayon::prelude::*;

pub fn dot_f32(lhs: &[f32], rhs: &[f32]) -> f32 {
    assert_eq!(lhs.len(), rhs.len(), "dot_f32 requiere slices del mismo largo");

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
            // SAFETY: se verifica soporte AVX2/FMA antes de invocar el kernel especializado.
            return unsafe { dot_f32_avx2(lhs, rhs) };
        }
    }

    dot_f32_scalar(lhs, rhs)
}

pub fn gemv_rows_f32(rows: &[f32], row_len: usize, vec: &[f32]) -> Vec<f32> {
    assert_eq!(vec.len(), row_len, "gemv_rows_f32 requiere vector del mismo ancho que la fila");
    assert_eq!(rows.len() % row_len, 0, "gemv_rows_f32 requiere rows divisible por row_len");

    rows.par_chunks_exact(row_len)
        .map(|row| dot_f32(row, vec))
        .collect()
}

fn dot_f32_scalar(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn dot_f32_avx2(lhs: &[f32], rhs: &[f32]) -> f32 {
    use std::arch::x86_64::{
        __m256, _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_loadu_ps,
        _mm256_setzero_ps, _mm_add_ps, _mm_cvtss_f32, _mm_hadd_ps,
    };

    let len = lhs.len();
    let mut index = 0usize;
    let mut acc: __m256 = _mm256_setzero_ps();
    while index + 8 <= len {
        let a = _mm256_loadu_ps(lhs.as_ptr().add(index));
        let b = _mm256_loadu_ps(rhs.as_ptr().add(index));
        acc = _mm256_fmadd_ps(a, b, acc);
        index += 8;
    }

    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo, hi);
    let sum64 = _mm_hadd_ps(sum128, sum128);
    let sum32 = _mm_hadd_ps(sum64, sum64);
    let mut total = _mm_cvtss_f32(sum32);

    while index < len {
        total += lhs[index] * rhs[index];
        index += 1;
    }
    total
}

pub fn gemm_f32_row_major(m: usize, n: usize, k: usize, lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
    if lhs.len() != m * k {
        panic!("lhs len {} != m*k {}", lhs.len(), m * k);
    }
    if rhs.len() != k * n {
        panic!("rhs len {} != k*n {}", rhs.len(), k * n);
    }

    let mut out = vec![0.0f32; m * n];
    // SAFETY: todos los punteros apuntan a buffers contiguos de tamano valido
    // en layout row-major y no se solapan entre si.
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            out.as_mut_ptr(),
            1,
            n as isize,
            false,
            lhs.as_ptr(),
            1,
            k as isize,
            rhs.as_ptr(),
            1,
            n as isize,
            0.0f32,
            1.0f32,
            false,
            false,
            false,
            gemm::Parallelism::Rayon(0),
        );
    }
    out
}
