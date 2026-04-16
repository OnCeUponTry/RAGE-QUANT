use anyhow::Result;
use half::f16;

const QK8_0: usize = 32;
const QK_K: usize = 256;
const K_SCALE_SIZE: usize = 12;

pub fn decode_f16(bytes: [u8; 2]) -> f32 {
    f16::from_bits(u16::from_le_bytes(bytes)).to_f32()
}

pub fn dequantize_q8_0_block(block: &[u8]) -> Result<Vec<f32>> {
    if block.len() != 2 + QK8_0 {
        anyhow::bail!("Bloque Q8_0 invalido: {} bytes", block.len());
    }
    let d = decode_f16([block[0], block[1]]);
    let mut out = Vec::with_capacity(QK8_0);
    for quant in &block[2..] {
        out.push(d * (*quant as i8) as f32);
    }
    Ok(out)
}

fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

pub fn dequantize_q4_k_block(block: &[u8]) -> Result<Vec<f32>> {
    let expected = 2 * 2 + K_SCALE_SIZE + QK_K / 2;
    if block.len() != expected {
        anyhow::bail!("Bloque Q4_K invalido: {} bytes", block.len());
    }

    let d = decode_f16([block[0], block[1]]);
    let dmin = decode_f16([block[2], block[3]]);
    let scales = &block[4..4 + K_SCALE_SIZE];
    let quants = &block[4 + K_SCALE_SIZE..];

    let mut out = Vec::with_capacity(QK_K);
    let mut is = 0usize;
    let mut q_offset = 0usize;

    for _ in (0..QK_K).step_by(64) {
        let (sc1, m1) = get_scale_min_k4(is, scales);
        let d1 = d * f32::from(sc1);
        let min1 = dmin * f32::from(m1);
        let (sc2, m2) = get_scale_min_k4(is + 1, scales);
        let d2 = d * f32::from(sc2);
        let min2 = dmin * f32::from(m2);

        for l in 0..32 {
            out.push(d1 * f32::from(quants[q_offset + l] & 0x0F) - min1);
        }
        for l in 0..32 {
            out.push(d2 * f32::from(quants[q_offset + l] >> 4) - min2);
        }

        q_offset += 32;
        is += 2;
    }

    Ok(out)
}

pub fn dequantize_q6_k_block(block: &[u8]) -> Result<Vec<f32>> {
    let expected = 2 + QK_K / 16 + (3 * QK_K) / 4;
    if block.len() != expected {
        anyhow::bail!("Bloque Q6_K invalido: {} bytes", block.len());
    }

    let ql_len = QK_K / 2;
    let qh_len = QK_K / 4;
    let scales_len = QK_K / 16;

    let ql = &block[..ql_len];
    let qh = &block[ql_len..ql_len + qh_len];
    let scales = &block[ql_len + qh_len..ql_len + qh_len + scales_len];
    let d = decode_f16([
        block[ql_len + qh_len + scales_len],
        block[ql_len + qh_len + scales_len + 1],
    ]);

    let mut out = vec![0.0f32; QK_K];
    let mut ql_offset = 0usize;
    let mut qh_offset = 0usize;
    let mut scales_offset = 0usize;
    let mut y_offset = 0usize;

    for _ in (0..QK_K).step_by(128) {
        for l in 0..32 {
            let is = l / 16;
            let qh_byte = qh[qh_offset + l];
            let q1 = (((ql[ql_offset + l] & 0x0F) | (((qh_byte >> 0) & 3) << 4)) as i8) - 32;
            let q2 = (((ql[ql_offset + 32 + l] & 0x0F) | (((qh_byte >> 2) & 3) << 4)) as i8) - 32;
            let q3 = (((ql[ql_offset + l] >> 4) | (((qh_byte >> 4) & 3) << 4)) as i8) - 32;
            let q4 = (((ql[ql_offset + 32 + l] >> 4) | (((qh_byte >> 6) & 3) << 4)) as i8) - 32;

            out[y_offset + l] = d * f32::from(scales[scales_offset + is] as i8) * f32::from(q1);
            out[y_offset + 32 + l] = d * f32::from(scales[scales_offset + 2 + is] as i8) * f32::from(q2);
            out[y_offset + 64 + l] = d * f32::from(scales[scales_offset + 4 + is] as i8) * f32::from(q3);
            out[y_offset + 96 + l] = d * f32::from(scales[scales_offset + 6 + is] as i8) * f32::from(q4);
        }
        ql_offset += 64;
        qh_offset += 32;
        scales_offset += 8;
        y_offset += 128;
    }

    Ok(out)
}

/// Dot product directo sobre bloques Q8_0 contra un vector f32.
/// Evita dequantizar todo el tensor: procesa bloque a bloque acumulando.
pub fn dot_q8_0_f32(qdata: &[u8], vec_f32: &[f32], num_elements: usize) -> f32 {
    let block_size = QK8_0;
    let type_size = 2 + QK8_0;
    let num_blocks = num_elements / block_size;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: verificamos AVX2+FMA; los slices cuadran por la precondicion de bloques Q8_0
            return unsafe { dot_q8_0_f32_avx2(qdata, vec_f32, num_blocks, type_size, block_size) };
        }
    }

    dot_q8_0_f32_scalar(qdata, vec_f32, num_blocks, type_size, block_size)
}

fn dot_q8_0_f32_scalar(qdata: &[u8], vec_f32: &[f32], num_blocks: usize, type_size: usize, block_size: usize) -> f32 {
    let mut acc = 0.0f32;
    for block_idx in 0..num_blocks {
        let block_start = block_idx * type_size;
        let block = &qdata[block_start..block_start + type_size];
        let d = decode_f16([block[0], block[1]]);
        let quants = &block[2..];
        let vec_offset = block_idx * block_size;

        let mut block_acc = 0.0f32;
        for i in 0..block_size {
            block_acc += (quants[i] as i8) as f32 * vec_f32[vec_offset + i];
        }
        acc += d * block_acc;
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q8_0_f32_avx2(
    qdata: &[u8],
    vec_f32: &[f32],
    num_blocks: usize,
    type_size: usize,
    block_size: usize,
) -> f32 {
    use std::arch::x86_64::*;

    let mut acc = _mm256_setzero_ps();

    for block_idx in 0..num_blocks {
        let block_start = block_idx * type_size;
        let block = &qdata[block_start..block_start + type_size];
        let d = decode_f16([block[0], block[1]]);
        let quants = block.as_ptr().add(2);
        let vec_offset = block_idx * block_size;
        let vec_ptr = vec_f32.as_ptr().add(vec_offset);

        let d_vec = _mm256_set1_ps(d);

        // Procesar 32 i8 valores en 4 grupos de 8
        // Grupo 0: quants[0..8]
        let q_bytes = _mm_loadl_epi64(quants as *const __m128i);
        let q_i32 = _mm256_cvtepi8_epi32(q_bytes);
        let q_f32 = _mm256_cvtepi32_ps(q_i32);
        let x_f32 = _mm256_loadu_ps(vec_ptr);
        acc = _mm256_fmadd_ps(_mm256_mul_ps(d_vec, q_f32), x_f32, acc);

        // Grupo 1: quants[8..16]
        let q_bytes = _mm_loadl_epi64(quants.add(8) as *const __m128i);
        let q_i32 = _mm256_cvtepi8_epi32(q_bytes);
        let q_f32 = _mm256_cvtepi32_ps(q_i32);
        let x_f32 = _mm256_loadu_ps(vec_ptr.add(8));
        acc = _mm256_fmadd_ps(_mm256_mul_ps(d_vec, q_f32), x_f32, acc);

        // Grupo 2: quants[16..24]
        let q_bytes = _mm_loadl_epi64(quants.add(16) as *const __m128i);
        let q_i32 = _mm256_cvtepi8_epi32(q_bytes);
        let q_f32 = _mm256_cvtepi32_ps(q_i32);
        let x_f32 = _mm256_loadu_ps(vec_ptr.add(16));
        acc = _mm256_fmadd_ps(_mm256_mul_ps(d_vec, q_f32), x_f32, acc);

        // Grupo 3: quants[24..32]
        let q_bytes = _mm_loadl_epi64(quants.add(24) as *const __m128i);
        let q_i32 = _mm256_cvtepi8_epi32(q_bytes);
        let q_f32 = _mm256_cvtepi32_ps(q_i32);
        let x_f32 = _mm256_loadu_ps(vec_ptr.add(24));
        acc = _mm256_fmadd_ps(_mm256_mul_ps(d_vec, q_f32), x_f32, acc);
    }

    // Reduccion horizontal del acumulador AVX2
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(result)
}

/// Dot product directo sobre bloques Q6_K contra un vector f32.
pub fn dot_q6_k_f32(qdata: &[u8], vec_f32: &[f32], num_elements: usize) -> f32 {
    let ql_per_block = QK_K / 2;
    let qh_per_block = QK_K / 4;
    let scales_per_block = QK_K / 16;
    let type_size = ql_per_block + qh_per_block + scales_per_block + 2;
    let num_blocks = num_elements / QK_K;
    let mut acc = 0.0f32;

    for block_idx in 0..num_blocks {
        let block_start = block_idx * type_size;
        let block = &qdata[block_start..block_start + type_size];

        let ql = &block[..ql_per_block];
        let qh = &block[ql_per_block..ql_per_block + qh_per_block];
        let scales = &block[ql_per_block + qh_per_block..ql_per_block + qh_per_block + scales_per_block];
        let d = decode_f16([
            block[ql_per_block + qh_per_block + scales_per_block],
            block[ql_per_block + qh_per_block + scales_per_block + 1],
        ]);

        let vec_offset = block_idx * QK_K;
        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        let mut sc_off = 0usize;
        let mut y_off = 0usize;

        for _ in (0..QK_K).step_by(128) {
            for l in 0..32 {
                let is = l / 16;
                let qh_byte = qh[qh_off + l];
                let q1 = (((ql[ql_off + l] & 0x0F) | (((qh_byte >> 0) & 3) << 4)) as i8) - 32;
                let q2 = (((ql[ql_off + 32 + l] & 0x0F) | (((qh_byte >> 2) & 3) << 4)) as i8) - 32;
                let q3 = (((ql[ql_off + l] >> 4) | (((qh_byte >> 4) & 3) << 4)) as i8) - 32;
                let q4 = (((ql[ql_off + 32 + l] >> 4) | (((qh_byte >> 6) & 3) << 4)) as i8) - 32;

                let s1 = d * (scales[sc_off + is] as i8) as f32;
                let s2 = d * (scales[sc_off + 2 + is] as i8) as f32;
                let s3 = d * (scales[sc_off + 4 + is] as i8) as f32;
                let s4 = d * (scales[sc_off + 6 + is] as i8) as f32;

                acc += s1 * q1 as f32 * vec_f32[vec_offset + y_off + l];
                acc += s2 * q2 as f32 * vec_f32[vec_offset + y_off + 32 + l];
                acc += s3 * q3 as f32 * vec_f32[vec_offset + y_off + 64 + l];
                acc += s4 * q4 as f32 * vec_f32[vec_offset + y_off + 96 + l];
            }
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
            y_off += 128;
        }
    }
    acc
}

/// Dot product directo sobre bloques Q4_K contra un vector f32.
pub fn dot_q4_k_f32(qdata: &[u8], vec_f32: &[f32], num_elements: usize) -> f32 {
    let type_size = 2 * 2 + K_SCALE_SIZE + QK_K / 2;
    let num_blocks = num_elements / QK_K;
    let mut acc = 0.0f32;

    for block_idx in 0..num_blocks {
        let block_start = block_idx * type_size;
        let block = &qdata[block_start..block_start + type_size];

        let d = decode_f16([block[0], block[1]]);
        let dmin = decode_f16([block[2], block[3]]);
        let scales = &block[4..4 + K_SCALE_SIZE];
        let quants = &block[4 + K_SCALE_SIZE..];

        let vec_offset = block_idx * QK_K;
        let mut is = 0usize;
        let mut q_off = 0usize;
        let mut y_off = 0usize;

        for _ in (0..QK_K).step_by(64) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let d1 = d * f32::from(sc1);
            let min1 = dmin * f32::from(m1);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d2 = d * f32::from(sc2);
            let min2 = dmin * f32::from(m2);

            for l in 0..32 {
                let val = d1 * f32::from(quants[q_off + l] & 0x0F) - min1;
                acc += val * vec_f32[vec_offset + y_off + l];
            }
            for l in 0..32 {
                let val = d2 * f32::from(quants[q_off + l] >> 4) - min2;
                acc += val * vec_f32[vec_offset + y_off + 32 + l];
            }

            q_off += 32;
            is += 2;
            y_off += 64;
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_block_dequantizes_expected_values() {
        let mut block = vec![0u8; 34];
        block[0..2].copy_from_slice(&f16::from_f32(0.5).to_bits().to_le_bytes());
        for (index, value) in [2i8, -4, 6, -8].into_iter().enumerate() {
            block[2 + index] = value as u8;
        }
        let out = dequantize_q8_0_block(&block).unwrap();
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], -2.0);
        assert_eq!(out[2], 3.0);
        assert_eq!(out[3], -4.0);
    }

    #[test]
    fn q4_k_block_length_is_enforced() {
        let err = dequantize_q4_k_block(&[0u8; 10]).unwrap_err().to_string();
        assert!(err.contains("Bloque Q4_K invalido"));
    }

    #[test]
    fn dot_q8_0_matches_dequantize_then_dot() {
        // Crear un bloque Q8_0 con valores conocidos
        let mut block = vec![0u8; 34]; // 2 + 32
        block[0..2].copy_from_slice(&f16::from_f32(0.25).to_bits().to_le_bytes());
        for i in 0..32u8 {
            block[2 + i as usize] = (i as i8 - 16) as u8;
        }
        // Vector f32 de referencia
        let vec_f32: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();

        // Metodo 1: dequantizar y luego dot
        let dequant = dequantize_q8_0_block(&block).unwrap();
        let expected: f32 = dequant.iter().zip(vec_f32.iter()).map(|(a, b)| a * b).sum();

        // Metodo 2: dot directo cuantizado
        let actual = dot_q8_0_f32(&block, &vec_f32, 32);

        let diff = (expected - actual).abs();
        assert!(diff < 1e-3, "dot_q8_0 diverge: expected={expected}, actual={actual}, diff={diff}");
    }
}
