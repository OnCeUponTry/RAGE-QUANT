use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rage_quant::ggml_quant::{dot_q8_0_f32, dequantize_q8_0_block};
use half::f16;

fn create_q8_0_blocks(num_blocks: usize) -> Vec<u8> {
    let type_size = 2 + 32; // 2 bytes f16 scale + 32 bytes quants
    let mut data = vec![0u8; num_blocks * type_size];
    for b in 0..num_blocks {
        let offset = b * type_size;
        let scale = f16::from_f32(0.1 + (b as f32) * 0.001);
        data[offset..offset + 2].copy_from_slice(&scale.to_bits().to_le_bytes());
        for i in 0..32 {
            data[offset + 2 + i] = ((i as i8) - 16) as u8;
        }
    }
    data
}

fn bench_dot_q8_0(c: &mut Criterion) {
    let num_elements = 4096;
    let num_blocks = num_elements / 32;
    let qdata = create_q8_0_blocks(num_blocks);
    let vec_f32: Vec<f32> = (0..num_elements).map(|i| (i as f32) * 0.001).collect();

    c.bench_function("dot_q8_0_f32_4096", |b| {
        b.iter(|| black_box(dot_q8_0_f32(black_box(&qdata), black_box(&vec_f32), num_elements)))
    });
}

fn bench_dequantize_then_dot(c: &mut Criterion) {
    let num_elements = 4096;
    let num_blocks = num_elements / 32;
    let type_size = 2 + 32;
    let qdata = create_q8_0_blocks(num_blocks);
    let vec_f32: Vec<f32> = (0..num_elements).map(|i| (i as f32) * 0.001).collect();

    c.bench_function("dequantize_then_dot_4096", |b| {
        b.iter(|| {
            let mut acc = 0.0f32;
            let mut offset = 0;
            for block_idx in 0..num_blocks {
                let block = &qdata[block_idx * type_size..(block_idx + 1) * type_size];
                let deq = dequantize_q8_0_block(block).unwrap();
                let base = block_idx * 32;
                for i in 0..32 {
                    acc += deq[i] * vec_f32[base + i];
                }
                offset += type_size;
            }
            black_box(acc)
        })
    });
}

criterion_group!(benches, bench_dot_q8_0, bench_dequantize_then_dot);
criterion_main!(benches);
