use std::arch::x86_64::_mm512_scalef_pd;

use divan::Bencher;
use fixed::FixedI32;
use fixed::types::extra::{U9, U32};
use geohash::{
    interleave_latlon, naive_zcode_latlon, naive_zero_interleave, scale_lat, scale_lon,
    scalefix_lat, scalefix_lon, to_fixed,
};

pub fn main() {
    divan::main()
}

#[divan::bench(args = [0.1241274, -18.18101, 42., 42.1, 42.11, 42.122, 42.12412412, 42.2418176981])]
fn scalefix_one(n: f64) -> FixedI32<U32> {
    scalefix_lat(n)
}

#[divan::bench(args = [0.1241274, -18.18101, 42., 42.1, 42.11, 42.122, 42.12412412, 42.2418176981])]
fn scale_one(n: f64) -> f64 {
    scale_lat(n)
}

#[divan::bench]
fn fix_one(bencher: Bencher) {
    let value: f64 = scale_lat(42.18101701274);
    bencher.bench_local(|| to_fixed(value));
}

#[divan::bench(threads = true)]
fn manyconv_threaded(bencher: Bencher) {
    let mut out: Vec<f64> = Vec::new();
    for i in -92..-76 {
        for j in 0..=1999 {
            out.push((i as f64) + (j as f64 / 1000.))
        }
    }

    bencher.bench(|| {
        out.clone()
            .into_iter()
            .map(|v| scalefix_lon(v))
            .collect::<Vec<_>>()
    })
}

#[divan::bench(threads = false)]
fn manyconv_single(bencher: Bencher) {
    let mut out: Vec<f64> = Vec::new();
    for i in -92..-76 {
        for j in 0..=1999 {
            out.push((i as f64) + (j as f64 / 2000.))
        }
    }

    bencher.bench(|| {
        out.clone()
            .into_iter()
            .map(|v| scalefix_lon(v))
            .collect::<Vec<_>>()
    })
}

#[divan::bench]
fn naive_zero_interleave_one(bencher: Bencher) {
    let x = -88.2821308;
    let lat = scalefix_lon(x);

    bencher.bench(|| naive_zero_interleave(lat.to_bits()));
}

#[divan::bench]
fn zcode_latlon_scalar(bencher: Bencher) {
    let lon = -88.2821308;
    let lat = 42.9280809233;

    bencher.bench(|| naive_zcode_latlon(lat, lon));
}

#[divan::bench]
fn zcode_latlon_scalar_many_singlethread(bencher: Bencher) {
    let lon_out: Vec<f64> = generate_ll_seq(-87, -80, 100);
    let lat_out: Vec<f64> = generate_ll_seq(30, 37, 100);

    bencher.bench(move || {
        for lat in &lat_out {
            for lon in &lon_out {
                naive_zcode_latlon(*lat, *lon);
            }
        }
    });
}

#[divan::bench(threads = true)]
fn zcode_latlon_scalar_many_multithread(bencher: Bencher) {
    let lon_out: Vec<f64> = generate_ll_seq(-87, -80, 100);
    let lat_out: Vec<f64> = generate_ll_seq(30, 37, 100);

    bencher.bench(move || {
        for lat in &lat_out {
            for lon in &lon_out {
                naive_zcode_latlon(*lat, *lon);
            }
        }
    });
}

#[divan::bench(threads = false, args = [5, 10, 50, 100, 200,300,500])]
fn zcode_latlon_simd_many_sthread(bencher: Bencher, len: i32) {
    bencher
        .with_inputs(|| generate_arraylike_seq(generate_ll_seq(30, 37, len), generate_ll_seq(-87, -80, len)))
        .bench_local_values(move |(lat_arry, lon_arry)| interleave_latlon(lat_arry.clone(), lon_arry.clone()));
}

#[divan::bench(threads=true, args = [5, 10, 50, 100, 200,300,500])]
fn zcode_latlon_simd_many_mthread(bencher: Bencher, len: i32) {
    bencher
        .with_inputs(|| generate_arraylike_seq(generate_ll_seq(30, 37, len), generate_ll_seq(-87, -80, len)))
        .bench_values(move |(lat_arry, lon_arry)| {
            interleave_latlon(lat_arry.clone(), lon_arry.clone())
        });
}

// helper function to generate lat or lon sequences
fn generate_ll_seq(start: i32, end: i32, ndec: i32) -> Vec<f64> {
    let mut outvec: Vec<f64> = Vec::new();
    for i in start..=end {
        for j in 0..=ndec {
            outvec.push(i as f64 + (j as f64 / ndec as f64))
        }
    }

    outvec
}

fn generate_arraylike_seq(lat: Vec<f64>, lon: Vec<f64>) -> (Vec<f64>, Vec<f64>){
    let mut lon_out_arraylike: Vec<f64> = Vec::with_capacity(lat.len() * lon.len());
    let mut lat_out_arraylike: Vec<f64> = Vec::with_capacity(lat.len() * lon.len());

    for _ in 0..lon.len(){
        lat_out_arraylike.extend(lat.clone().iter())
    };
    for _ in 0..lat.len(){
        lon_out_arraylike.extend(lon.clone().iter())
    };

    (lat_out_arraylike, lon_out_arraylike)
}
