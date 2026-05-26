use std::arch::x86_64::{
    __m512, __m512i, _mm512_and_si512, _mm512_div_ps, _mm512_loadu_epi32, _mm512_loadu_epi64, _mm512_loadu_ps, _mm512_or_si512, _mm512_set1_epi64, _mm512_set_epi64, _mm512_slli_epi16, _mm512_slli_epi64, _mm512_store_epi64, _mm512_storeu_epi64, _mm512_xor_si512
};

use fixed::FixedI32;
use fixed::types::extra::{U9, U32};

#[inline(always)]
pub fn to_fixed(n: f64) -> FixedI32<U32> {
    fixed::FixedI32::from_num(n)
}

#[inline(always)]
pub fn scale_lat(n: f64) -> f64 {
    n / 90.
}

#[inline(always)]
pub fn scale_lon(n: f64) -> f64 {
    n / -180.
}

#[inline(always)]
pub fn scale_lot_simd(word: __m512) -> __m512 {
    let divisor = [-180_f32; 16];
    unsafe{
        let div = _mm512_loadu_ps(divisor.as_ptr());
        _mm512_div_ps(word, div)
    }
}


#[inline(always)]
pub fn scalefix_lat(n: f64) -> FixedI32<U32> {
    to_fixed(scale_lat(n))
}

#[inline(always)]
pub fn scalefix_lon(n: f64) -> FixedI32<U32> {
    to_fixed(scale_lon(n))
}

#[inline(always)]
/// Interleaves an i32 with 0s, making an i64 in the process;
/// takes the `.to_bits()` representation of a FixedI32 number.
pub fn naive_zero_interleave(a: i32) -> i64 {
    // cast as i64 for working space
    let mut word = a as i64;

    // bit shift 16 over, OR with itself (makes two copies)
    // and then mask lower bytes
    // ffff is 16 bits of 1s
    word = (word ^ (word << 16)) & 0x0000ffff0000ffff;
    // bit shift & over, OR with itself (again two copies);
    // mask lower bytes of copies again.
    // ff is a full byte of 1s
    word = (word ^ (word << 8)) & 0x00ff00ff00ff00ff;
    // bit shifts 4 over & makes two copies, grabbing
    // only the lower one again in a smaller pattern.
    // 0f is 00001111
    word = (word ^ (word << 4)) & 0x0f0f0f0f0f0f0f0f;
    // 3 =  0011;
    word = (word ^ (word << 2)) & 0x3333333333333333;
    // 5 = 0101;
    word = (word ^ (word << 1)) & 0x5555555555555555;
    word
}

pub fn naive_zcode_latlon(lat: f64, lon: f64) -> i64 {
    let latitude = scalefix_lat(lat);
    let longitude = scalefix_lon(lon);

    let lat_bits = naive_zero_interleave(latitude.to_bits());
    let lon_bits = naive_zero_interleave(longitude.to_bits());

    // longitude goes in the high bits of the interleave
    (lon_bits << 1) | lat_bits
}

// this is the 'inner' function for the 'outer' interleaving
pub fn simd_zero_interleave(mut word: __m512i) -> __m512i {
    unsafe {
        // these ar ethe same as masks 3, 4, and 5 from the scalar interleave above, we just broadcast them to multiple inputs...
        let m1: __m512i = _mm512_set1_epi64(0x0000ffff0000ffff);
        let m2: __m512i = _mm512_set1_epi64(0x00ff00ff00ff00ff);
        let m3: __m512i = _mm512_set1_epi64(0x0f0f0f0f0f0f0f0f);
        let m4: __m512i = _mm512_set1_epi64(0x3333333333333333);
        let m5: __m512i = _mm512_set1_epi64(0x5555555555555555);

        word = _mm512_xor_si512(word, _mm512_slli_epi64(word, 16));
        word = _mm512_and_si512(word, m1);
        word = _mm512_xor_si512(word, _mm512_slli_epi64(word, 8));
        word = _mm512_and_si512(word, m2);
        word = _mm512_xor_si512(word, _mm512_slli_epi64(word, 4));
        word = _mm512_and_si512(word, m3);
        word = _mm512_xor_si512(word, _mm512_slli_epi64(word, 2));
        word = _mm512_and_si512(word, m4);
        word = _mm512_xor_si512(word, _mm512_slli_epi64(word, 1));
        word = _mm512_and_si512(word, m5);
        word
    }
}

pub fn interleave_latlon(lat: Vec<f64>, lon: Vec<f64>) -> Vec<i64> {
    // step 1: pack down our lats and lons to SIMD sizes.
    let preprocessed_lat: Vec<i64> = lat
        .iter()
        .map(|l| scalefix_lat(*l).to_bits() as i64)
        .collect();
    let preprocessed_lon: Vec<i64> = lon
        .iter()
        .map(|l| scalefix_lon(*l).to_bits() as i64)
        .collect();

    unsafe {
        let ileaved_lat: Vec<__m512i> = preprocessed_lat
            .chunks_exact(8)
            .map(|chunk| simd_zero_interleave(_mm512_loadu_epi64(chunk.as_ptr())))
            .collect();
        let ileaved_lon: Vec<__m512i> = preprocessed_lon
            .chunks_exact(8)
            .map(|chunk| simd_zero_interleave(_mm512_loadu_epi64(chunk.as_ptr())))
            .collect();

        ileaved_lat
            .iter()
            .zip(ileaved_lon)
            .into_iter()
            .map(|(lat, lon)| {
                let jndreg: __m512i = _mm512_or_si512(_mm512_slli_epi16(lon, 1), *lat);
                let mut outbuf = [0i64; 8];
                _mm512_storeu_epi64(outbuf.as_mut_ptr(), jndreg);
                outbuf
            })
            .flatten()
            .collect()
    }
}

// pub mod helpers{
//    pub fn generate_ll_seq(start: i32, end: i32, ndec: i32) -> Vec<f64> {
//        let mut outvec: Vec<f64> = Vec::new();
//        for i in start..=end {
//            for j in 0..=ndec {
//                outvec.push(i as f64 + (j as f64 / ndec as f64))
//            }
//        }

//        outvec
//    }
// }
