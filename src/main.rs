use std::arch::x86_64::{_mm512_load_epi32, _mm512_loadu_epi32, _mm512_mask_shuffle_pd};

use fixed::FixedI32;
use fixed::traits::Fixed;
use fixed::types::extra::{U9, U32};
use geohash::{
    interleave_latlon, naive_zcode_latlon, scale_lat, scale_lon, scalefix_lat, scalefix_lon,
    to_fixed,
};

fn main() {
    println!("{}", naive_zcode_latlon(42.581208, -88.1280810));

    let seq = generate_ll_seq(30, 37, 1000);
    println!("Seq len: {}", seq.len());
    println!("Seq divisible by 8: {}", (seq.len() % 8) == 0);

    let seq2 = generate_ll_seq(-87, -80, 1000);
    println!("seq2 len: {}", seq.len());
    println!("seq2 divisible by 8: {}", (seq.len() % 8) == 0);

    let latseq = generate_ll_seq(30, 38, 10);
    let lonseq = generate_ll_seq(-88, -80, 10);

    let codes = interleave_latlon(latseq.clone(), lonseq.clone());
    //println!("{:?}", &codes);

    let codes2: Vec<i64> = latseq
        .iter()
        .zip(lonseq)
        .map(|(lat, lon)| naive_zcode_latlon(*lat, lon))
        .collect();
    //println!("{:?}", &codes2);

    let equivs: Vec<bool> = codes
        .iter()
        .zip(codes2)
        .map(|(code1, code2)| *code1 == code2)
        .collect();
    println!(
        "All values match each other: {:?}",
        &equivs.iter().all(|e| *e == true)
    );

    println!("{}",(1 << 20) as i64);
    println!("{}",(-81.23808_f64 / -180.));
    println!("{}",((-81.23808_f64 / -180.) * (1 << 20) as f64));
    println!("{}",((-81.23808_f64 / -180.)* (1 << 20) as f64) as i32);
}

fn generate_ll_seq(start: i32, end: i32, ndec: i32) -> Vec<f64> {
    let mut outvec: Vec<f64> = Vec::new();
    for i in start..=end {
        for j in 0..=ndec {
            outvec.push(i as f64 + (j as f64 / ndec as f64))
        }
    }

    outvec
}
