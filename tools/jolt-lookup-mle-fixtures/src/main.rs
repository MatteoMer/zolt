use num_bigint::BigUint;
use num_traits::{One, Zero};
use std::{env, fs, path::PathBuf};

const UPSTREAM_JOLT_REV: &str = "b10e80ac6a9b62672bd196b8a6daf7dce652c9b3";
const CASES_PER_TABLE: usize = 4;
const XLEN: usize = 64;
const NUM_LOOKUP_BITS: usize = 2 * XLEN;
const LCG_MUL: u64 = 6_364_136_223_846_793_005;
const LCG_INC: u64 = 1_442_695_040_888_963_407;
const CHALLENGE_MODULUS: u64 = 1_000_000;

// Field modulus for ark_bn254::Fr used by upstream Jolt at UPSTREAM_JOLT_REV.
const BN254_FR_MODULUS_BE: &str =
    "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001";

const TABLES: &[(usize, &str)] = &[
    (22, "Pow2W"),
    (29, "VirtualChangeDivisor"),
    (30, "VirtualChangeDivisorW"),
    (31, "MulUNoOverflow"),
    (32, "VirtualXORROT32"),
    (33, "VirtualXORROT24"),
    (34, "VirtualXORROT16"),
    (35, "VirtualXORROT63"),
    (36, "VirtualXORROTW16"),
    (37, "VirtualXORROTW12"),
    (38, "VirtualXORROTW8"),
    (39, "VirtualXORROTW7"),
];

#[derive(Clone, Debug, Eq, PartialEq)]
struct Fr {
    n: BigUint,
}

impl Fr {
    fn modulus() -> BigUint {
        BigUint::parse_bytes(BN254_FR_MODULUS_BE.as_bytes(), 16).unwrap()
    }

    fn new(n: BigUint) -> Self {
        Self {
            n: n % Self::modulus(),
        }
    }

    fn zero() -> Self {
        Self { n: BigUint::zero() }
    }

    fn one() -> Self {
        Self { n: BigUint::one() }
    }

    fn from_u64(n: u64) -> Self {
        Self::new(BigUint::from(n))
    }

    fn from_u128(n: u128) -> Self {
        Self::new(BigUint::from(n))
    }

    fn to_hex_be_32(&self) -> String {
        let mut bytes = self.n.to_bytes_be();
        if bytes.len() < 32 {
            let mut padded = vec![0u8; 32 - bytes.len()];
            padded.extend_from_slice(&bytes);
            bytes = padded;
        }
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }
}

impl std::ops::Add for Fr {
    type Output = Fr;
    fn add(self, rhs: Fr) -> Fr {
        Fr::new(self.n + rhs.n)
    }
}

impl std::ops::AddAssign for Fr {
    fn add_assign(&mut self, rhs: Fr) {
        *self = self.clone() + rhs;
    }
}

impl std::ops::Sub for Fr {
    type Output = Fr;
    fn sub(self, rhs: Fr) -> Fr {
        let m = Fr::modulus();
        Fr::new((self.n + &m) - rhs.n)
    }
}

impl std::ops::Mul for Fr {
    type Output = Fr;
    fn mul(self, rhs: Fr) -> Fr {
        Fr::new(self.n * rhs.n)
    }
}

impl std::ops::MulAssign for Fr {
    fn mul_assign(&mut self, rhs: Fr) {
        *self = self.clone() * rhs;
    }
}

fn evaluate_upstream_formula(table_index: usize, r: &[Fr]) -> Fr {
    // These are direct translations of a16z/jolt jolt-core lookup table
    // evaluate_mle implementations at UPSTREAM_JOLT_REV:
    // pow2_w.rs, virtual_change_divisor.rs, virtual_change_divisor_w.rs,
    // mulu_no_overflow.rs, virtual_xor_rot.rs, and virtual_xor_rotw.rs.
    match table_index {
        22 => eval_pow2_w(r),
        29 => eval_change_divisor(r),
        30 => eval_change_divisor_w(r),
        31 => eval_mulu_no_overflow(r),
        32 => eval_xor_rot(r, 32),
        33 => eval_xor_rot(r, 24),
        34 => eval_xor_rot(r, 16),
        35 => eval_xor_rot(r, 63),
        36 => eval_xor_rotw(r, 16),
        37 => eval_xor_rotw(r, 12),
        38 => eval_xor_rotw(r, 8),
        39 => eval_xor_rotw(r, 7),
        _ => panic!("unsupported table index {table_index}"),
    }
}

fn eval_pow2_w(r: &[Fr]) -> Fr {
    let mut result = Fr::one();
    for i in 0..5 {
        result *= Fr::one() + Fr::from_u64((1 << (1 << i)) - 1) * r[r.len() - i - 1].clone();
    }
    result
}

fn eval_change_divisor(r: &[Fr]) -> Fr {
    let mut divisor_value = Fr::zero();
    for i in 0..XLEN {
        divisor_value += Fr::from_u128(1u128 << (XLEN - 1 - i)) * r[2 * i + 1].clone();
    }
    let mut x_product = r[0].clone();
    for i in 1..XLEN {
        x_product *= Fr::one() - r[2 * i].clone();
    }
    let mut y_product = Fr::one();
    for i in 0..XLEN {
        y_product *= r[2 * i + 1].clone();
    }
    let adjustment = Fr::from_u64(2) - Fr::from_u128(1u128 << XLEN);
    divisor_value + x_product * y_product * adjustment
}

fn eval_change_divisor_w(r: &[Fr]) -> Fr {
    let sign_bit = r[XLEN + 1].clone();
    let mut divisor_value = Fr::zero();
    for i in XLEN / 2..XLEN {
        divisor_value += Fr::from_u64(1u64 << (XLEN - 1 - i)) * r[2 * i + 1].clone();
    }
    let mut x_product = r[XLEN].clone();
    for i in XLEN / 2 + 1..XLEN {
        x_product *= Fr::one() - r[2 * i].clone();
    }
    let mut y_product = Fr::one();
    for i in XLEN / 2..XLEN {
        y_product *= r[2 * i + 1].clone();
    }
    let sign_extension = Fr::from_u128((1u128 << XLEN) - (1u128 << (XLEN / 2))) * sign_bit;
    let adjustment = Fr::from_u64(2) - Fr::from_u128(1u128 << XLEN);
    divisor_value + adjustment * x_product * y_product + sign_extension
}

fn eval_mulu_no_overflow(r: &[Fr]) -> Fr {
    let mut result = Fr::one();
    for value in r.iter().take(XLEN) {
        result *= Fr::one() - value.clone();
    }
    result
}

fn eval_xor_rot(r: &[Fr], rotation: usize) -> Fr {
    let mut result = Fr::zero();
    for i in 0..XLEN {
        let x_i = r[2 * i].clone();
        let y_i = r[2 * i + 1].clone();
        let rotated_position = (i + rotation) % XLEN;
        let bit_position = XLEN - 1 - rotated_position;
        let xor_bit = (Fr::one() - x_i.clone()) * y_i.clone() + x_i * (Fr::one() - y_i);
        result += Fr::from_u64(1u64 << bit_position) * xor_bit;
    }
    result
}

fn eval_xor_rotw(r: &[Fr], rotation: usize) -> Fr {
    let mut result = Fr::zero();
    for idx in XLEN / 2..XLEN {
        let r_x = r[2 * idx].clone();
        let r_y = r[2 * idx + 1].clone();
        let xor_bit = (Fr::one() - r_x.clone()) * r_y.clone() + r_x * (Fr::one() - r_y);
        let position = idx - (XLEN / 2);
        let rotated_position = (XLEN / 2) - 1 - ((position + rotation) % (XLEN / 2));
        result += Fr::from_u64(1u64 << rotated_position) * xor_bit;
    }
    result
}

fn seed_for(table_index: usize, case_index: usize) -> u64 {
    0x6a09_e667_f3bc_c909u64
        ^ ((table_index as u64) << 32)
        ^ (case_index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

fn non_boolean_challenges(seed: u64) -> Vec<Fr> {
    let mut state = seed;
    let mut r = Vec::with_capacity(NUM_LOOKUP_BITS);
    for _ in 0..NUM_LOOKUP_BITS {
        state = state.wrapping_mul(LCG_MUL).wrapping_add(LCG_INC);
        r.push(Fr::from_u64(2 + (state % CHALLENGE_MODULUS)));
    }
    r
}

fn output_path_from_args() -> PathBuf {
    let mut out = PathBuf::from("testdata/lookup-table-mle/nonboolean_bn254.txt");
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => {
                out = args
                    .next()
                    .map(PathBuf::from)
                    .expect("--out requires a path argument");
            }
            "--help" | "-h" => {
                println!(
                    "usage: cargo run --release --manifest-path tools/jolt-lookup-mle-fixtures/Cargo.toml -- --out testdata/lookup-table-mle/nonboolean_bn254.txt"
                );
                std::process::exit(0);
            }
            other => panic!("unknown argument {other}"),
        }
    }
    out
}

fn main() {
    let out_path = output_path_from_args();
    let mut out = String::new();
    out.push_str(&format!(
        "# Generated by tools/jolt-lookup-mle-fixtures from a16z/jolt rev {UPSTREAM_JOLT_REV}\n"
    ));
    out.push_str("# format: table_index|table_name|case|seed_u64|expected_fr_be_hex\n");

    for &(table_index, table_name) in TABLES {
        for case_index in 0..CASES_PER_TABLE {
            let seed = seed_for(table_index, case_index);
            let r = non_boolean_challenges(seed);
            let expected = evaluate_upstream_formula(table_index, &r);
            out.push_str(&format!(
                "{table_index}|{table_name}|random_{case_index}|{seed}|{}\n",
                expected.to_hex_be_32()
            ));
        }
    }

    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent).expect("create fixture parent directory");
    }
    fs::write(&out_path, out).expect("write lookup-table MLE fixture");
    println!("wrote {}", out_path.display());
}
