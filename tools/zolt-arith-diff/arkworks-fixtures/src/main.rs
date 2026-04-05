use ark_bn254::{Bn254, Fq, Fq12, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{
    pairing::Pairing,
    scalar_mul::variable_base::{msm_i128, VariableBaseMSM},
    AdditiveGroup, AffineRepr, CurveGroup, PrimeGroup,
};
use ark_ff::{BigInteger, Field, PrimeField, UniformRand, Zero};
use ark_std::rand::{rngs::StdRng, SeedableRng};
use std::{
    env, fs,
    path::{Path, PathBuf},
};

fn hex_le<F: PrimeField>(value: &F) -> String {
    let mut bytes = value.into_bigint().to_bytes_le();
    bytes.resize(32, 0);
    hex_encode(&bytes)
}

fn hex_be<F: PrimeField>(value: &F) -> String {
    let mut bytes = value.into_bigint().to_bytes_be();
    if bytes.len() < 32 {
        let mut padded = vec![0u8; 32 - bytes.len()];
        padded.extend_from_slice(&bytes);
        bytes = padded;
    }
    hex_encode(&bytes)
}

fn hex_encode(bytes: &[u8]) -> String {
    const LUT: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(LUT[(byte >> 4) as usize] as char);
        out.push(LUT[(byte & 0x0f) as usize] as char);
    }
    out
}

fn fq12_hex_le(value: &Fq12) -> String {
    let coords = [
        &value.c0.c0.c0,
        &value.c0.c0.c1,
        &value.c0.c1.c0,
        &value.c0.c1.c1,
        &value.c0.c2.c0,
        &value.c0.c2.c1,
        &value.c1.c0.c0,
        &value.c1.c0.c1,
        &value.c1.c1.c0,
        &value.c1.c1.c1,
        &value.c1.c2.c0,
        &value.c1.c2.c1,
    ];

    let mut bytes = Vec::with_capacity(384);
    for coord in coords {
        let mut le = coord.into_bigint().to_bytes_le();
        le.resize(32, 0);
        bytes.extend_from_slice(&le);
    }
    hex_encode(&bytes)
}

fn write_file(path: PathBuf, contents: String) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent dirs");
    }
    fs::write(path, contents).expect("write fixture file");
}

fn generate_g1_bases(n: usize) -> Vec<G1Affine> {
    let gen = G1Affine::generator();
    let mut proj = G1Projective::from(gen);
    let mut bases = Vec::with_capacity(n);
    for _ in 0..n {
        bases.push(proj.into_affine());
        proj = proj.double();
    }
    bases
}

fn generate_g2_bases(n: usize) -> Vec<G2Affine> {
    let gen = G2Affine::generator();
    let mut proj = G2Projective::from(gen);
    let mut bases = Vec::with_capacity(n);
    for _ in 0..n {
        bases.push(proj.into_affine());
        proj = proj.double();
    }
    bases
}

fn generate_field_ops(out_dir: &Path) {
    let mut rng = StdRng::seed_from_u64(0x5eed_f11d_2540);

    let fr_cases = [
        (Fr::from(0u64), Fr::from(1u64)),
        (Fr::from(1u64), Fr::from(2u64)),
        (Fr::from(7u64), Fr::from(3u64)),
        (Fr::rand(&mut rng), Fr::rand(&mut rng)),
        (Fr::rand(&mut rng), Fr::rand(&mut rng)),
        (Fr::rand(&mut rng), Fr::rand(&mut rng)),
    ];

    let fq_cases = [
        (Fq::from(0u64), Fq::from(1u64)),
        (Fq::from(1u64), Fq::from(2u64)),
        (Fq::from(11u64), Fq::from(19u64)),
        (Fq::rand(&mut rng), Fq::rand(&mut rng)),
        (Fq::rand(&mut rng), Fq::rand(&mut rng)),
        (Fq::rand(&mut rng), Fq::rand(&mut rng)),
    ];

    let mut fr_out = String::from("# op|a_be_hex|b_be_hex|expected_be_hex\n");
    for (a, b) in fr_cases {
        fr_out.push_str(&format!("add|{}|{}|{}\n", hex_be(&a), hex_be(&b), hex_be(&(a + b))));
        fr_out.push_str(&format!("sub|{}|{}|{}\n", hex_be(&a), hex_be(&b), hex_be(&(a - b))));
        fr_out.push_str(&format!("mul|{}|{}|{}\n", hex_be(&a), hex_be(&b), hex_be(&(a * b))));
        if !a.is_zero() {
            fr_out.push_str(&format!("inv|{}|-|{}\n", hex_be(&a), hex_be(&a.inverse().unwrap())));
        }
    }

    let mut fq_out = String::from("# op|a_be_hex|b_be_hex|expected_be_hex\n");
    for (a, b) in fq_cases {
        fq_out.push_str(&format!("add|{}|{}|{}\n", hex_be(&a), hex_be(&b), hex_be(&(a + b))));
        fq_out.push_str(&format!("sub|{}|{}|{}\n", hex_be(&a), hex_be(&b), hex_be(&(a - b))));
        fq_out.push_str(&format!("mul|{}|{}|{}\n", hex_be(&a), hex_be(&b), hex_be(&(a * b))));
        if !a.is_zero() {
            fq_out.push_str(&format!("inv|{}|-|{}\n", hex_be(&a), hex_be(&a.inverse().unwrap())));
        }
    }

    write_file(out_dir.join("field/fr_ops.txt"), fr_out);
    write_file(out_dir.join("field/fp_ops.txt"), fq_out);
}

fn generate_pairing_cases(out_dir: &Path) {
    let cases: &[(u64, u64)] = &[(0, 1), (1, 0), (1, 1), (2, 3), (5, 7), (17, 19)];
    let g1_gen = G1Affine::generator();
    let g2_gen = G2Affine::generator();

    let mut out = String::from("# name|g1_scalar_u64|g2_scalar_u64|expected_fp12_le_hex\n");
    for (index, (g1_scalar, g2_scalar)) in cases.iter().enumerate() {
        let g1 = G1Projective::from(g1_gen)
            .mul_bigint(Fr::from(*g1_scalar).into_bigint())
            .into_affine();
        let g2 = G2Projective::from(g2_gen)
            .mul_bigint(Fr::from(*g2_scalar).into_bigint())
            .into_affine();
        let expected = Bn254::pairing(g1, g2).0;
        out.push_str(&format!(
            "case_{index}|{g1_scalar}|{g2_scalar}|{}\n",
            fq12_hex_le(&expected)
        ));
    }

    write_file(out_dir.join("pairing/generator_cases.txt"), out);
}

fn format_g1_point(point: G1Affine) -> (u8, String, String) {
    if point.infinity {
        return (1, String::new(), String::new());
    }
    (0, hex_le(&point.x), hex_le(&point.y))
}

fn format_g2_point(point: G2Affine) -> (u8, String, String, String, String) {
    if point.infinity {
        return (1, String::new(), String::new(), String::new(), String::new());
    }
    (
        0,
        hex_le(&point.x.c0),
        hex_le(&point.x.c1),
        hex_le(&point.y.c0),
        hex_le(&point.y.c1),
    )
}

fn join_field_scalars(values: &[Fr]) -> String {
    values.iter().map(hex_be).collect::<Vec<_>>().join(",")
}

fn join_i128_scalars(values: &[i128]) -> String {
    values.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",")
}

fn generate_i128_scalars(n: usize) -> Vec<i128> {
    let mut seed: u64 = 0xcafebabe;
    (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let magnitude = (seed & 0x7fff_ffff_ffff_ffff) as i128;
            if (seed & 1) == 0 {
                magnitude
            } else {
                -magnitude
            }
        })
        .collect()
}

fn generate_msm_cases(out_dir: &Path) {
    let g1_sizes = [1usize, 2, 4, 8, 16];
    let g2_sizes = [1usize, 2, 4, 8];
    let max_g1 = *g1_sizes.iter().max().unwrap();
    let max_g2 = *g2_sizes.iter().max().unwrap();

    let g1_bases = generate_g1_bases(max_g1);
    let g2_bases = generate_g2_bases(max_g2);

    let mut rng = StdRng::seed_from_u64(0x51ca_4f11);
    let fr_scalars: Vec<Fr> = (0..max_g1).map(|_| Fr::rand(&mut rng)).collect();
    let i128_scalars = generate_i128_scalars(max_g1);

    let mut g1_fr_out = String::from("# name|scalars_be_hex_csv|expected_infinity|expected_x_le_hex|expected_y_le_hex\n");
    for (index, &n) in g1_sizes.iter().enumerate() {
        let expected = G1Projective::msm(&g1_bases[..n], &fr_scalars[..n]).unwrap().into_affine();
        let (inf, x, y) = format_g1_point(expected);
        g1_fr_out.push_str(&format!(
            "case_{index}|{}|{inf}|{x}|{y}\n",
            join_field_scalars(&fr_scalars[..n])
        ));
    }

    let mut g1_i128_out = String::from("# name|scalars_i128_csv|expected_infinity|expected_x_le_hex|expected_y_le_hex\n");
    for (index, &n) in g1_sizes.iter().enumerate() {
        let expected = msm_i128::<G1Projective>(&g1_bases[..n], &i128_scalars[..n], true).into_affine();
        let (inf, x, y) = format_g1_point(expected);
        g1_i128_out.push_str(&format!(
            "case_{index}|{}|{inf}|{x}|{y}\n",
            join_i128_scalars(&i128_scalars[..n])
        ));
    }

    let mut g2_fr_out = String::from(
        "# name|scalars_be_hex_csv|expected_infinity|expected_x_c0_le_hex|expected_x_c1_le_hex|expected_y_c0_le_hex|expected_y_c1_le_hex\n",
    );
    for (index, &n) in g2_sizes.iter().enumerate() {
        let expected = G2Projective::msm(&g2_bases[..n], &fr_scalars[..n]).unwrap().into_affine();
        let (inf, x0, x1, y0, y1) = format_g2_point(expected);
        g2_fr_out.push_str(&format!(
            "case_{index}|{}|{inf}|{x0}|{x1}|{y0}|{y1}\n",
            join_field_scalars(&fr_scalars[..n])
        ));
    }

    write_file(out_dir.join("msm/g1_fr_cases.txt"), g1_fr_out);
    write_file(out_dir.join("msm/g1_i128_cases.txt"), g1_i128_out);
    write_file(out_dir.join("msm/g2_fr_cases.txt"), g2_fr_out);
}

fn parse_out_dir() -> PathBuf {
    let mut args = env::args().skip(1);
    let mut out_dir = PathBuf::from("testdata/zolt-arith-diff");

    while let Some(arg) = args.next() {
        if arg == "--out-dir" {
            out_dir = PathBuf::from(args.next().expect("missing value for --out-dir"));
        } else {
            panic!("unknown argument: {arg}");
        }
    }

    out_dir
}

fn main() {
    let out_dir = parse_out_dir();
    generate_field_ops(&out_dir);
    generate_pairing_cases(&out_dir);
    generate_msm_cases(&out_dir);
    eprintln!("generated zolt-arith differential fixtures under {}", out_dir.display());
}
