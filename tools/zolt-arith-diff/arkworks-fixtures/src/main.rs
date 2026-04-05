mod transcript;

use ark_bn254::{Bn254, Fq, Fq2, Fq6, Fq12, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{
    pairing::Pairing,
    scalar_mul::variable_base::{msm_i128, VariableBaseMSM},
    AdditiveGroup, AffineRepr, CurveGroup, PrimeGroup,
};
use ark_ff::{BigInteger, CyclotomicMultSubgroup, Field, PrimeField, UniformRand, Zero, One};
use ark_serialize::CanonicalSerialize;
use ark_std::rand::{rngs::StdRng, SeedableRng};
use sha3::{Sha3_256, Digest};
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

fn fq2_hex_le(value: &Fq2) -> String {
    let coords = [&value.c0, &value.c1];
    let mut bytes = Vec::with_capacity(64);
    for coord in coords {
        let mut le = coord.into_bigint().to_bytes_le();
        le.resize(32, 0);
        bytes.extend_from_slice(&le);
    }
    hex_encode(&bytes)
}

fn fq6_hex_le(value: &Fq6) -> String {
    let coords = [
        &value.c0.c0, &value.c0.c1,
        &value.c1.c0, &value.c1.c1,
        &value.c2.c0, &value.c2.c1,
    ];
    let mut bytes = Vec::with_capacity(192);
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
    let g2_sizes = [1usize, 2, 4, 8, 16, 32];
    let max_g1 = *g1_sizes.iter().max().unwrap();
    let max_g2 = *g2_sizes.iter().max().unwrap();
    let max_n = std::cmp::max(max_g1, max_g2);

    let g1_bases = generate_g1_bases(max_g1);
    let g2_bases = generate_g2_bases(max_g2);

    let mut rng = StdRng::seed_from_u64(0x51ca_4f11);
    let fr_scalars: Vec<Fr> = (0..max_n).map(|_| Fr::rand(&mut rng)).collect();
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

fn generate_accumulator_ops(out_dir: &Path) {
    let mut rng = StdRng::seed_from_u64(0xacc0_0001);

    // --- sum_of_products: a0*b0 + a1*b1 ---
    let sop_cases: Vec<(Fr, Fr, Fr, Fr)> = vec![
        (Fr::from(0u64), Fr::from(1u64), Fr::from(0u64), Fr::from(1u64)),
        (Fr::from(1u64), Fr::from(1u64), Fr::from(1u64), Fr::from(1u64)),
        (Fr::from(3u64), Fr::from(5u64), Fr::from(7u64), Fr::from(11u64)),
        (Fr::rand(&mut rng), Fr::rand(&mut rng), Fr::rand(&mut rng), Fr::rand(&mut rng)),
        (Fr::rand(&mut rng), Fr::rand(&mut rng), Fr::rand(&mut rng), Fr::rand(&mut rng)),
        (Fr::rand(&mut rng), Fr::rand(&mut rng), Fr::rand(&mut rng), Fr::rand(&mut rng)),
    ];
    let mut sop_out = String::from("# name|a0_be_hex|b0_be_hex|a1_be_hex|b1_be_hex|expected_be_hex\n");
    for (i, (a0, b0, a1, b1)) in sop_cases.iter().enumerate() {
        let expected = *a0 * *b0 + *a1 * *b1;
        sop_out.push_str(&format!(
            "case_{i}|{}|{}|{}|{}|{}\n",
            hex_be(a0), hex_be(b0), hex_be(a1), hex_be(b1), hex_be(&expected)
        ));
    }
    write_file(out_dir.join("accumulator/sum_of_products.txt"), sop_out);

    // --- batch_inverse ---
    let mut bi_out = String::from("# name|count|inputs_be_hex_csv|expected_be_hex_csv\n");
    let batch_sizes = [1usize, 2, 4, 8];
    let mut bi_rng = StdRng::seed_from_u64(0xacc0_0002);
    for (i, &n) in batch_sizes.iter().enumerate() {
        let inputs: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut bi_rng)).collect();
        let outputs: Vec<Fr> = inputs.iter().map(|x| x.inverse().unwrap()).collect();
        bi_out.push_str(&format!(
            "case_{i}|{n}|{}|{}\n",
            join_field_scalars(&inputs),
            join_field_scalars(&outputs),
        ));
    }
    // Case with a zero element (zero inverse → zero)
    {
        let inputs = vec![Fr::rand(&mut bi_rng), Fr::from(0u64), Fr::rand(&mut bi_rng)];
        let outputs: Vec<Fr> = inputs
            .iter()
            .map(|x| {
                if x.is_zero() {
                    Fr::from(0u64)
                } else {
                    x.inverse().unwrap()
                }
            })
            .collect();
        bi_out.push_str(&format!(
            "case_with_zero|3|{}|{}\n",
            join_field_scalars(&inputs),
            join_field_scalars(&outputs),
        ));
    }
    write_file(out_dir.join("accumulator/batch_inverse.txt"), bi_out);

    // --- mul_u64: field * u64 scalar ---
    let mut mu64_rng = StdRng::seed_from_u64(0xacc0_0003);
    let mu64_cases: Vec<(Fr, u64)> = vec![
        (Fr::rand(&mut mu64_rng), 0),
        (Fr::rand(&mut mu64_rng), 1),
        (Fr::rand(&mut mu64_rng), 2),
        (Fr::rand(&mut mu64_rng), 12345),
        (Fr::rand(&mut mu64_rng), 0xdeadbeef_cafebabe),
        (Fr::rand(&mut mu64_rng), u64::MAX),
    ];
    let mut mu64_out = String::from("# name|field_be_hex|scalar_u64|expected_be_hex\n");
    for (i, (field_val, scalar)) in mu64_cases.iter().enumerate() {
        let expected = *field_val * Fr::from(*scalar);
        mu64_out.push_str(&format!(
            "case_{i}|{}|{scalar}|{}\n",
            hex_be(field_val),
            hex_be(&expected)
        ));
    }
    write_file(out_dir.join("accumulator/mul_u64.txt"), mu64_out);

    // --- mul_u128: field * u128 scalar ---
    let mut mu128_rng = StdRng::seed_from_u64(0xacc0_0004);
    let mu128_cases: Vec<(Fr, u128)> = vec![
        (Fr::rand(&mut mu128_rng), 0),
        (Fr::rand(&mut mu128_rng), 1),
        (Fr::rand(&mut mu128_rng), u64::MAX as u128),
        (Fr::rand(&mut mu128_rng), (u64::MAX as u128) + 1),
        (Fr::rand(&mut mu128_rng), 0xdeadbeef_cafebabe_12345678_9abcdef0),
        (Fr::rand(&mut mu128_rng), u128::MAX),
    ];
    let mut mu128_out = String::from("# name|field_be_hex|scalar_u128|expected_be_hex\n");
    for (i, (field_val, scalar)) in mu128_cases.iter().enumerate() {
        let expected = *field_val * Fr::from(*scalar);
        mu128_out.push_str(&format!(
            "case_{i}|{}|{scalar}|{}\n",
            hex_be(field_val),
            hex_be(&expected)
        ));
    }
    write_file(out_dir.join("accumulator/mul_u128.txt"), mu128_out);
}

fn generate_transcript_fixtures(out_dir: &Path) {
    use transcript::Blake2bTranscript;

    // Uses only the PUBLIC transcript API (init, appendLabel, appendU64,
    // appendScalar, challengeU128, challengeScalar128Bits) so that the Zig
    // differential verifier can exercise the same paths without accessing
    // private/raw methods.

    // --- State vectors ---
    // Each line: name|init_label|ops_desc|expected_state_hex|expected_rounds
    // ops_desc uses semicolons to separate steps, colons for args:
    //   append_label:data ; append_u64:count:999 ; append_scalar:val:7
    let mut state_out = String::from(
        "# name|init_label|ops_desc|expected_state_hex|expected_rounds\n",
    );

    let emit = |out: &mut String, name: &str, t: &Blake2bTranscript, init_label: &str, ops: &str| {
        out.push_str(&format!(
            "{name}|{init_label}|{ops}|{}|{}\n",
            hex_encode(t.state()),
            t.n_rounds()
        ));
    };

    // init only
    {
        let t = Blake2bTranscript::new(b"init_test");
        emit(&mut state_out, "init_basic", &t, "init_test", "-");
    }
    {
        let t = Blake2bTranscript::new(b"");
        emit(&mut state_out, "init_empty", &t, "", "-");
    }

    // appendLabel
    {
        let mut t = Blake2bTranscript::new(b"pub_test");
        t.append_label(b"data");
        emit(&mut state_out, "label_data", &t, "pub_test", "append_label:data");
    }
    {
        let mut t = Blake2bTranscript::new(b"zolt_test");
        t.append_label(b"hello");
        emit(&mut state_out, "label_hello", &t, "zolt_test", "append_label:hello");
    }

    // appendU64
    {
        let mut t = Blake2bTranscript::new(b"pub_test");
        t.append_u64(b"count", 999);
        emit(&mut state_out, "u64_999", &t, "pub_test", "append_u64:count:999");
    }
    {
        let mut t = Blake2bTranscript::new(b"pub_test");
        t.append_u64(b"size", 0);
        emit(&mut state_out, "u64_zero", &t, "pub_test", "append_u64:size:0");
    }

    // appendScalar
    {
        let mut t = Blake2bTranscript::new(b"pub_test");
        t.append_scalar(b"val", Fr::from(7u64));
        emit(&mut state_out, "scalar_7", &t, "pub_test", "append_scalar:val:7");
    }
    {
        let mut t = Blake2bTranscript::new(b"scalar_test");
        t.append_scalar(b"x", Fr::from(42u64));
        emit(&mut state_out, "scalar_42", &t, "scalar_test", "append_scalar:x:42");
    }
    {
        let mut t = Blake2bTranscript::new(b"scalar_test");
        t.append_scalar(b"x", Fr::from(0u64));
        emit(&mut state_out, "scalar_zero", &t, "scalar_test", "append_scalar:x:0");
    }

    // Multi-step (public API only)
    {
        let mut t = Blake2bTranscript::new(b"sequence");
        t.append_label(b"step1");
        t.append_scalar(b"x", Fr::from(100u64));
        t.append_u64(b"n", 42);
        emit(&mut state_out, "multi_pub_3ops", &t, "sequence",
            "append_label:step1;append_scalar:x:100;append_u64:n:42");
    }
    {
        let mut t = Blake2bTranscript::new(b"multi");
        t.append_label(b"round1");
        t.append_label(b"round2");
        t.append_scalar(b"r", Fr::from(999u64));
        emit(&mut state_out, "multi_labels_scalar", &t, "multi",
            "append_label:round1;append_label:round2;append_scalar:r:999");
    }

    write_file(out_dir.join("transcript/state_vectors.txt"), state_out);

    // --- Challenge vectors ---
    // Each line: name|init_label|ops_desc|expected_u128|expected_limb2_hex|expected_limb3_hex
    let mut chal_out = String::from(
        "# name|init_label|ops_desc|expected_u128|expected_limb2_hex|expected_limb3_hex\n",
    );

    // challenge after appendLabel
    {
        let mut t = Blake2bTranscript::new(b"chal_test");
        t.append_label(b"data");
        let val = t.challenge_u128();
        chal_out.push_str(&format!("u128_after_label|chal_test|append_label:data|{val}|-|-\n"));
    }
    {
        let mut t = Blake2bTranscript::new(b"chal_test");
        t.append_label(b"data");
        let (low, high) = t.challenge_scalar_128bits();
        chal_out.push_str(&format!(
            "scalar128_after_label|chal_test|append_label:data|-|{low:016x}|{high:016x}\n"
        ));
    }

    // challenge after appendScalar
    {
        let mut t = Blake2bTranscript::new(b"chal_test");
        t.append_scalar(b"input", Fr::from(42u64));
        let val = t.challenge_u128();
        chal_out.push_str(&format!(
            "u128_after_scalar|chal_test|append_scalar:input:42|{val}|-|-\n"
        ));
    }
    {
        let mut t = Blake2bTranscript::new(b"chal_test");
        t.append_scalar(b"input", Fr::from(42u64));
        let (low, high) = t.challenge_scalar_128bits();
        chal_out.push_str(&format!(
            "scalar128_after_scalar|chal_test|append_scalar:input:42|-|{low:016x}|{high:016x}\n"
        ));
    }

    // challenge after multi-step
    {
        let mut t = Blake2bTranscript::new(b"multi_chal");
        t.append_label(b"round1");
        t.append_u64(b"size", 256);
        let val = t.challenge_u128();
        chal_out.push_str(&format!(
            "u128_multi_step|multi_chal|append_label:round1;append_u64:size:256|{val}|-|-\n"
        ));
    }
    {
        let mut t = Blake2bTranscript::new(b"multi_chal");
        t.append_label(b"round1");
        t.append_u64(b"size", 256);
        let (low, high) = t.challenge_scalar_128bits();
        chal_out.push_str(&format!(
            "scalar128_multi_step|multi_chal|append_label:round1;append_u64:size:256|-|{low:016x}|{high:016x}\n"
        ));
    }

    write_file(out_dir.join("transcript/challenge_vectors.txt"), chal_out);
}

fn generate_extension_field_ops(out_dir: &Path) {
    let mut rng = StdRng::seed_from_u64(0xe47f_1e1d);

    // --- Fp2 ---
    let fp2_cases: Vec<(Fq2, Fq2)> = vec![
        (Fq2::zero(), Fq2::one()),
        (Fq2::one(), Fq2::one()),
        (Fq2::new(Fq::from(3u64), Fq::from(0u64)), Fq2::new(Fq::from(0u64), Fq::from(7u64))),
        (Fq2::new(Fq::from(5u64), Fq::from(11u64)), Fq2::new(Fq::from(13u64), Fq::from(17u64))),
        (Fq2::rand(&mut rng), Fq2::rand(&mut rng)),
        (Fq2::rand(&mut rng), Fq2::rand(&mut rng)),
        (Fq2::rand(&mut rng), Fq2::rand(&mut rng)),
        (Fq2::rand(&mut rng), Fq2::rand(&mut rng)),
    ];

    let mut fp2_out = String::from("# op|a_le_hex|b_le_hex|expected_le_hex\n");
    for (a, b) in &fp2_cases {
        fp2_out.push_str(&format!("add|{}|{}|{}\n", fq2_hex_le(a), fq2_hex_le(b), fq2_hex_le(&(*a + *b))));
        fp2_out.push_str(&format!("sub|{}|{}|{}\n", fq2_hex_le(a), fq2_hex_le(b), fq2_hex_le(&(*a - *b))));
        fp2_out.push_str(&format!("mul|{}|{}|{}\n", fq2_hex_le(a), fq2_hex_le(b), fq2_hex_le(&(*a * *b))));
        fp2_out.push_str(&format!("square|{}|-|{}\n", fq2_hex_le(a), fq2_hex_le(&a.square())));
        if !a.is_zero() {
            fp2_out.push_str(&format!("inv|{}|-|{}\n", fq2_hex_le(a), fq2_hex_le(&a.inverse().unwrap())));
        }
        {
            let mut conj = *a;
            conj.conjugate_in_place();
            fp2_out.push_str(&format!("conjugate|{}|-|{}\n", fq2_hex_le(a), fq2_hex_le(&conj)));
        }
    }
    write_file(out_dir.join("extensions/fp2_ops.txt"), fp2_out);

    // --- Fp6 ---
    let fp6_cases: Vec<(Fq6, Fq6)> = vec![
        (Fq6::zero(), Fq6::one()),
        (Fq6::one(), Fq6::one()),
        (Fq6::new(Fq2::new(Fq::from(1u64), Fq::from(2u64)),
                   Fq2::new(Fq::from(3u64), Fq::from(4u64)),
                   Fq2::new(Fq::from(5u64), Fq::from(6u64))),
         Fq6::new(Fq2::new(Fq::from(7u64), Fq::from(8u64)),
                   Fq2::new(Fq::from(9u64), Fq::from(10u64)),
                   Fq2::new(Fq::from(11u64), Fq::from(12u64)))),
        (Fq6::rand(&mut rng), Fq6::rand(&mut rng)),
        (Fq6::rand(&mut rng), Fq6::rand(&mut rng)),
        (Fq6::rand(&mut rng), Fq6::rand(&mut rng)),
        (Fq6::rand(&mut rng), Fq6::rand(&mut rng)),
        (Fq6::rand(&mut rng), Fq6::rand(&mut rng)),
    ];

    let mut fp6_out = String::from("# op|a_le_hex|b_le_hex|expected_le_hex\n");
    for (a, b) in &fp6_cases {
        fp6_out.push_str(&format!("add|{}|{}|{}\n", fq6_hex_le(a), fq6_hex_le(b), fq6_hex_le(&(*a + *b))));
        fp6_out.push_str(&format!("sub|{}|{}|{}\n", fq6_hex_le(a), fq6_hex_le(b), fq6_hex_le(&(*a - *b))));
        fp6_out.push_str(&format!("mul|{}|{}|{}\n", fq6_hex_le(a), fq6_hex_le(b), fq6_hex_le(&(*a * *b))));
        fp6_out.push_str(&format!("square|{}|-|{}\n", fq6_hex_le(a), fq6_hex_le(&a.square())));
        if !a.is_zero() {
            fp6_out.push_str(&format!("inv|{}|-|{}\n", fq6_hex_le(a), fq6_hex_le(&a.inverse().unwrap())));
        }
    }
    write_file(out_dir.join("extensions/fp6_ops.txt"), fp6_out);

    // --- Fp12 ---
    let fp12_cases: Vec<(Fq12, Fq12)> = vec![
        (Fq12::zero(), Fq12::one()),
        (Fq12::one(), Fq12::one()),
        (Fq12::new(
            Fq6::new(Fq2::new(Fq::from(1u64), Fq::from(2u64)),
                     Fq2::new(Fq::from(3u64), Fq::from(4u64)),
                     Fq2::new(Fq::from(5u64), Fq::from(6u64))),
            Fq6::new(Fq2::new(Fq::from(7u64), Fq::from(8u64)),
                     Fq2::new(Fq::from(9u64), Fq::from(10u64)),
                     Fq2::new(Fq::from(11u64), Fq::from(12u64)))),
         Fq12::new(
            Fq6::new(Fq2::new(Fq::from(13u64), Fq::from(14u64)),
                     Fq2::new(Fq::from(15u64), Fq::from(16u64)),
                     Fq2::new(Fq::from(17u64), Fq::from(18u64))),
            Fq6::new(Fq2::new(Fq::from(19u64), Fq::from(20u64)),
                     Fq2::new(Fq::from(21u64), Fq::from(22u64)),
                     Fq2::new(Fq::from(23u64), Fq::from(24u64))))),
        (Fq12::rand(&mut rng), Fq12::rand(&mut rng)),
        (Fq12::rand(&mut rng), Fq12::rand(&mut rng)),
        (Fq12::rand(&mut rng), Fq12::rand(&mut rng)),
        (Fq12::rand(&mut rng), Fq12::rand(&mut rng)),
        (Fq12::rand(&mut rng), Fq12::rand(&mut rng)),
    ];

    let mut fp12_out = String::from("# op|a_le_hex|b_le_hex|expected_le_hex\n");
    for (a, b) in &fp12_cases {
        fp12_out.push_str(&format!("add|{}|{}|{}\n", fq12_hex_le(a), fq12_hex_le(b), fq12_hex_le(&(*a + *b))));
        fp12_out.push_str(&format!("sub|{}|{}|{}\n", fq12_hex_le(a), fq12_hex_le(b), fq12_hex_le(&(*a - *b))));
        fp12_out.push_str(&format!("mul|{}|{}|{}\n", fq12_hex_le(a), fq12_hex_le(b), fq12_hex_le(&(*a * *b))));
        fp12_out.push_str(&format!("square|{}|-|{}\n", fq12_hex_le(a), fq12_hex_le(&a.square())));
        if !a.is_zero() {
            fp12_out.push_str(&format!("inv|{}|-|{}\n", fq12_hex_le(a), fq12_hex_le(&a.inverse().unwrap())));
        }
        {
            let mut conj = *a;
            conj.conjugate_in_place();
            fp12_out.push_str(&format!("conjugate|{}|-|{}\n", fq12_hex_le(a), fq12_hex_le(&conj)));
        }
        fp12_out.push_str(&format!("frobenius|{}|-|{}\n", fq12_hex_le(a), fq12_hex_le(&a.frobenius_map(1))));
        fp12_out.push_str(&format!("frobenius2|{}|-|{}\n", fq12_hex_le(a), fq12_hex_le(&a.frobenius_map(2))));
        fp12_out.push_str(&format!("frobenius3|{}|-|{}\n", fq12_hex_le(a), fq12_hex_le(&a.frobenius_map(3))));
        if !a.is_zero() {
            let mut cyc = *a;
            cyc.cyclotomic_square_in_place();
            fp12_out.push_str(&format!("cyclotomic_square|{}|-|{}\n", fq12_hex_le(a), fq12_hex_le(&cyc)));
        }
    }
    write_file(out_dir.join("extensions/fp12_ops.txt"), fp12_out);
}

fn generate_g2_ops(out_dir: &Path) {
    let mut rng = StdRng::seed_from_u64(0x6200_0001);
    let gen = G2Affine::generator();

    let mut out = String::from(
        "# op|arg1_be_hex|arg2_be_hex|expected_infinity|expected_x_c0_le|expected_x_c1_le|expected_y_c0_le|expected_y_c1_le\n",
    );

    // scalar_mul cases
    let scalars: Vec<Fr> = vec![
        Fr::from(0u64),
        Fr::from(1u64),
        Fr::from(2u64),
        Fr::from(3u64),
        Fr::from(42u64),
        Fr::from(0xdeadbeef12345678u64),
        Fr::rand(&mut rng),
        Fr::rand(&mut rng),
        Fr::rand(&mut rng),
    ];
    for s in &scalars {
        let result = G2Projective::from(gen)
            .mul_bigint(s.into_bigint())
            .into_affine();
        let (inf, x0, x1, y0, y1) = format_g2_point(result);
        out.push_str(&format!(
            "scalar_mul|{}|-|{inf}|{x0}|{x1}|{y0}|{y1}\n",
            hex_be(s)
        ));
    }

    // add cases
    let add_pairs: Vec<(Fr, Fr)> = vec![
        (Fr::from(1u64), Fr::from(1u64)),
        (Fr::from(3u64), Fr::from(5u64)),
        (Fr::rand(&mut rng), Fr::rand(&mut rng)),
    ];
    for (s1, s2) in &add_pairs {
        let p1 = G2Projective::from(gen).mul_bigint(s1.into_bigint());
        let p2 = G2Projective::from(gen).mul_bigint(s2.into_bigint());
        let result = (p1 + p2).into_affine();
        let (inf, x0, x1, y0, y1) = format_g2_point(result);
        out.push_str(&format!(
            "add|{}|{}|{inf}|{x0}|{x1}|{y0}|{y1}\n",
            hex_be(s1),
            hex_be(s2)
        ));
    }

    // double cases
    let double_scalars = [Fr::from(1u64), Fr::from(42u64)];
    for s in &double_scalars {
        let p = G2Projective::from(gen).mul_bigint(s.into_bigint());
        let result = p.double().into_affine();
        let (inf, x0, x1, y0, y1) = format_g2_point(result);
        out.push_str(&format!(
            "double|{}|-|{inf}|{x0}|{x1}|{y0}|{y1}\n",
            hex_be(s)
        ));
    }

    // neg cases
    let neg_scalars = [Fr::from(1u64), Fr::from(42u64)];
    for s in &neg_scalars {
        let p = G2Projective::from(gen)
            .mul_bigint(s.into_bigint())
            .into_affine();
        let result = -p;
        let (inf, x0, x1, y0, y1) = format_g2_point(result);
        out.push_str(&format!(
            "neg|{}|-|{inf}|{x0}|{x1}|{y0}|{y1}\n",
            hex_be(s)
        ));
    }

    write_file(out_dir.join("g2/g2_ops.txt"), out);
}

fn generate_point_compression(out_dir: &Path) {
    let mut rng = StdRng::seed_from_u64(0xc04d_0001);
    let g1_gen = G1Affine::generator();
    let g2_gen = G2Affine::generator();

    // --- G1 ---
    let mut g1_out = String::from(
        "# name|uncompressed_x_le|uncompressed_y_le|compressed_hex\n",
    );

    let g1_scalars: Vec<(String, Fr)> = vec![
        ("identity".into(), Fr::from(0u64)),
        ("generator".into(), Fr::from(1u64)),
        ("double".into(), Fr::from(2u64)),
        ("scalar_42".into(), Fr::from(42u64)),
        ("random_0".into(), Fr::rand(&mut rng)),
        ("random_1".into(), Fr::rand(&mut rng)),
        ("random_2".into(), Fr::rand(&mut rng)),
        ("random_3".into(), Fr::rand(&mut rng)),
    ];

    for (name, s) in &g1_scalars {
        let point = G1Projective::from(g1_gen)
            .mul_bigint(s.into_bigint())
            .into_affine();
        let mut compressed = Vec::new();
        point.serialize_compressed(&mut compressed).unwrap();
        let (_, x, y) = format_g1_point(point);
        g1_out.push_str(&format!(
            "{name}|{x}|{y}|{}\n",
            hex_encode(&compressed)
        ));
    }

    write_file(out_dir.join("point_compression/g1_compress.txt"), g1_out);

    // --- G2 ---
    let mut g2_out = String::from(
        "# name|x_c0_le|x_c1_le|y_c0_le|y_c1_le|compressed_hex\n",
    );

    let g2_scalars: Vec<(String, Fr)> = vec![
        ("identity".into(), Fr::from(0u64)),
        ("generator".into(), Fr::from(1u64)),
        ("double".into(), Fr::from(2u64)),
        ("scalar_42".into(), Fr::from(42u64)),
        ("random_0".into(), Fr::rand(&mut rng)),
        ("random_1".into(), Fr::rand(&mut rng)),
        ("random_2".into(), Fr::rand(&mut rng)),
        ("random_3".into(), Fr::rand(&mut rng)),
    ];

    for (name, s) in &g2_scalars {
        let point = G2Projective::from(g2_gen)
            .mul_bigint(s.into_bigint())
            .into_affine();
        let mut compressed = Vec::new();
        point.serialize_compressed(&mut compressed).unwrap();
        let (_, x0, x1, y0, y1) = format_g2_point(point);
        g2_out.push_str(&format!(
            "{name}|{x0}|{x1}|{y0}|{y1}|{}\n",
            hex_encode(&compressed)
        ));
    }

    write_file(out_dir.join("point_compression/g2_compress.txt"), g2_out);
}

fn generate_glv_fixtures(out_dir: &Path) {
    let mut rng = StdRng::seed_from_u64(0x614f_0001);
    let g1_gen = G1Affine::generator();
    let g2_gen = G2Affine::generator();

    let scalars: Vec<(String, Fr)> = vec![
        ("zero".into(), Fr::from(0u64)),
        ("one".into(), Fr::from(1u64)),
        ("two".into(), Fr::from(2u64)),
        ("small_42".into(), Fr::from(42u64)),
        ("medium".into(), Fr::from(0xdeadbeef12345678u64)),
        ("random_0".into(), Fr::rand(&mut rng)),
        ("random_1".into(), Fr::rand(&mut rng)),
        ("random_2".into(), Fr::rand(&mut rng)),
        ("random_3".into(), Fr::rand(&mut rng)),
        ("random_4".into(), Fr::rand(&mut rng)),
    ];

    // G1 scalar mul
    let mut g1_out = String::from(
        "# name|scalar_be_hex|expected_infinity|expected_x_le|expected_y_le\n",
    );
    for (name, s) in &scalars {
        let result = G1Projective::from(g1_gen)
            .mul_bigint(s.into_bigint())
            .into_affine();
        let (inf, x, y) = format_g1_point(result);
        g1_out.push_str(&format!("{name}|{}|{inf}|{x}|{y}\n", hex_be(s)));
    }
    write_file(out_dir.join("glv/glv_g1_scalar_mul.txt"), g1_out);

    // G2 scalar mul
    let mut g2_out = String::from(
        "# name|scalar_be_hex|expected_infinity|expected_x_c0_le|expected_x_c1_le|expected_y_c0_le|expected_y_c1_le\n",
    );
    for (name, s) in &scalars {
        let result = G2Projective::from(g2_gen)
            .mul_bigint(s.into_bigint())
            .into_affine();
        let (inf, x0, x1, y0, y1) = format_g2_point(result);
        g2_out.push_str(&format!(
            "{name}|{}|{inf}|{x0}|{x1}|{y0}|{y1}\n",
            hex_be(s)
        ));
    }
    write_file(out_dir.join("glv/glv_g2_scalar_mul.txt"), g2_out);
}

fn generate_dory_commit(out_dir: &Path) {
    // Replicate Zolt's exact SRS generation:
    //   seed = SHA3-256("Jolt Dory URS seed")
    //   G1[i] = [SHA3-256(seed || le_u64(i) || "G1") as Fr] * G1::generator
    //   G2[i] = [SHA3-256(seed || le_u64(i+n) || "G2") as Fr] * G2::generator
    //   h1 = G1::generator, h2 = G2::generator

    let mut hasher = Sha3_256::new();
    hasher.update(b"Jolt Dory URS seed");
    let seed: [u8; 32] = hasher.finalize().into();

    let g1_gen = G1Affine::generator();
    let g2_gen = G2Affine::generator();

    let mut out = String::from(
        "# name|max_num_vars|evals_be_csv|expected_commitment_fp12_le_hex\n",
    );

    for max_num_vars in [2u64, 3, 4] {
        let sigma = (max_num_vars + 1) / 2;
        let nu = max_num_vars - sigma;
        let num_cols = 1usize << sigma;
        let num_rows = 1usize << nu;
        let n = std::cmp::max(num_cols, num_rows);

        // Generate SRS points matching Zolt's hash-to-curve
        let g1_vec: Vec<G1Affine> = (0..n)
            .map(|i| {
                let mut h = Sha3_256::new();
                h.update(&seed);
                h.update(&(i as u64).to_le_bytes());
                h.update(b"G1");
                let hash: [u8; 32] = h.finalize().into();
                let scalar = Fr::from_le_bytes_mod_order(&hash);
                G1Projective::from(g1_gen)
                    .mul_bigint(scalar.into_bigint())
                    .into_affine()
            })
            .collect();

        let g2_vec: Vec<G2Affine> = (0..n)
            .map(|i| {
                let mut h = Sha3_256::new();
                h.update(&seed);
                h.update(&((i + n) as u64).to_le_bytes());
                h.update(b"G2");
                let hash: [u8; 32] = h.finalize().into();
                let scalar = Fr::from_le_bytes_mod_order(&hash);
                G2Projective::from(g2_gen)
                    .mul_bigint(scalar.into_bigint())
                    .into_affine()
            })
            .collect();

        // Known polynomial: [1, 2, ..., 2^max_num_vars]
        let poly_len = 1usize << max_num_vars;
        let evals: Vec<Fr> = (1..=poly_len as u64).map(Fr::from).collect();

        // Compute Dory commitment: for each row, MSM(g1_vec, row_evals),
        // then multi-pairing(row_commits, g2_vec), final exponentiation
        let mut commitment = Fq12::one();
        for row in 0..num_rows {
            let row_start = row * num_cols;
            let row_end = std::cmp::min(row_start + num_cols, evals.len());
            let row_evals = &evals[row_start..row_end];
            let row_commit = G1Projective::msm(&g1_vec[..row_evals.len()], row_evals)
                .unwrap()
                .into_affine();
            if !row_commit.infinity && row < g2_vec.len() {
                let pairing_result = Bn254::pairing(row_commit, g2_vec[row]).0;
                commitment *= pairing_result;
            }
        }

        let evals_csv = evals.iter().map(hex_be).collect::<Vec<_>>().join(",");
        out.push_str(&format!(
            "case_nv{max_num_vars}|{max_num_vars}|{evals_csv}|{}\n",
            fq12_hex_le(&commitment)
        ));
    }

    write_file(out_dir.join("dory/commit_cases.txt"), out);
}

fn generate_gpu_crossover(out_dir: &Path) {
    let mut rng = StdRng::seed_from_u64(0x6900_0001);

    let mut out = String::from(
        "# op|size|a_be_hex_csv|b_be_hex_csv|expected_be_hex_csv\n",
    );

    let sizes = [4usize, 16, 64];
    for &size in &sizes {
        let a: Vec<Fr> = (0..size).map(|_| Fr::rand(&mut rng)).collect();
        let b: Vec<Fr> = (0..size).map(|_| Fr::rand(&mut rng)).collect();

        // mul
        let mul_expected: Vec<Fr> = a.iter().zip(b.iter()).map(|(x, y)| *x * *y).collect();
        out.push_str(&format!(
            "mul|{size}|{}|{}|{}\n",
            join_field_scalars(&a),
            join_field_scalars(&b),
            join_field_scalars(&mul_expected)
        ));

        // add
        let add_expected: Vec<Fr> = a.iter().zip(b.iter()).map(|(x, y)| *x + *y).collect();
        out.push_str(&format!(
            "add|{size}|{}|{}|{}\n",
            join_field_scalars(&a),
            join_field_scalars(&b),
            join_field_scalars(&add_expected)
        ));

        // sub
        let sub_expected: Vec<Fr> = a.iter().zip(b.iter()).map(|(x, y)| *x - *y).collect();
        out.push_str(&format!(
            "sub|{size}|{}|{}|{}\n",
            join_field_scalars(&a),
            join_field_scalars(&b),
            join_field_scalars(&sub_expected)
        ));

        // neg
        let neg_expected: Vec<Fr> = a.iter().map(|x| -*x).collect();
        out.push_str(&format!(
            "neg|{size}|{}|-|{}\n",
            join_field_scalars(&a),
            join_field_scalars(&neg_expected)
        ));
    }

    write_file(out_dir.join("gpu/field_crossover.txt"), out);
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
    generate_accumulator_ops(&out_dir);
    generate_transcript_fixtures(&out_dir);
    generate_extension_field_ops(&out_dir);
    generate_g2_ops(&out_dir);
    generate_point_compression(&out_dir);
    generate_glv_fixtures(&out_dir);
    generate_dory_commit(&out_dir);
    generate_gpu_crossover(&out_dir);
    eprintln!("generated zolt-arith differential fixtures under {}", out_dir.display());
}
