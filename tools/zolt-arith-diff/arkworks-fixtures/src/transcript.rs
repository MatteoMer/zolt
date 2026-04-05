//! Blake2b-based Fiat-Shamir transcript matching Jolt's Blake2bTranscript.
//!
//! This is an independent Rust reimplementation used as a differential oracle.
//! The protocol is defined in packages/zolt-arith/src/transcripts/blake2b.zig.

use ark_bn254::Fr;
use ark_ff::{BigInteger, PrimeField};
use blake2::{Blake2b, Digest};
use blake2::digest::consts::U32;

type Blake2b256 = Blake2b<U32>;

/// Blake2b-based Fiat-Shamir transcript matching Jolt (upstream 97b2c96).
pub struct Blake2bTranscript {
    state: [u8; 32],
    n_rounds: u32,
}

impl Blake2bTranscript {
    pub fn new(label: &[u8]) -> Self {
        assert!(label.len() < 33);
        let mut padded = [0u8; 32];
        let copy_len = label.len().min(32);
        padded[..copy_len].copy_from_slice(&label[..copy_len]);

        let mut h = Blake2b256::new();
        h.update(&padded);
        let result = h.finalize();
        let mut state = [0u8; 32];
        state.copy_from_slice(&result);

        Self { state, n_rounds: 0 }
    }

    fn hasher(&self) -> Blake2b256 {
        let mut h = Blake2b256::new();
        h.update(&self.state);
        let mut round_data = [0u8; 32];
        round_data[28..32].copy_from_slice(&self.n_rounds.to_be_bytes());
        h.update(&round_data);
        h
    }

    fn update_state(&mut self, new_state: [u8; 32]) {
        self.state = new_state;
        self.n_rounds += 1;
    }

    fn finalize_to_state(h: Blake2b256) -> [u8; 32] {
        let result = h.finalize();
        let mut state = [0u8; 32];
        state.copy_from_slice(&result);
        state
    }

    // ===================== Raw methods =====================

    pub fn raw_append_label(&mut self, label: &[u8]) {
        assert!(label.len() < 33);
        let mut h = self.hasher();
        if label.len() == 32 {
            h.update(label);
        } else {
            let mut padded = [0u8; 32];
            padded[..label.len()].copy_from_slice(label);
            h.update(&padded);
        }
        let new_state = Self::finalize_to_state(h);
        self.update_state(new_state);
    }

    fn raw_append_label_with_len(&mut self, label: &[u8], len: u64) {
        assert!(label.len() <= 24);
        let mut label_buf = [0u8; 32];
        label_buf[..label.len()].copy_from_slice(label);
        label_buf[24..32].copy_from_slice(&len.to_be_bytes());
        self.raw_append_bytes(&label_buf);
    }

    pub fn raw_append_bytes(&mut self, bytes: &[u8]) {
        let mut h = self.hasher();
        h.update(bytes);
        let new_state = Self::finalize_to_state(h);
        self.update_state(new_state);
    }

    pub fn raw_append_u64(&mut self, x: u64) {
        let mut data = [0u8; 32];
        data[24..32].copy_from_slice(&x.to_be_bytes());
        let mut h = self.hasher();
        h.update(&data);
        let new_state = Self::finalize_to_state(h);
        self.update_state(new_state);
    }

    pub fn raw_append_scalar(&mut self, scalar: Fr) {
        let bigint = scalar.into_bigint();
        let le_bytes = bigint.to_bytes_le();
        let mut reversed = [0u8; 32];
        for i in 0..32 {
            reversed[i] = le_bytes[31 - i];
        }
        self.raw_append_bytes(&reversed);
    }

    // ===================== Public API =====================

    pub fn append_label(&mut self, label: &[u8]) {
        self.raw_append_label(label);
    }

    pub fn append_bytes(&mut self, label: &[u8], bytes: &[u8]) {
        self.raw_append_label_with_len(label, bytes.len() as u64);
        self.raw_append_bytes(bytes);
    }

    pub fn append_u64(&mut self, label: &[u8], x: u64) {
        self.raw_append_label(label);
        self.raw_append_u64(x);
    }

    pub fn append_scalar(&mut self, label: &[u8], scalar: Fr) {
        self.raw_append_label(label);
        self.raw_append_scalar(scalar);
    }

    pub fn append_scalars(&mut self, label: &[u8], scalars: &[Fr]) {
        self.raw_append_label_with_len(label, scalars.len() as u64);
        for scalar in scalars {
            self.raw_append_scalar(*scalar);
        }
    }

    // ===================== Challenge generation =====================

    fn challenge_bytes_32(&mut self) -> [u8; 32] {
        let h = self.hasher();
        let out = Self::finalize_to_state(h);
        self.update_state(out);
        out
    }

    pub fn challenge_bytes(&mut self, len: usize) -> Vec<u8> {
        let mut out = vec![0u8; len];
        let mut remaining = len;
        let mut start = 0;
        while remaining > 32 {
            let full_rand = self.challenge_bytes_32();
            out[start..start + 32].copy_from_slice(&full_rand);
            start += 32;
            remaining -= 32;
        }
        let full_rand = self.challenge_bytes_32();
        out[start..start + remaining].copy_from_slice(&full_rand[..remaining]);
        out
    }

    pub fn challenge_u128(&mut self) -> u128 {
        let buf = self.challenge_bytes(16);
        u128::from_le_bytes(buf[..16].try_into().unwrap())
    }

    /// 128-bit challenge masked to 125 bits, stored as [0, 0, low, high] Montgomery limbs.
    /// Returns (low, high) u64 pair.
    pub fn challenge_scalar_128bits(&mut self) -> (u64, u64) {
        let buf = self.challenge_bytes(16);
        let full_value = u128::from_le_bytes(buf[..16].try_into().unwrap());
        let mask_125: u128 = (1u128 << 125) - 1;
        let masked = full_value & mask_125;
        let low = masked as u64;
        let high = (masked >> 64) as u64;
        (low, high)
    }

    // ===================== Accessors =====================

    pub fn state(&self) -> &[u8; 32] {
        &self.state
    }

    pub fn n_rounds(&self) -> u32 {
        self.n_rounds
    }
}
