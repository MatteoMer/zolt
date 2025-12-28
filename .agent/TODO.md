# Zolt zkVM Implementation TODO

## Current Status (Jolt Compatibility Phase)

**Project Status: DORY COMMITMENT ALGORITHM MISMATCH**

The SRS is now loadable from Jolt-exported files, but the commitment computation
algorithm itself differs between Zolt and Jolt's dory-pcs crate.

Key achievements:
1. ✅ JoltProofWithDory bundle ensures consistent commitments
2. ✅ SRS loading from Jolt-exported file
3. ✅ --srs CLI option added
4. ✅ All 608 Zolt tests pass
5. ✅ Cross-deserialization works

Remaining issue: **Dory commitment algorithm differs**
- Jolt uses `Polynomial::commit` from dory-pcs with specific nu/sigma params
- Jolt uses `DoryGlobals` to track current matrix dimensions
- Zolt's simple MSM + pairing approach produces different GT values
- Even with same SRS points, the commitments differ

---

## Understanding Jolt's Dory Commit

Jolt's commit in `commitment_scheme.rs`:
```rust
fn commit(poly, setup) {
    let num_cols = DoryGlobals::get_num_columns();
    let num_rows = DoryGlobals::get_max_num_rows();
    let sigma = num_cols.log_2();
    let nu = num_rows.log_2();

    let (tier_2, row_commitments) = Polynomial::commit::<BN254, JoltG1Routines>(
        poly, nu, sigma, setup
    );
    (tier_2, row_commitments)
}
```

Key differences from Zolt:
1. **DoryGlobals**: Global state tracks current matrix dimensions
2. **nu/sigma parameters**: Log2 of rows/columns, used in the algorithm
3. **JoltG1Routines**: Custom G1 routines for the commitment
4. **tier_2 commitment**: Uses a specific multi-tier commitment structure

---

## Next Steps

1. [ ] Study dory-pcs `Polynomial::commit` implementation
2. [ ] Port exact algorithm including tier structure
3. [ ] Match DoryGlobals matrix dimension management
4. [ ] Consider alternative: generate proof in Jolt, convert to Zolt

---

## Major Milestones Achieved

1. ✅ Blake2b Transcript
2. ✅ JoltProof 7-stage structure
3. ✅ Arkworks serialization
4. ✅ GT element serialization
5. ✅ Cross-deserialization
6. ✅ UniSkip infrastructure
7. ✅ 48 opening claims
8. ✅ 19 R1CS constraints
9. ✅ JoltDevice support
10. ✅ Fiat-Shamir preamble
11. ✅ Byte reversal for transcript
12. ✅ GT elements in transcript
13. ✅ JoltProofWithDory bundle
14. ✅ SRS loading from file
15. ✅ --srs CLI option

---

## Test Status

### Zolt: 608/608 tests passing

### Jolt Cross-Verification

| Test | Status | Details |
|------|--------|---------|
| `test_deserialize_zolt_proof` | ✅ PASS | Deserializes correctly |
| `test_debug_zolt_format` | ✅ PASS | All claims valid |
| `test_verify_zolt_proof` | ❌ FAIL | UniSkip fails (commitment algorithm mismatch) |

---

## Summary

**Serialization: COMPLETE**
**Transcript: COMPLETE (using GT elements with reversal)**
**SRS: COMPLETE (can load from Jolt-exported file)**
**Commitment Algorithm: NEEDS PORTING**

The remaining blocker is porting the exact Dory commitment algorithm from
the dory-pcs crate to Zig. This includes:
- The tier structure (tier_2 commitment)
- Matrix dimension management (nu/sigma)
- JoltG1Routines integration

This is a complex port requiring deep understanding of the dory-pcs algorithm.
