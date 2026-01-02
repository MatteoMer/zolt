# Zolt-Jolt Compatibility TODO

## Current Status: Session 34 - January 2, 2026

**All 702 Zolt tests pass**

### Root Cause Identified: Architectural Mismatch

After comprehensive research comparing Jolt's `outer.rs` with Zolt's `streaming_outer.zig`, the issue has been identified as a **fundamental architectural mismatch** in the linear phase of the outer sumcheck.

**Key Finding**: Jolt separates streaming and linear phases into distinct structures with clean trait implementations. Zolt tries to handle both in one monolithic struct with conditional code paths, leading to:

1. **t_prime_poly not being bound** after each round
2. **t_prime_poly not being rebuilt** from bound Az/Bz polynomials in linear rounds
3. **Missing E_active projection** for correct t_prime evaluation

---

## Jolt's Exact Architecture

### Struct Definitions (from outer.rs)

#### OuterSharedState (lines 518-531)
```rust
pub struct OuterSharedState<F: JoltField> {
    bytecode_preprocessing: BytecodePreprocessing,
    trace: Arc<Vec<Cycle>>,
    split_eq_poly: GruenSplitEqPolynomial<F>,
    t_prime_poly: Option<MultiquadraticPolynomial<F>>,
    r_grid: ExpandingTable<F>,
    params: OuterStreamingProverParams<F>,
    lagrange_evals_r0: [F; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE],  // Size 10
}
```

#### OuterLinearStage (lines 847-851)
```rust
pub struct OuterLinearStage<F: JoltField> {
    az: DensePolynomial<F>,
    bz: DensePolynomial<F>,
}
```

#### OuterStreamingWindow (lines 805-809)
```rust
pub struct OuterStreamingWindow<F: JoltField> {
    _phantom: PhantomData<F>,  // Stateless! All state in shared
}
```

#### OuterStreamingProverParams (lines 476-479)
```rust
struct OuterStreamingProverParams<F: JoltField> {
    num_cycles_bits: usize,
    r0_uniskip: F::Challenge,  // UniSkip challenge, used for lagrange_evals_r0
}
```

### Trait Definitions (from streaming_sumcheck.rs)

#### StreamingSumcheckWindow trait (lines 12-25)
```rust
pub trait StreamingSumcheckWindow<F: JoltField> {
    type Shared;
    fn initialize(shared: &mut Self::Shared, window_size: usize) -> Self;
    fn compute_message(&self, shared: &Self::Shared, window_size: usize, previous_claim: F) -> UniPoly<F>;
    fn ingest_challenge(&mut self, shared: &mut Self::Shared, r: F::Challenge, round: usize);
}
```

#### LinearSumcheckStage trait (lines 27-55)
```rust
pub trait LinearSumcheckStage<F: JoltField> {
    type Shared;
    type Streaming: StreamingSumcheckWindow<F>;

    fn initialize(streaming: Option<Self::Streaming>, shared: &mut Self::Shared, window_size: usize) -> Self;
    fn next_window(&mut self, shared: &mut Self::Shared, window_size: usize);
    fn compute_message(&self, shared: &Self::Shared, window_size: usize, previous_claim: F) -> UniPoly<F>;
    fn ingest_challenge(&mut self, shared: &mut Self::Shared, r: F::Challenge, round: usize);
    fn cache_openings<T: Transcript>(&self, shared: &Self::Shared, accumulator: &mut ProverOpeningAccumulator<F>, transcript: &mut T, sumcheck_challenges: &[F::Challenge]);
}
```

#### SharedStreamingSumcheckState trait (lines 57-63)
```rust
pub trait SharedStreamingSumcheckState<F: JoltField> {
    fn degree(&self) -> usize;
    fn num_rounds(&self) -> usize;
    fn input_claim(&self, accumulator: &ProverOpeningAccumulator<F>) -> F;
}
```

### StreamingSumcheck Orchestrator (lines 65-96)
```rust
pub struct StreamingSumcheck<F, S, Shared, Streaming, Linear> {
    streaming: Option<Streaming>,
    linear: Option<Linear>,
    shared: Shared,
    schedule: S,
}
```

---

## Jolt's Round Flow

### Overall Protocol Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 1 Outer Sumcheck                       │
├─────────────────────────────────────────────────────────────────┤
│  1. UniSkip First Round (degree 27 polynomial)                  │
│     └── Produces r0 challenge                                   │
│                                                                 │
│  2. Remaining Sumcheck (degree 3 polynomials)                   │
│     └── Uses LinearOnlySchedule with switch_over = 0            │
│     └── All rounds use OuterLinearStage                         │
└─────────────────────────────────────────────────────────────────┘
```

### UniSkip Integration

```rust
// In prover.rs prove_stage1():
// 1. Create and prove UniSkip first round
let first_round_proof = prove_uniskip_round(&mut uni_skip, ...);

// 2. Create shared state (extracts r0 from opening accumulator)
let shared = OuterSharedState::new(trace, bytecode, &uni_skip_params, &opening_accumulator);

// 3. Create streaming sumcheck with LinearOnlySchedule
let schedule = LinearOnlySchedule::new(tau.len() - 1);
let mut spartan_outer = StreamingSumcheck::new(shared, schedule);

// 4. Prove remaining rounds
let sumcheck_proof = BatchedSumcheck::prove(vec![&mut spartan_outer], ...);
```

### How r0 Is Used

```rust
// In OuterSharedState::new():
let r0 = params.r0_uniskip;

// 1. Compute Lagrange basis evaluations at r0
let lagrange_evals_r0 = LagrangePolynomial::evals::<DOMAIN_SIZE>(&r0);

// 2. Compute L(tau_high, r0) as scaling factor for split_eq
let lagrange_tau_r0 = LagrangePolynomial::lagrange_kernel(&r0, &tau_high);

// 3. Initialize split_eq with this scaling factor
let split_eq_poly = GruenSplitEqPolynomial::new_with_scaling(tau_low, Some(lagrange_tau_r0));
```

### Remaining Sumcheck Flow (LinearOnlySchedule with switch_over = 0)

```
compute_message(round=0):
├── round.cmp(&switch_over) == Equal
├── linear = Linear::initialize(streaming.take(), shared, window_size=1)
│   ├── Calls fused_materialise_polynomials_round_zero(shared, 1)
│   │   ├── Iterates over E_out × E_in × window grid
│   │   ├── Computes Az, Bz from trace using lagrange_evals_r0
│   │   ├── Stores in self.az, self.bz DensePolynomials
│   │   └── Builds t_prime_poly from Az*Bz products
│   └── Returns OuterLinearStage { az, bz }
└── linear.compute_message(shared, window_size=1, previous_claim)
    ├── (t_prime_0, t_prime_inf) = shared.compute_t_evals(1)
    │   └── Projects t_prime_poly using E_active weights
    └── Returns gruen_poly_deg_3(t_prime_0, t_prime_inf, previous_claim)

ingest_challenge(round=0, r):
├── shared.split_eq_poly.bind(r)
├── shared.t_prime_poly.bind(r, LowToHigh)  // CRITICAL!
└── rayon::join(az.bind_parallel(r), bz.bind_parallel(r))

compute_message(round=1+):
├── round.cmp(&switch_over) == Greater
├── linear.next_window(shared, window_size=1)
│   └── compute_evaluation_grid_from_polynomials_parallel(shared, 1)
│       ├── Reads from bound self.az, self.bz (NO trace access!)
│       ├── Expands to multiquadratic grid
│       └── Rebuilds shared.t_prime_poly
└── linear.compute_message(shared, 1, previous_claim)
    └── Same as round 0

ingest_challenge(round=1+, r):
└── Same as round 0
```

### Critical Difference: Streaming vs Linear ingest_challenge

```rust
// OuterStreamingWindow::ingest_challenge (not used with LinearOnlySchedule)
fn ingest_challenge(&mut self, shared: &mut Self::Shared, r_j: F::Challenge, _round: usize) {
    shared.split_eq_poly.bind(r_j);
    shared.t_prime_poly.bind(r_j, LowToHigh);
    shared.r_grid.update(r_j);  // <-- Updates r_grid!
}

// OuterLinearStage::ingest_challenge
fn ingest_challenge(&mut self, shared: &mut Self::Shared, r_j: F::Challenge, _round: usize) {
    shared.split_eq_poly.bind(r_j);
    shared.t_prime_poly.bind(r_j, LowToHigh);
    // NO r_grid.update() - instead bind the materialized polynomials:
    self.az.bind_parallel(r_j, LowToHigh);
    self.bz.bind_parallel(r_j, LowToHigh);
}
```

---

## What Zolt Does Wrong

1. **t_prime_poly is NOT bound** after each round
2. **t_prime_poly is NOT rebuilt** from bound polynomials in `next_window`
3. **MultiquadraticPolynomial lacks `bind()` method**
4. **`E_active_for_window()` is missing** for correct projection
5. **Mixes streaming and linear logic** in conditional code paths

---

## Implementation Plan

### Phase 1: Add Missing Primitives

#### 1.1 Add `MultiquadraticPolynomial.bind()`
**File**: `src/poly/multiquadratic.zig`

```zig
/// Bind first variable to challenge r (LowToHigh order)
///
/// For multilinear on ternary grid:
///   new[i] = old[3*i+0] + r * old[3*i+2]
/// where old[3*i+2] is the slope f(1)-f(0) stored at ∞ position.
pub fn bind(self: *Self, r: F) void {
    if (self.num_vars == 0) return;

    const new_size = pow3(self.num_vars - 1);

    for (0..new_size) |i| {
        const f_0 = self.evaluations[3 * i + 0];
        const slope = self.evaluations[3 * i + 2];
        self.evaluations[i] = f_0.add(r.mul(slope));
    }

    self.num_vars -= 1;
}
```

#### 1.2 Add `GruenSplitEqPolynomial.getEActiveForWindow()`
**File**: `src/poly/split_eq.zig`

```zig
/// Get E_active for projecting t_prime in current window
/// Matches Jolt's E_active_for_window()
pub fn getEActiveForWindow(self: *const Self, window_size: usize) []const F {
    const num_unbound = self.current_index;
    const actual_window = @min(window_size, num_unbound);
    const head_len = num_unbound -| actual_window;

    const m = self.tau.len / 2;
    const head_in_bits = head_len -| @min(head_len, m);

    return self.E_in_vec.items[head_in_bits];
}
```

### Phase 2: Restructure streaming_outer.zig

#### 2.1 Create `OuterSharedState` struct (matches Jolt exactly)

```zig
const OuterSharedState = struct {
    const Self = @This();

    // Matches Jolt's OuterSharedState fields
    split_eq_poly: split_eq.GruenSplitEqPolynomial(F),
    t_prime_poly: ?multiquadratic.MultiquadraticPolynomial(F),
    r_grid: expanding_table.ExpandingTable(F),
    lagrange_evals_r0: [DOMAIN_SIZE]F,
    cycle_witnesses: []const constraints.R1CSCycleInputs(F),
    tau_high: F,
    r0: F,  // UniSkip challenge
    allocator: Allocator,

    pub fn init(
        allocator: Allocator,
        tau: []const F,
        r0: F,
        cycle_witnesses: []const constraints.R1CSCycleInputs(F),
    ) !Self { ... }

    pub fn deinit(self: *Self) void { ... }

    /// Compute t'(0) and t'(∞) from t_prime_poly
    /// Matches Jolt's compute_t_evals()
    pub fn computeTevals(self: *const Self, window_size: usize) struct { t_zero: F, t_inf: F } {
        const t_prime = self.t_prime_poly.?;
        const E_active = self.split_eq_poly.getEActiveForWindow(window_size);
        return t_prime.projectToFirstVariable(E_active);
    }

    /// Number of rounds in the remaining sumcheck
    pub fn numRounds(self: *const Self) usize {
        return self.split_eq_poly.current_index;
    }
};
```

#### 2.2 Create `OuterLinearStage` struct (matches Jolt exactly)

```zig
const OuterLinearStage = struct {
    const Self = @This();

    // Matches Jolt's OuterLinearStage fields
    az: poly_mod.DensePolynomial(F),
    bz: poly_mod.DensePolynomial(F),

    /// Initialize linear stage - materialize Az/Bz from trace
    /// Matches Jolt's LinearSumcheckStage::initialize()
    pub fn initialize(shared: *OuterSharedState, window_size: usize) !Self {
        // Calls fused_materialise_polynomials_round_zero equivalent
        // Computes az, bz AND t_prime in one pass
        ...
    }

    /// Rebuild t_prime from bound polynomials (no trace access)
    /// Matches Jolt's LinearSumcheckStage::next_window()
    pub fn nextWindow(self: *Self, shared: *OuterSharedState, window_size: usize) void {
        // compute_evaluation_grid_from_polynomials_parallel equivalent
        // Reads from bound self.az, self.bz
        // Rebuilds shared.t_prime_poly
        ...
    }

    /// Compute sumcheck round message
    /// Matches Jolt's LinearSumcheckStage::compute_message()
    pub fn computeMessage(self: *const Self, shared: *const OuterSharedState, window_size: usize, prev_claim: F) [4]F {
        const t_evals = shared.computeTevals(window_size);
        return shared.split_eq_poly.computeCubicRoundPoly(
            t_evals.t_zero,
            t_evals.t_inf,
            prev_claim,
        );
    }

    /// Ingest challenge - bind all polynomials
    /// Matches Jolt's LinearSumcheckStage::ingest_challenge()
    pub fn ingestChallenge(self: *Self, shared: *OuterSharedState, r: F) void {
        // 1. Bind split_eq
        shared.split_eq_poly.bind(r);

        // 2. Bind t_prime (CRITICAL - this was missing!)
        if (shared.t_prime_poly) |*t_prime| {
            t_prime.bind(r);
        }

        // 3. Bind az and bz (NOT r_grid - that's for streaming mode)
        self.az.bindLow(r);
        self.bz.bindLow(r);
    }
};
```

#### 2.3 Create orchestrator (matches Jolt's StreamingSumcheck)

```zig
const OuterStreamingProver = struct {
    const Self = @This();

    shared: OuterSharedState,
    linear: ?OuterLinearStage,
    current_round: usize,
    current_claim: F,

    pub fn init(shared: OuterSharedState, initial_claim: F) Self {
        return .{
            .shared = shared,
            .linear = null,
            .current_round = 0,
            .current_claim = initial_claim,
        };
    }

    /// Generate the remaining sumcheck proof
    /// Matches Jolt's StreamingSumcheck flow with LinearOnlySchedule
    pub fn generateProof(self: *Self, transcript: anytype) !Proof {
        var round: usize = 0;
        const num_rounds = self.shared.numRounds();

        while (round < num_rounds) : (round += 1) {
            // compute_message phase
            const message = if (round == 0) blk: {
                // Round 0: Initialize linear stage (switch_over point)
                self.linear = try OuterLinearStage.initialize(&self.shared, 1);
                break :blk self.linear.?.computeMessage(&self.shared, 1, self.current_claim);
            } else blk: {
                // Round 1+: next_window then compute_message
                self.linear.?.nextWindow(&self.shared, 1);
                break :blk self.linear.?.computeMessage(&self.shared, 1, self.current_claim);
            };

            // Append to proof, get challenge
            transcript.appendRoundPoly(message);
            const r = transcript.challengeScalar();

            // Update claim: evaluate polynomial at challenge
            self.current_claim = evaluateAt(message, r);

            // ingest_challenge phase
            self.linear.?.ingestChallenge(&self.shared, r);
        }

        return proof;
    }
};
```

### Phase 3: Verification

- [ ] Add debug output for intermediate values at each round
- [ ] Compare t_prime_poly values with Jolt
- [ ] Compare Az, Bz polynomial values after binding
- [ ] Run `test_verify_zolt_proof`

---

## Implementation Checklist

### Phase 1: Primitives
- [ ] Add `MultiquadraticPolynomial.bind()` method
- [ ] Add unit tests for bind()
- [ ] Add `GruenSplitEqPolynomial.getEActiveForWindow()`
- [ ] Add unit tests for getEActiveForWindow()

### Phase 2: Restructure
- [ ] Create `OuterSharedState` struct with all fields matching Jolt
- [ ] Create `OuterLinearStage` struct with:
  - [ ] `initialize()` - fused materialization from trace
  - [ ] `nextWindow()` - rebuild t_prime from bound polys
  - [ ] `computeMessage()` - project and compute cubic
  - [ ] `ingestChallenge()` - bind split_eq, t_prime, az, bz
- [ ] Create `OuterStreamingProver` orchestrator
- [ ] Update proof generation to use new architecture
- [ ] Remove old monolithic StreamingOuterProver

### Phase 3: Verification
- [ ] Add debug output for intermediate values
- [ ] Compare with Jolt at each step
- [ ] Run `test_verify_zolt_proof`

---

## Key Files to Modify

| File | Changes |
|------|---------|
| `src/poly/multiquadratic.zig` | Add `bind()` method |
| `src/poly/split_eq.zig` | Add `getEActiveForWindow()` |
| `src/zkvm/spartan/streaming_outer.zig` | Full restructure to match Jolt |

---

## Test Commands

```bash
# Run Zolt tests
zig build test --summary all

# Generate proof
zig build -Doptimize=ReleaseFast && ./zig-out/bin/zolt prove examples/fibonacci.elf \
  --jolt-format -o /tmp/zolt_proof_dory.bin

# Jolt verification tests
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```

---

## Reference Files in Jolt

| File | Contents |
|------|----------|
| `jolt-core/src/subprotocols/streaming_sumcheck.rs` | Trait definitions and StreamingSumcheck orchestrator |
| `jolt-core/src/zkvm/spartan/outer.rs` | OuterSharedState, OuterLinearStage, OuterStreamingWindow |
| `jolt-core/src/poly/multiquadratic_poly.rs` | MultiquadraticPolynomial with bind() |
| `jolt-core/src/poly/split_eq_poly.rs` | GruenSplitEqPolynomial with E_active_for_window() |
| `jolt-core/src/zkvm/prover.rs:536-567` | prove_stage1() showing UniSkip + remaining sumcheck integration |

---

## Verified Correct Components

### Transcript
- [x] Blake2b transcript format matches Jolt
- [x] Challenge scalar computation (128-bit, no masking)
- [x] Field serialization (Arkworks LE format)

### Polynomial Computation
- [x] Gruen cubic polynomial formula
- [x] Split eq polynomial factorization (E_out/E_in)
- [x] bind() operation (eq factor computation)
- [x] Lagrange interpolation
- [x] evalsToCompressed format
- [x] DensePolynomial.bindLow() matches Jolt's bound_poly_var_bot

### RISC-V & R1CS
- [x] R1CS constraint definitions (19 constraints, 2 groups)
- [x] Constraint 8 (RightLookupSub) has 2^64 constant
- [x] UniSkip polynomial generation
- [x] Memory layout constants match Jolt
- [x] R1CS input ordering matches Jolt's ALL_R1CS_INPUTS

### Supporting Structures
- [x] ExpandingTable implementation
- [x] GruenSplitEqPolynomial (partial - missing E_active)
- [x] MultiquadraticPolynomial (partial - missing bind)

### All Tests Pass
- [x] 702/702 Zolt tests pass
