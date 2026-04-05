use clap::Parser;
use common::jolt_device::MemoryConfig;
use jolt_core::guest;
use jolt_core::poly::commitment::dory::DoryCommitmentScheme;
use jolt_core::zkvm::prover::JoltProverPreprocessing;
use jolt_core::zkvm::verifier::JoltSharedPreprocessing;
use jolt_core::zkvm::RV64IMACProver;
use std::fs;
use std::time::Instant;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::{fmt, EnvFilter};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

type ProverPreproc =
    JoltProverPreprocessing<ark_bn254::Fr, jolt_core::curve::Bn254Curve, DoryCommitmentScheme>;

#[derive(Parser)]
#[command(name = "jolt-bench", about = "Benchmark Jolt (Rust) prover on raw ELF files")]
struct Cli {
    /// Path to the RISC-V ELF file
    elf: String,

    /// Maximum padded trace length (power of 2)
    #[arg(long, default_value = "65536")]
    max_trace: usize,

    /// Enable span close timing (shows Dory sub-operation durations)
    #[arg(long)]
    span_timing: bool,
}

fn main() {
    let cli = Cli::parse();
    let bench = std::env::var("JOLT_BENCH").is_ok();

    // Enable tracing for stage-level timing
    // In bench mode, use info level which picks up Jolt's native [BENCH] lines
    if bench || cli.span_timing {
        let level = if cli.span_timing {
            "jolt_core=trace"
        } else {
            "jolt_core=info"
        };
        let builder = fmt()
            .with_env_filter(EnvFilter::new(level))
            .with_target(false)
            .with_timer(fmt::time::uptime());
        if cli.span_timing {
            builder.with_span_events(FmtSpan::CLOSE).init();
        } else {
            builder.init();
        }
    } else {
        fmt()
            .with_env_filter(EnvFilter::new("jolt_core=info"))
            .with_target(false)
            .with_timer(fmt::time::uptime())
            .init();
    }

    eprintln!("=== Jolt (Rust) Prover Benchmark ===");
    eprintln!("ELF: {}", cli.elf);

    // Read ELF
    let elf_bytes = fs::read(&cli.elf).expect("Failed to read ELF file");
    eprintln!("ELF size: {} bytes", elf_bytes.len());

    // Decode bytecode
    let t_decode = Instant::now();
    let (bytecode, init_memory_state, program_size, entry_address) =
        guest::program::decode(&elf_bytes);
    let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "Decode: {:.2} ms ({} instructions, program_size={}, entry=0x{:x})",
        decode_ms,
        bytecode.len(),
        program_size,
        entry_address
    );

    // Build memory config
    let memory_config = MemoryConfig {
        heap_size: 33554432,
        stack_size: 33554432,
        max_input_size: 2000000,
        max_untrusted_advice_size: 0,
        max_trusted_advice_size: 0,
        max_output_size: 4096,
        program_size: Some(program_size),
    };
    let memory_layout = common::jolt_device::MemoryLayout::new(&memory_config);

    // Trace to get padded trace length
    let t_trace = Instant::now();
    let (_lazy_trace, trace, _memory, _io_device, _advice_tape) = guest::program::trace(
        &elf_bytes,
        None,
        &[], // no inputs
        &[], // no untrusted advice
        &[], // no trusted advice
        &memory_config,
        None,
    );
    let trace_ms = t_trace.elapsed().as_secs_f64() * 1000.0;
    let padded_trace_len = (trace.len() + 1).next_power_of_two();
    eprintln!(
        "Trace: {:.2} ms ({} cycles, padded to {})",
        trace_ms,
        trace.len(),
        padded_trace_len
    );
    drop(trace);
    drop(_lazy_trace);
    drop(_memory);

    // Preprocessing
    let t_preproc = Instant::now();
    let shared = JoltSharedPreprocessing::new(
        bytecode,
        memory_layout,
        init_memory_state,
        padded_trace_len.max(cli.max_trace),
        entry_address,
    )
    .expect("Failed to preprocess");
    let prover_preprocessing = ProverPreproc::new(shared);
    let preproc_ms = t_preproc.elapsed().as_secs_f64() * 1000.0;
    eprintln!("Preprocessing: {:.2} ms", preproc_ms);

    // Generate prover from ELF
    let t_gen = Instant::now();
    let prover = RV64IMACProver::gen_from_elf(
        &prover_preprocessing,
        &elf_bytes,
        &[], // no inputs
        &[], // no untrusted advice
        &[], // no trusted advice
        None, // trusted_advice_commitment
        None, // trusted_advice_hint
        None, // advice_tape
    );
    let gen_ms = t_gen.elapsed().as_secs_f64() * 1000.0;
    eprintln!("Prover gen: {:.2} ms", gen_ms);

    // Prove
    let t_prove = Instant::now();
    let (proof, _debug_info) = prover.prove();
    let prove_ms = t_prove.elapsed().as_secs_f64() * 1000.0;
    eprintln!("Prove: {:.2} ms", prove_ms);

    // Output prove_total for bench script parsing
    if bench {
        eprintln!("[BENCH] prove_total={:.1}", prove_ms);
    }

    let total_ms = decode_ms + trace_ms + preproc_ms + gen_ms + prove_ms;
    eprintln!("\n--- Summary ---");
    eprintln!("Decode:         {:>10.2} ms", decode_ms);
    eprintln!("Trace:          {:>10.2} ms", trace_ms);
    eprintln!("Preprocessing:  {:>10.2} ms", preproc_ms);
    eprintln!("Prover gen:     {:>10.2} ms", gen_ms);
    eprintln!("Prove:          {:>10.2} ms", prove_ms);
    eprintln!("Total:          {:>10.2} ms", total_ms);
    eprintln!("Commitments: {}", proof.commitments.len());
    eprintln!("Trace length: {}", proof.trace_length);
}
