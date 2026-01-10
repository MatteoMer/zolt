zig build && \
    ./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format \   --export-preprocessing /tmp/zolt_preprocessing.bin \
    -o /tmp/zolt_proof_dory.bin --srs /tmp/jolt_dory_srs.bin && \
    cd ../jolt && cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture 2>&1 | tee /tmp/jolt.log && cd -


