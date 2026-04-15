// Zolt Prover Worker — runs all WASM operations off the main thread.
// The main thread stays responsive while proving runs here.

let zolt = null; // WASM instance exports

self.onmessage = async ({ data }) => {
  switch (data.type) {

    case "init": {
      // Instantiate WASM module (with shared memory if threaded, standalone if not).
      const { wasmModule, memory } = data;
      try {
        const imports = memory ? { env: { memory } } : {};
        const instance = await WebAssembly.instantiate(wasmModule, imports);
        zolt = instance.exports;
        self.postMessage({ type: "ready" });
      } catch (e) {
        self.postMessage({ type: "error", message: e.message });
      }
      break;
    }

    case "setup": {
      // Create pool, allocate stacks + TLS for compute workers.
      const { workerCount } = data;
      try {
        const poolHandle = zolt.zolt_thread_pool_create_wasm(workerCount);
        if (!poolHandle) throw new Error("zolt_thread_pool_create_wasm returned null");
        const poolPtr = zolt.zolt_thread_pool_ptr(poolHandle);

        const tlsSize = zolt.__tls_size.value;
        const tlsAlign = zolt.__tls_align.value;
        const STACK_SIZE = 1 * 1024 * 1024; // 1 MB

        const workerInfos = [];
        for (let i = 0; i < workerCount; i++) {
          const stackAlloc = zolt.zolt_alloc(STACK_SIZE);
          if (!stackAlloc) throw new Error(`Failed to allocate stack for worker ${i}`);
          const stackBase = stackAlloc + STACK_SIZE; // stacks grow downward

          const tlsAlloc = zolt.zolt_alloc(tlsSize + tlsAlign);
          if (!tlsAlloc) throw new Error(`Failed to allocate TLS for worker ${i}`);
          const tlsBase = (tlsAlloc + tlsAlign - 1) & ~(tlsAlign - 1);

          workerInfos.push({ stackBase, tlsBase });
        }

        self.postMessage({ type: "setup-done", poolPtr, workerInfos, poolHandle });
      } catch (e) {
        self.postMessage({ type: "error", message: e.message });
      }
      break;
    }

    case "prove": {
      // Load ELF, run prover, return result.
      const { elfBytes, elfName, poolHandle } = data;
      try {
        // Copy ELF into WASM memory
        const elfPtr = zolt.zolt_alloc(elfBytes.length);
        if (!elfPtr) throw new Error("zolt_alloc returned null");
        const mem = new Uint8Array(zolt.memory.buffer);
        mem.set(elfBytes, elfPtr);

        // Load ELF with pool handle
        const elfHandle = zolt.zolt_load_elf_bytes(poolHandle, elfPtr, elfBytes.length);
        zolt.zolt_free(elfPtr, elfBytes.length);
        if (!elfHandle) throw new Error("zolt_load_elf_bytes returned null");

        const bytecodeSize = zolt.zolt_loaded_elf_size(elfHandle);
        self.postMessage({ type: "status", message: `ELF loaded -- bytecode: ${bytecodeSize} bytes` });

        // Prove
        const t0 = performance.now();
        const proofHandle = zolt.zolt_prove(elfHandle);
        const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
        zolt.zolt_loaded_elf_destroy(elfHandle);

        if (!proofHandle) throw new Error("zolt_prove returned null");

        const proofSize = zolt.zolt_proof_result_size(proofHandle);
        const proofPtr = zolt.zolt_proof_result_ptr(proofHandle);
        const proofBytes = new Uint8Array(zolt.memory.buffer, proofPtr, proofSize).slice();
        zolt.zolt_proof_result_destroy(proofHandle);

        const memMB = (zolt.memory.buffer.byteLength / 1024 / 1024).toFixed(0);

        // Hash the proof
        const hash = await crypto.subtle.digest("SHA-256", proofBytes);
        const sha256 = [...new Uint8Array(hash)].map(b => b.toString(16).padStart(2, "0")).join("");

        self.postMessage({
          type: "proof-done",
          elapsed, proofSize, memMB, sha256,
        });
      } catch (e) {
        self.postMessage({ type: "error", message: e.message });
      }
      break;
    }
  }
};
