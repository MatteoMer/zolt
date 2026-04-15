// Zolt Web Worker — two-phase init for proper TLS separation.
// Phase 1 (init): Instantiate WASM module over shared memory.
// Phase 2 (start): Allocate TLS, set stack pointer, enter work-stealing loop.

let wasmInstance = null;

self.onmessage = async ({ data }) => {
  if (data.type === "init") {
    const { wasmModule, memory, workerId } = data;
    try {
      wasmInstance = await WebAssembly.instantiate(wasmModule, {
        env: { memory },
      });
      self.postMessage({ type: "ready", workerId });
    } catch (e) {
      self.postMessage({ type: "error", workerId, message: e.message });
    }

  } else if (data.type === "start") {
    const { poolPtr, workerId, stackBase, tlsBase } = data;
    try {
      // Set per-worker stack and TLS regions.
      // Each instance has its own __stack_pointer and __tls_base globals,
      // but they must point to separate regions in shared linear memory.
      wasmInstance.exports.__stack_pointer.value = stackBase;

      // Initialize TLS for this worker — copies initial TLS data to tlsBase
      // and sets __tls_base to point there.
      wasmInstance.exports.__wasm_init_tls(tlsBase);

      wasmInstance.exports.zolt_worker_entry(poolPtr, workerId);
      self.postMessage({ type: "exited", workerId });
    } catch (e) {
      self.postMessage({ type: "error", workerId, message: e.message });
    }
  }
};
