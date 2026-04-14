//! zolt-pool: Thread pool and parallel primitives for Zolt.
//!
//! A work-stealing thread pool (Chase-Lev deque) with parallel sort.

pub const thread_pool = @import("thread_pool.zig");
pub const ThreadPool = thread_pool.ThreadPool;
pub const is_wasm = thread_pool.is_wasm;

pub const parallel_sort = @import("parallel_sort.zig");
pub const parallelSort = parallel_sort.parallelSort;

pub const helpers = @import("helpers.zig");
pub const parallelReduceOptional = helpers.parallelReduceOptional;
pub const parallelReduceForceOptional = helpers.parallelReduceForceOptional;
pub const parallelForOptional = helpers.parallelForOptional;

test {
    _ = @import("thread_pool.zig");
    _ = @import("parallel_sort.zig");
}
