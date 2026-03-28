# tncc/sync/ - Synchronization Layer

## 两类原语

### 远端原子操作 (tile_atomic_*)

跨 GPU 远端内存操作，传入 `target_rank` + `caller_rank` + `heap_bases`，内部 `translate_ptr`。

- 算术: `tile_atomic_add`, `tile_atomic_cas`, `tile_atomic_xchg`
- 比较: `tile_atomic_min`, `tile_atomic_max`
- 位运算: `tile_atomic_and`, `tile_atomic_or`, `tile_atomic_xor`

### 本地信号原语

仅操作本地 GPU 内存，不做指针翻译。用于 WG Specialization / Producer-Consumer 模式的 tile 级同步。

- `tile_signal` / `tile_signal_add` — 发信号
- `tile_wait` / `tile_wait_ge` — 等待（CAS 自旋 / zero-add 轮询）
- `tile_barrier` — 一次性屏障（重复使用需不同 slot 或重置计数器）

跨 GPU 信号需调用方先用 `translate_ptr` 翻译 locks 指针。

## 内存序语义（C++ 模型）

- `sem`: `"relaxed"` / `"acquire"` / `"release"` / `"acq_rel"`
- `scope`: `"block"` / `"gpu"` / `"sys"`
- `tile_signal` 默认 release，`tile_wait` 默认 acquire（形成 release-acquire 对）

## 约束

- 所有函数 `@triton.jit`。
