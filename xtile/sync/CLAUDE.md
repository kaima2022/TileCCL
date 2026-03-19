# xtile/sync/ - Synchronization Layer

## 两类原语，严格区分

### 远端原子操作 (tile_atomic_*)
- 支持跨 GPU 远端内存操作
- 必须传入 `target_rank` + `caller_rank` + `heap_bases`
- 内部调用 translate_ptr 做指针翻译
- 当 target_rank == caller_rank 时，翻译是恒等操作（零开销）

### 本地信号原语 (tile_signal / tile_wait)
- 仅操作本地 GPU 内存（如同 kernel 内 compute/comm worker 共享的 locks tensor）
- 不做指针翻译，不需要 heap_bases
- 用于 WG Specialization / Producer-Consumer 模式内的 tile 级同步
- 跨 GPU 信号需要调用方先用 translate_ptr 翻译 locks 指针

## 内存序语义（C++ 模型，非 SHMEM）
- sem: "relaxed" / "acquire" / "release" / "acq_rel"
- scope: "block" / "gpu" / "sys"
- tile_signal 默认 release，tile_wait 默认 acquire（形成 release-acquire 对）

## 关键约束
- 所有函数 @triton.jit
- tile_wait 使用 CAS 自旋（while atomic_cas != expected: pass）
- tile_wait_ge 使用 zero-add 轮询（atomic_add(ptr, 0) 读取不修改）
- tile_barrier 是一次性的，重复使用需要不同 slot 或重置计数器
