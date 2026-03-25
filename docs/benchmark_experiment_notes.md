# Benchmark 实验注意事项

只保留容易踩坑、且会直接污染结论的规则。

## 基本规则

- 只信 `figures/data/*.json` 里的 canonical 结果，不信临时 stdout、不信截图、不信口头转述。
- 每次写结论都要带清楚环境：`GPU SKU`、`interconnect`、`world_size`、`transport`、`dtype`。
- 同一轮 benchmark 期间不要并发占用同一组 GPU；并发会直接污染 latency 和 bandwidth。
- 先过 correctness，再谈 performance；correctness 未闭环的结果不能上图。
- 图和 headline 只能基于“当前 public surface”写，不能拿内部实验 kernel 结果冒充公开能力。

## 计量口径

- 小消息看 latency，大消息看 bandwidth；不要把两者混成一张单口径图后直接下 headline。
- 结论里必须写清 timing mode：`host wall` 还是 `device event`。
- 如果 JSON / 图里混用了不同 measurement mode，必须显式标注，不能伪装成同一口径。
- 不要用固定的少量点同时承担 latency 与 bandwidth 结论；至少分成 `<= 64 KiB` 和 `>= 256 KiB` 两段看。
- 大消息如果单轮 sweep 明显异常慢，要先拆成 `single op + single size` 诊断，再决定是否回到总控 benchmark。

## 采样规则

- 小消息需要足够 warmup/iters，避免把首次 setup、JIT、workspace 初始化直接算进 headline。
- 大消息可以降低采样预算，但必须在结果里保留采样策略，避免后续横向比较失真。
- 如果结果对首轮极敏感，优先补 steady-state 复测，不要直接拿第一次跑出来的图发结论。
- 出现异常塌缩或异常领先时，先怀疑 benchmark 口径，再怀疑协议本身。

## collective 特别注意

- `allreduce` 要单独盯，因为它最容易在大消息段暴露真实协议退化，不能拿其他 collective 的表现替它背书。
- `allgather / scatter / reduce_scatter / broadcast` 如果只补了锚点，就在图里明确写“anchor”，不要假装已经有完整大消息曲线。
- multiprocess baseline 目前只写 `world_size=2 + ctypes_ipc`；其他 transport 仍属诊断面。

## 出图前检查

- 这张图表达的是 latency、bandwidth，还是 speedup。
- 这张图是否混入了 incomplete anchor、不同 timing mode、不同 transport。
- headline 是否只来自当前图中真实可见的数据，而不是补脑。
- JSON 中是否留下 measurement mode、sampling policy、execution metadata。
