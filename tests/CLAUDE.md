# tests/ - Test Suite

## 目录结构

```
tests/
├── conftest.py              # GPU 检测 fixtures (DeviceInfo)
├── test_*.py                # 单元/集成测试
├── test_e2e/                # 端到端测试
├── test_patterns/           # Pattern 专项测试
├── test_memory/             # 内存子系统测试
└── benchmarks/              # 性能基准测试
```

## Pytest Markers

`nvidia`, `amd`, `multigpu`, `benchmark`

## 运行方式

| 命令 | 范围 |
|------|------|
| `make test` | 全部测试 |
| `make test-unit` | 单元测试（不需要多 GPU） |
| `make test-multigpu` | 多 GPU 测试 |
| `make bench` | Benchmark 测试 |

## 约定

- Microbenchmark 输出归一化带宽百分比，P2P 目标 >= 95%。
- `conftest.py` 提供 GPU 检测 fixtures 和 backend 参数化。
