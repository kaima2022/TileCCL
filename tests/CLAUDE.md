# tests/ - Test Suite

## 规范
- microbenchmark 输出归一化带宽百分比
- 目标：P2P >= 95%
- 使用 pytest markers: nvidia, amd, multigpu, benchmark
- conftest.py 提供 GPU 检测 fixtures

## 运行方式
- make test: 全部测试
- make test-unit: 单元测试（不需要多GPU）
- make bench: benchmark 测试
