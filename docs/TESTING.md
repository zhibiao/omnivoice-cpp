# 跨平台测试

测试分为三类：

| 标签 | 内容 | 外部资产 |
|---|---|---|
| `portable` | 类型、WAV I/O、Prompt round-trip、CLI 冒烟测试 | 不需要 |
| `requires-assets` | GGUF/config、模型 golden、完整 Voice Clone | 需要 `models/` 和 `testdata/` |
| `cuda` | CUDA backend 与 CUDA 端到端测试 | 需要 NVIDIA GPU 和模型资产 |

`models/`、`testdata/` 和 WAV 文件默认被 `.gitignore` 忽略，因此普通 CI 只运行
`portable` 测试。完整测试必须显式准备模型和 golden 数据。

## 便携测试

Windows：

```powershell
cmake --preset windows-cpu
cmake --build --preset windows-cpu
ctest --preset windows-cpu-portable
```

Linux：

```bash
cmake --preset linux-cpu
cmake --build --preset linux-cpu
ctest --preset linux-cpu-portable
```

便携测试覆盖：

- `AudioBuffer`、`AudioCodes` 等基础数据结构。
- WAV 写入、读取、样本格式转换和 UTF-8 路径。
- Voice Clone Prompt 保存、加载、损坏数据拒绝和 UTF-8 路径。
- CLI 的版本、帮助、设备枚举和错误参数行为。

## 完整资产测试

先按 README 将参考模型转换为：

```text
models/omnivoice-lm-f32.gguf
models/omnivoice-tokenizer-f32.gguf
```

再生成测试 golden。具体脚本位于 `tests/tools/`，所有脚本都接受显式路径参数，
因此可在 Windows PowerShell 和 Linux shell 中运行。例如：

```text
python tests/tools/generate_qwen3_block_fixture.py --output testdata/qwen3_block_fixture.bin
```

执行全部 CPU 资产测试。Windows：

```powershell
ctest --test-dir build/windows-cpu -C Release -L requires-assets --output-on-failure
```

Linux：

```bash
ctest --test-dir build/linux-cpu -L requires-assets --output-on-failure
```

## CUDA 测试

CUDA 测试需要真实 GPU，GitHub 托管 runner 默认没有可用 GPU，因此放入自托管
runner 或本地验证：

```powershell
ctest --preset windows-cuda-assets
```

```bash
ctest --preset linux-cuda-assets
```

CUDA 测试通过 `OMNIVOICE_TEST_DEVICE=CUDA0` 选择设备。设备不存在或资产缺失时，
测试可以返回 `77` 并标记为 skipped。

CUDA 数值回归同时设置 `NVIDIA_TF32_OVERRIDE=0`，让 cuBLAS 使用完整 F32 而不是
Ampere/Ada 默认允许的 TF32。生产运行不强制关闭 TF32；测试关闭它是为了与
PyTorch CPU golden 做稳定、可重复的数值比较。

## CI

`.github/workflows/ci.yml` 在 Windows 和 Ubuntu 上各自执行：

1. CPU-only Release 配置。
2. 完整编译。
3. 带 `portable` 标签的测试。

这样无需公开大型模型和 GPU 也能持续验证两端构建、动态库加载、CLI 和基础文件
I/O。CUDA 与模型数值回归应在自托管 GPU runner 上增加独立 job。

## 添加新测试

新增测试时遵循以下规则：

1. 不依赖模型或 golden 的测试使用 `omnivoice_add_test` 后设置
   `LABELS "portable"`。
2. 依赖模型或 golden 的测试保留默认 `requires-assets` 标签。
3. CUDA 测试使用 `omnivoice_add_cuda_test`，并确保缺失设备时返回 `77`。
4. 新增 PATH 参数时使用 CMake cache 变量，不要在测试中写死盘符。
5. 文件路径 API 输入 UTF-8，测试至少覆盖普通路径和带空格的临时路径。
