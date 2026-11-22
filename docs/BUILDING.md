# 跨平台构建

本项目使用 C++17、CMake 和 vendored GGML。CPU 构建支持 Windows x64 与
Linux x86_64；CUDA 是可选后端，需要额外的 NVIDIA 工具链和显卡。

## 支持范围

| 平台 | 推荐编译器 | 生成器 | 默认后端 |
|---|---|---|---|
| Windows 10/11 x64 | Visual Studio 2026 | Visual Studio 18 2026 | CPU |
| Linux x86_64 | GCC 11+ 或 Clang 14+ | Unix Makefiles | CPU |

通用要求：

- CMake 3.21+，以便使用 `CMakePresets.json`。
- 支持 C++17 的 C++ 编译器。
- CUDA 构建需要 CUDA Toolkit 和匹配的 NVIDIA 驱动。

构建预设分为两类：

- `windows-cpu`、`windows-cuda`、`windows-cuda-portable`（以及对应的 `linux-*`）
  只构建 `omnivoice-cli`，用于日常开发和发布。
- `*-tests` 构建全部测试目标。`ctest` 预设默认假设测试已经构建。

## Windows

安装 Visual Studio 2026 的“使用 C++ 的桌面开发”工作负载。

CPU 构建：

```powershell
cmake --preset windows-cpu
cmake --build --preset windows-cpu          # 只构建 CLI
cmake --build --preset windows-cpu-tests    # 需要测试时再构建测试目标
ctest --preset windows-cpu-portable
```

构建目录为 `build/windows-cpu`。Visual Studio 多配置生成器的产物位于：

```text
build/windows-cpu/bin/Release/omnivoice-cli.exe
```

CUDA 构建：

```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5"

cmake --preset windows-cuda
cmake --build --preset windows-cuda          # 只构建 CLI
cmake --build --preset windows-cuda-tests    # 需要测试时再构建测试目标
```

运行 CUDA 程序前也需要保证 CUDA 运行库目录位于 `PATH` 中。

## Linux

Ubuntu/Debian 的基础依赖：

```bash
sudo apt install build-essential cmake
```

CPU 构建：

```bash
cmake --preset linux-cpu
cmake --build --preset linux-cpu          # 只构建 CLI
cmake --build --preset linux-cpu-tests    # 需要测试时再构建测试目标
ctest --preset linux-cpu-portable
```

构建目录为 `build/linux-cpu`，CLI 位于：

```text
build/linux-cpu/bin/omnivoice-cli
```

CUDA 构建：

```bash
cmake --preset linux-cuda \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build --preset linux-cuda          # 只构建 CLI
cmake --build --preset linux-cuda-tests    # 需要测试时再构建测试目标
```

如果 `nvcc` 已在 `PATH` 中，也可以只设置 `CUDACXX`：

```bash
export CUDACXX=/usr/local/cuda/bin/nvcc
cmake --preset linux-cuda
```

## 验证构建

Windows：

```powershell
.\build\windows-cpu\bin\Release\omnivoice-cli.exe --list-devices
.\build\windows-cuda\bin\Release\omnivoice-cli.exe --list-devices
```

Linux：

```bash
./build/linux-cpu/bin/omnivoice-cli --list-devices
./build/linux-cuda/bin/omnivoice-cli --list-devices
```

CUDA 构建需要能够加载 CUDA backend；Windows 运行前应确保 CUDA 运行库目录
位于 `PATH` 中，Linux 运行前应确保 CUDA 运行库可被动态链接器找到。

## CUDA 架构

`windows-cuda` 和 `linux-cuda` 默认使用 `native`，只编译构建机上探测到的 GPU
架构。例如构建机只有 RTX 2060 时，最终会使用 `75-real`。

需要兼容多代 GPU 时，使用 portable preset：

```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5"
cmake --preset windows-cuda-portable
cmake --build --preset windows-cuda-portable
```

```bash
cmake --preset linux-cuda-portable
cmake --build --preset linux-cuda-portable
```

portable preset 使用以下架构：

```text
75-real;75-virtual;80-virtual;86-real;89-real;90-virtual;120a-real;121a-real
```

普通 CUDA preset 也可以用 CMake 参数临时覆盖：

```bash
cmake --preset linux-cuda \
  -DOMNIVOICE_CUDA_ARCHITECTURES="86-real"
```

建议：

- 只在本机运行：使用默认的 `native`。
- 发布通用 CUDA 包：使用 `windows-cuda-portable` 或
  `linux-cuda-portable`。
- 没有 GPU 的构建机：不能使用 `native`，应显式指定架构或使用 portable
  preset。
- 架构列表越长，编译时间和二进制体积越大。

## 路径约定

- 源码、CMake 和测试都使用 `/`，Windows 同样接受正斜杠。
- API 接收 UTF-8 路径；包含空格或中文的路径应加引号。
- 模型、测试 golden 和生成音频不提交到 Git，通过 CMake cache 变量指定位置：

```text
OMNIVOICE_MODEL_DIR
OMNIVOICE_TEST_DATA_DIR
```

例如：

```bash
cmake --preset linux-cuda \
  -DOMNIVOICE_MODEL_DIR=/data/omnivoice/models \
  -DOMNIVOICE_TEST_DATA_DIR=/data/omnivoice/testdata
```

Windows 路径同样使用正斜杠，或引用 CMake 的 `PATH` 类型变量：

```powershell
cmake --preset windows-cuda `
  -DOMNIVOICE_MODEL_DIR="D:/omnivoice/models" `
  -DOMNIVOICE_TEST_DATA_DIR="D:/omnivoice/testdata"
```

## 不使用 Presets

CPU 构建也可以直接使用通用 CMake 命令。

Windows：

```powershell
cmake -S . -B build/windows-cpu -A x64 `
  -DOMNIVOICE_ENABLE_CUDA=OFF
cmake --build build/windows-cpu --config Release --parallel
```

Linux：

```bash
cmake -S . -B build/linux-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DOMNIVOICE_ENABLE_CUDA=OFF
cmake --build build/linux-cpu --parallel
```
