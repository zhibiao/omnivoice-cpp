# OmniVoice C++ / GGML

基于 OmniVoice 0.2.1 的 C++17 / GGML Voice Clone 原生实现。运行时只读取
GGUF，不依赖 Python、PyTorch 或 Transformers。

## 当前状态

已实现：

- 24 kHz 单声道 WAV 输入输出。
- Qwen BPE 文本 tokenizer。
- Higgs HuBERT + DAC + RVQ 参考音频编码。
- 28 层双向 Qwen3 audio language model。
- CFG 迭代式 MASK 解遮码。
- Higgs Audio Decoder。
- 可复用 `.ovprompt` 音色 Prompt。
- 长文本分块、音频拼接和 cross-fade。
- CPU / CUDA 共用一个由 `--device` 指定的 backend。
- LM F32、Q8_0、F16、BF16。

当前计算路径：

- Conditional / unconditional LM 使用固定形状持久图。
- 默认 `class_temperature=0` 时，CPU 和 CUDA 共用 backend Selector graph。
- `class_temperature>0` 使用 host 随机采样路径。
- CUDA Higgs Decoder 使用 phase 分解转置卷积、`1x1` 矩阵乘路径和持久图。
- Qwen3 使用 packed QKV、无显式 mask FlashAttention 和 RMSNorm 融合。

## 当前限制

- 只实现 Voice Clone。
- 未实现 Voice Design、`instruct` 音色设计和 Auto Voice CLI。
- 不支持 Whisper 自动转写参考文本。
- 不包含训练、微调或 Web UI。
- 输入必须是 24 kHz 单声道 PCM WAV。
- 只允许转换 LM；Higgs audio tokenizer 必须保持 F32。

## 编译

当前 CMake 同时构建 CPU 和 CUDA backend：

```powershell
cmake -S . -B build -A x64 -DOMNIVOICE_ENABLE_CUDA=ON
cmake --build build --config Release --parallel
```

已有 `build` 配置时直接增量编译：

```powershell
cmake --build build --config Release --parallel
```

运行 CUDA 前加载 CUDA Toolkit DLL：

```powershell
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5\bin\x64;" + $env:PATH
```

查看设备：

```powershell
.\build\bin\Release\omnivoice-cli.exe --list-devices
```

## 模型转换

公开转换入口：

```powershell
.\.venv\Scripts\python.exe .\tools\convert_hf_to_gguf.py `
  --model-dir .\docs\OmniVoice-0.2.1\models\k2-fsa\OmniVoice `
  --output-dir .\models `
  --overwrite
```

生成：

```text
models/omnivoice-lm-f32.gguf
models/omnivoice-tokenizer-f32.gguf
```

校验转换结果：

```powershell
.\.venv\Scripts\python.exe .\tests\tools\verify_gguf.py `
  --model-dir .\docs\OmniVoice-0.2.1\models\k2-fsa\OmniVoice `
  --lm-gguf .\models\omnivoice-lm-f32.gguf `
  --audio-gguf .\models\omnivoice-tokenizer-f32.gguf
```

## LM 精度

使用同一个脚本生成不同精度：

```powershell
.\.venv\Scripts\python.exe .\tools\quantize_gguf.py `
  --input .\models\omnivoice-lm-f32.gguf `
  --output .\models\omnivoice-lm-q8_0.gguf `
  --type q8_0 `
  --overwrite
```

支持的 `--type`：

```text
q8_0
f16
bf16
```

只转换大型二维矩阵权重；norm、bias 和敏感 tensor 保持 F32。

RTX 2060 长文本、9.04 秒音频、32 步扩散实测：

| LM | 大小 | 生成时间 | RTF |
|---|---:|---:|---:|
| Q8_0 | 656 MB | 2.72 s | 0.301 |
| F16 | 1.23 GB | 2.51 s | 0.278 |
| BF16 | 1.23 GB | 6.92 s | 0.765 |

- 默认推荐 Q8_0。
- F16 更快，但生成音频与 Q8_0 不完全相同，需要音质评估。
- RTX 2060 没有原生 BF16 Tensor Core，不建议使用 BF16。

## CLI

克隆命令：

```powershell
.\build\bin\Release\omnivoice-cli.exe --clone `
  --model .\models\omnivoice-lm-q8_0.gguf `
  --audio-tokenizer .\models\omnivoice-tokenizer-f32.gguf `
  --prompt-input .\voices\nana.ovprompt `
  --text "今天，中国人民正在意气风发推进中国式现代化建设。" `
  --language zh `
  --device CUDA0 `
  --output .\clone.wav `
  --num-step 32
```

创建可复用 Prompt：

```powershell
.\build\bin\Release\omnivoice-cli.exe --create-prompt `
  --model .\models\omnivoice-lm-f32.gguf `
  --audio-tokenizer .\models\omnivoice-tokenizer-f32.gguf `
  --ref-audio .\voices\nana.wav `
  --ref-text .\voices\nama.txt `
  --language zh `
  --device CUDA0 `
  --prompt-output .\voices\nana.ovprompt
```

`--ref-text` 支持直接文本或 UTF-8 `.txt` 文件。

## 日志与输出

进度、模型加载、扩散步骤、解码和 RTF 写入 `stderr`：

```text
chunk 1/1: selector: CUDA0 GGML graph, fixed output mask
chunk 1/1: diffusion step 32/32: selected=412 remaining=0 ...
chunk 1/1: audio decoded: 212160 samples, 8.84 s ...
generation complete in 2.72s, audio=9.040000 s, RTF=0.300702
```

克隆成功时，`stdout` 只输出最终文件：

```text
wrote cloned audio: .\clone.wav
```

`generation_seconds` 不包含模型加载和已有 Prompt 的加载时间。

## 当前性能

RTX 2060、Q8_0 LM、8 x 221 target cells、32 步扩散：

```text
conditional LM:   约 46 ms/step
unconditional LM: 约 27 ms/step
Selector:         约 2.2 ms/step
Higgs Decoder:    约 165-200 ms
generation:       约 2.72 s
RTF:              约 0.301
```

## 测试

```powershell
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5\bin\x64;" + $env:PATH
ctest --test-dir build -C Release --output-on-failure
```

当前 Release 配置 `36/36` 测试通过，覆盖：

- GGUF/config 和 tokenizer，包括中英混合与中英文标点。
- Qwen3 block、完整 LM 的 CPU/CUDA golden。
- backend Selector graph 的 CPU/CUDA、guidance 变体和持久图复用。
- Higgs encoder/decoder 的 CPU/CUDA golden。
- Diffusion、mask schedule 和 token selection。
- Voice Clone 端到端，包括参考音频 → `.ovprompt` → 生成，以及 Q8_0 多分块。
- CLI、Context/Model 生命周期。

## 目录

```text
app/             CLI
src/             C++ / GGML 模型与音频实现
models/          GGUF 模型
testdata/        测试数据和生成音频
tests/           C++ 测试
tests/tools/     golden 生成与 GGUF 校验工具
tools/           模型转换和精度转换入口
voices/          参考音频、参考文本和 `.ovprompt`
docs/            OmniVoice PyTorch 参考源码
vendor/ggml/     vendored GGML
```
