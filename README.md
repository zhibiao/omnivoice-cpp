# OmniVoice C++ / GGML

基于 OmniVoice 0.2.1 的 C++17 / GGML Voice Clone 原生实现。运行时只读取
GGUF，不依赖 Python、PyTorch 或 Transformers。

## 当前状态

已实现：

- 24 kHz 单声道 WAV 输出；参考音频支持 WAV / MP3 输入。
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
- 参考音频支持 WAV 和 MP3，统一转换为 24 kHz 单声道后送入模型；FLAC、OGG、M4A 等其他格式直接报错。
- 只允许转换 LM；Higgs audio tokenizer 必须保持 F32。

## 平台与编译

支持 Windows x64 和 Linux x86_64。默认只构建 CPU backend，CUDA backend 需要
显式启用。详细依赖、CUDA 工具链和路径设置见
[docs/BUILDING.md](docs/BUILDING.md)。

Windows CPU 快速开始：

```powershell
cmake --preset windows-cpu
cmake --build --preset windows-cpu
```

Linux、CUDA、portable CUDA、测试命令和设备检查统一参见构建文档。

## 模型转换

公开转换入口：

```text
python tools/convert_hf_to_gguf.py --model-dir docs/OmniVoice-0.2.1/models/k2-fsa/OmniVoice --output-dir models --overwrite
```

Windows venv 使用 `.\.venv\Scripts\python.exe`，Linux venv 使用
`.venv/bin/python`；其余参数完全相同。

生成：

```text
models/omnivoice-lm-f32.gguf
models/omnivoice-tokenizer-f32.gguf
```

校验转换结果：

```text
python tests/tools/verify_gguf.py --model-dir docs/OmniVoice-0.2.1/models/k2-fsa/OmniVoice --lm-gguf models/omnivoice-lm-f32.gguf --audio-gguf models/omnivoice-tokenizer-f32.gguf
```

## LM 精度

使用同一个脚本生成不同精度：

```text
python tools/quantize_gguf.py --input models/omnivoice-lm-f32.gguf --output models/omnivoice-lm-q8_0.gguf --type q8_0 --overwrite
```

支持的 `--type`：

```text
q8_0
f16
bf16
```

只转换大型二维矩阵权重；norm、bias 和敏感 tensor 保持 F32。

- 默认推荐 Q8_0。
- F16 更快，但生成音频与 Q8_0 不完全相同，需要音质评估。
- RTX 2060 没有原生 BF16 Tensor Core，不建议使用 BF16。
- GPU 性能对比见[当前性能](#当前性能)。

## CLI

克隆命令。Windows：

```powershell
.\build\windows-cuda\bin\Release\omnivoice-cli.exe --clone `
  --model models\omnivoice-lm-q8_0.gguf `
  --audio-tokenizer models\omnivoice-tokenizer-f32.gguf `
  --prompt-input voices\nana.ovprompt `
  --text "今天，中国人民正在意气风发推进中国式现代化建设。" `
  --language zh `
  --device CUDA0 `
  --output clone.wav `
  --num-step 32
```

Linux：

```bash
./build/linux-cuda/bin/omnivoice-cli --clone \
  --model models/omnivoice-lm-q8_0.gguf \
  --audio-tokenizer models/omnivoice-tokenizer-f32.gguf \
  --prompt-input voices/nana.ovprompt \
  --text "今天，中国人民正在意气风发推进中国式现代化建设。" \
  --language zh \
  --device CUDA0 \
  --output clone.wav \
  --num-step 32
```

创建可复用 Prompt：

```powershell
.\build\windows-cuda\bin\Release\omnivoice-cli.exe --create-prompt `
  --model models\omnivoice-lm-f32.gguf `
  --audio-tokenizer models\omnivoice-tokenizer-f32.gguf `
  --ref-audio voices\nana.wav `
  --ref-text voices\nama.txt `
  --language zh `
  --device CUDA0 `
  --prompt-output voices\nana.ovprompt
```

```bash
./build/linux-cuda/bin/omnivoice-cli --create-prompt \
  --model models/omnivoice-lm-f32.gguf \
  --audio-tokenizer models/omnivoice-tokenizer-f32.gguf \
  --ref-audio voices/nana.wav \
  --ref-text voices/nama.txt \
  --language zh \
  --device CUDA0 \
  --prompt-output voices/nana.ovprompt
```

`--ref-text` 支持直接文本或 UTF-8 `.txt` 文件。

`--denoise` 控制是否在 prompt 中加入 `<|denoise|>`，默认取 GGUF 配置
（通常为 `true`）。`--no-denoise` 等价于 `--denoise false`；关闭后可能
保留更多参考音频的噪声/表现力，但一致性可能下降。示例：

```bash
./build/linux-cuda/bin/omnivoice-cli --clone \
  --prompt-input voices/nana.ovprompt \
  --text "今天，中国人民正在意气风发推进中国式现代化建设。" \
  --no-denoise
```

## HTTP 服务

`omnivoice-server` 暴露 OpenAI 兼容的语音合成端点，声音来自启动时加载的
`.ovprompt`：

```powershell
.\build\windows-cuda\bin\Release\omnivoice-server.exe `
  --model models\omnivoice-lm-q8_0.gguf `
  --audio-tokenizer models\omnivoice-tokenizer-f32.gguf `
  --voice-dir voices `
  --device CUDA0 `
  --port 8080
```

```bash
curl http://127.0.0.1:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"omnivoice","input":"今天天气不错。","voice":"nana","response_format":"wav"}' \
  --output speech.wav
```

| 端点 | 说明 |
|---|---|
| `POST /v1/audio/speech` | 文本转语音 |
| `GET /v1/models` | 返回配置的模型名 |
| `GET /health` | 存活探测 |

请求字段：`input`（必填，≤4096 字符）、`voice`（缺省时用
`--default-voice`，只有一个声音时自动选中）、`response_format`（`wav` 或
`pcm`，默认 `wav`）、`speed`（0.25–4.0）；扩展字段 `language`、`denoise`、
`seed`。`model` 与 `instructions` 会被接受但忽略，以便官方 SDK 直接调用。

没有 `/v1/audio/voices`：可用声音由 `--voice-dir`（扫描 `*.ovprompt`）和
`--voice NAME=PATH` 决定，`voice` 未知时 400 响应会列出全部可用名字。

推理串行执行。模型内部的 ggml graph 带可变缓存，不能并发进入，因此 server
用单工作线程 + 有界队列（`--max-queue`，默认 2）处理请求，队列满时返回
429；HTTP 线程只负责入队和等待，`/health` 与 `/v1/models` 不会被生成阻塞。
需要更高吞吐时应启动多个进程，每个进程独占一张 GPU。

`--api-key KEY` 要求 `Authorization: Bearer KEY`（`/health` 除外）。`wav`
输出 24 kHz 单声道 16-bit PCM；`pcm` 为裸 s16le、同样 24 kHz 单声道；
`mp3`/`flac` 等其它格式返回 400。

## 当前性能

相同负载：实际生成 9.04 秒音频、8 x 221 target cells、32 步扩散、
CUDA backend、默认后处理；目标文本为上文示例中文句重复两次。RTX 2060
数据为 3 次运行的中位数。

| LM | RTX 2060 RTF | RTX A4000 RTF | RTX 5070 Ti RTF |
|---|---:|---:|---:|
| Q8_0 | 0.290 | 0.126 | 0.063 |
| F16 | 0.262 | - | - |
| BF16 | 0.764 | 0.144 | 0.100 |

RTX 2060 实测环境：Ryzen 7 4800H、GeForce RTX 2060 6 GB、驱动
616.92（CUDA UMD 13.4）、CUDA Toolkit 13.5、Visual Studio 2026、
`native` 架构解析为 `75-real`。

## 目录

```text
app/             CLI
src/             C++ / GGML 模型与音频实现
models/          GGUF 模型
tools/           模型转换和精度转换入口
voices/          参考音频、参考文本和 `.ovprompt`
docs/            跨平台构建说明和 OmniVoice PyTorch 参考源码
vendor/ggml-0.29.0/  vendored GGML
```
