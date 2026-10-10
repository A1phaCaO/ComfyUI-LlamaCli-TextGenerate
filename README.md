# ComfyUI-LlamaCli-TextGenerate（vibe coding产物🤗）

用 llama.cpp（`llama-cli` 子进程）在工作流里跑 GGUF 文本生成，输入/输出对齐原版
Generate Text（`TextGenerate`）节点，但模型来源完全不受 ComfyUI 模型目录限制，
并额外暴露 llama.cpp 的部署参数。

## 与原版 TextGenerate 的对齐

- 输入同名同语义：`prompt` / `max_length` / `sampling_mode`（on/off 动态组合：
  temperature、top_k、top_p、min_p、repetition_penalty、seed、presence_penalty）/
  `thinking` / `use_default_template` / `system_prompt` / `mtp` / 可选 `image`；
- 输出：`generated_text`、`thinking`（额外多一个 `stats`：llama.cpp 的
  `[Prompt | Generation]` 性能行 + 节点墙钟时间）。
- 区别：不接 `CLIP`，改接 GGUF；每次执行 spawn 一次 `llama-cli`
  （模型加载 ~15-40s 计入墙钟，生成速度是 llama.cpp 原生 40-60 tok/s 档）。

## 模型加载（比 LLM-text-processor 灵活）

`model` 下拉框自动扫描 ComfyUI 已注册的所有模型目录（含 extra_model_paths
配置的路径）里 `LLM / clip / text_encoders / diffusion_models / unet / checkpoints`
这几个键下的 `*.gguf`，显示绝对路径，列表 30s 缓存、F5 即刷新。

目录外的文件（比如 `D:\Model\...` 里没注册的）不用复制不用链接：
高级输入 `model_path_override` 直接填**绝对路径**即可（优先级高于下拉框）。

## llama.cpp 二进制

**已内置**：`vendor/llama-b10472/`（官方 b10472 win-cuda-13.3 x64 全量，约 670MB），开箱即用。
换版本只需把新的解压目录放进 `vendor/` 下（旧目录删掉或改名即可，查找按名称排序取先）。

查找顺序：

1. 节点输入 `llama_cli_path`（目录或 llama-cli.exe 均可）；
2. 环境变量 `LLAMA_CLI`；
3. 本目录 `llama_path.txt`（首行非注释路径，默认留空）；
4. `PATH`；
5. 本包 `vendor/`（内置版）；
6. 本包目录、`ComfyUI-LLM-text-processor` 下的 `llama-cli.exe`；
7. 各盘符根目录的 `llama*/llama-cli.exe`。

## 部署参数（advanced 组，对齐 llama.cpp CLI）

| 输入 | CLI | 说明 |
|---|---|---|
| `ctx_size` | `-c` | 默认 16384；0 = 模型训练长度 |
| `n_gpu_layers` | `-ngl` | 默认 -1 = llama auto（能放下就全放） |
| `flash_attn` | `--flash-attn` | on/off/auto，on 可降 KV 内存 |
| `kv_cache_type` | `-ctk/-ctv` | f16/q8_0/bf16/f32，q8_0 约省一半 KV |
| `threads` | `-t` | -1 = auto |
| `batch_size` / `ubatch_size` | `-b` / `-ub` | prefill 速度/显存权衡 |
| `load_mode` | `--load-mode` | RAM 紧张且反复加载时选 `mmap+mlock` |
| `fit` | `--fit` | 默认 on：DiT 驻留时自动缩 ctx/层保证不崩 |
| `mtp` | `--spec-type draft-mtp` | auto 先读 GGUF 头部，有 MTP 张量才开 |
| `extra_args` | 原样追加 | 最后一手，可覆盖任何参数 |

`use_default_template=off` 等价原版的 skip_template：走 `llama-completion.exe` 裸补全
（当前 llama-cli 的 `--no-conversation` 会挂起不退出，所以 raw 模式用独立二进制；
此时 `system_prompt`/`image`/`thinking`/`mtp` 不生效）。

## 8GB 显存提示

- 和图像模型同跑时保持 `fit=on`，llama 会自动缩上下文避免 OOM；
- 想要更大 ctx：`flash_attn=on` + `kv_cache_type=q8_0`；
- 16GB 内存下反复加载同一模型，首次读盘最慢；想让每次加载稳定，
  `load_mode=mmap+mlock` 或生成前先在工作流里跑一次。
