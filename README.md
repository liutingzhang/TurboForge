# TurboForge

新模型发布后，要在自己的显卡上跑起来、跑得动、上下文拉得长，先把权重变成这台机器能用的产物，再交给对应的推理框架。TurboForge 做前半段：识别、量化、验收。Dense 产物交给 NInfer；MoE 按显卡交给 Basalt 或 Strata。

[English](README_EN.md) · [Apache-2.0](LICENSE) · [第三方许可](NOTICE) · [配置模板](.env.example)

**Windows 10/11 x64 + NVIDIA 显卡。双击启动，在网页里完成：选模型 → 检查 → 选量化 → 转换 → 验收 → 启动 API。**

## 它帮你做什么

- **认模型**：读配置、几何和权重，不靠文件名猜测。支持 Hugging Face Safetensors、受支持的 NVFP4 和 GGUF。
- **选量化**：只列出当前模型和引擎能用的方案。显卡有原生 FP4/FP8 时，对应方案排在前面。
- **跑起来**：自动准备环境，转换后做 GPU 加载检查，可选测速和质量门禁，然后启动本机 OpenAI 兼容接口。
- **选引擎**：Dense 用 ninfer-fusion-kvmem 拉长上下文；RTX 50 上已适配的 MoE 用 Basalt；其他 MoE 用 Strata。

现成聊天工具直接打开别人做好的模型。TurboForge 把你自己的、微调过的或越狱过的权重，做成这台电脑上可复现的模型服务。

## 安装并打开

1. 从 Releases 下载两个分卷包 `TurboForge-Windows-x64-chinese-part1.zip` 和 `TurboForge-Windows-x64-chinese-part2.zip`，放在同一个文件夹。GitHub 单文件 2 GB 限制是分卷的原因。
2. 两个包都解压到同一个上级目录，例如 `D:\`；Windows 询问时选择“合并”。解压后应得到 `D:\TurboForge`。不要在压缩包里直接运行。
3. 打开 `TurboForge`，双击 **`启动 TurboForge.bat`**，等浏览器打开。窗口不要关，关掉就会停服务。
4. 网页打不开时，看窗口里打印的地址。默认是 `http://127.0.0.1:8090`，这个端口被占用会自动换一个。

首次需要联网，也需要已经装好的 NVIDIA 驱动。不用预先安装 Git、Node.js、Python 或 CUDA Toolkit。两个分卷共约 2.8 GB（1.2 GB + 1.6 GB），带源码、网页、Python 3.13.18 和匹配的 NInfer 引擎，不含模型和个人密钥。校验值见 `SHA256SUMS.txt`，两个分卷缺一不可。

只想检查环境、不做转换：

```powershell
.\"启动 TurboForge.bat" --check
```

结果在 `.turboforge/deployment.json`，日志在 `.turboforge/logs/`。

从源码启动时，先准备 Python 环境，再用整合包里的引擎，或设置 `TURBOFORGE_NINFER_BIN` 指向引擎目录：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe scripts\start_web.py
```

## 使用：六步到本机 API

1. **选模型**  
   填本机目录、具体 GGUF 文件，或 Hugging Face 的 `组织/模型名`。完整 HF 目录留空「原始模型来源」。Qwen3.5 dense、hidden size 1024 的单文件 Q8_0 GGUF 可直接用；其他 GGUF 需要配上原始模型目录或仓库。

2. **检查**  
   页面读取架构、大小、分片和量化方式，同时看显卡型号、计算能力和显存。当前组合可用才会继续。

3. **选量化**  
   选 `auto`，或从列表里指定一个方案。列表只含已适配配方。有多种 NVFP4 时需要你自己选一种。已经量化好的权重只导入，不再假装重新高精度编辑。

4. **转换**  
   首次会下载并校验依赖。产物、日志和记录在 `.turboforge/builds/`。大模型请预留内存和临时磁盘，不要只按最终文件大小估算。

5. **验收**  
   通过后表示 GPU 能加载。质量门禁和测速分开显示：测了才有数字。已有 Qwen3.5-0.8B GPTQ 4-bit 记录：PPL 20.6995 → 21.7993（+5.31%），三次 64 token 生成平均 21.27 tok/s。

6. **启动**  
   打开构建目录里的 `scripts/run.bat`。再用下面的地址确认模型服务：

```powershell
curl.exe http://127.0.0.1:18080/v1/models
```

端口以结果页为准。默认模型端口来自 `.env` 里 `TURBOFORGE_AI_ALLOWED_LOCAL_ORIGINS` 的第一个地址，模板是 `18080`。网页端口 `8090` 和模型端口不是同一个。

调用示例：

```powershell
curl.exe http://127.0.0.1:18080/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d "{\"model\":\"<上一步返回的 id>\",\"messages\":[{\"role\":\"user\",\"content\":\"用一句话解释 KV cache\"}]}"
```

## 转换之后用哪套引擎

| 你的模型 | 建议 | 文件 |
| --- | --- | --- |
| Dense | [ninfer-fusion-kvmem](https://github.com/1314521gjy/ninfer-fusion-kvmem)：显存放 KV 窗口，其余放内存 | `.ninfer` |
| MoE，且是 RTX 50（计算能力 12.0），模型为 Qwen3.8-Flash-Next 或 DeepSeek-V4.1-Flash | [Basalt](https://github.com/jesdga95/basalt) | `.basalt` |
| 其他 MoE，含 Qwen3.6-35B-A3B | [Strata](https://github.com/Niko1221/Strata) | GGUF |

页面会给出仓库链接和启动步骤。三种文件不能互相直接替换。数据中心 Blackwell（计算能力 10.0）用 Strata，不用 Basalt。

Qwen3.5-0.8B 这条独立 GGUF 路线的上下文上限是 1024 token，页面会自动填上。更长的上下文用 ninfer-fusion-kvmem 的 KV 分层。并发在启动服务时设置，先看显存够不够。

## 使用时注意

- HF 路线会保留已有的 MTP/DFlash2 文件；要真正跑这些加速头，用兼容的 NInfer 服务。
- Qwen3.5 新量化走 GPTQ 或 NInfer。已有的标准 AWQ 可以导入。
- 质量门禁默认用英文 WikiText。中文、代码、数学可以另备评测文本。
- 适合本机单人使用。要开放到公网，自己加鉴权和访问控制。

## 排错

| 现象 | 处理 |
| --- | --- |
| 窗口马上关掉 | 看窗口里的错误；再运行 `启动 TurboForge.bat --check` |
| 网页打不开 | 用窗口里打印的地址，不要死记 8090 |
| API 连不上 | 用结果页端口，先访问 `/v1/models` |
| 磁盘不够 | 在页面删除失败的构建；成功模型和原始模型会保留 |
| 换了目录后启动失败 | 删掉对应虚拟环境目录，重新双击启动，模型和构建结果留着 |

## 友情链接

- [ninfer-fusion-kvmem](https://github.com/1314521gjy/ninfer-fusion-kvmem)：Dense、长上下文
- [Strata](https://github.com/Niko1221/Strata)：MoE，覆盖 RTX 20/30/40/50
- [Basalt](https://github.com/jesdga95/basalt)：RTX 50 上已适配的 MoE
- [NInfer 上游](https://github.com/iamwavecut/ninfer-all)：转换与引擎来源

各仓库的许可证和模型列表以对方页面为准。没有许可证文件的 `ninfer-kvmem-actkv` 不在推荐里。
