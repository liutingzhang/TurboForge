# TurboForge Windows 干净安装包说明

本发布提供中文和英文文档及启动入口；两套安装包使用相同程序，当前界面及部分运行日志主要为中文。请选择一种语言的完整分包，不要混用语言、版本或旧安装目录。

## 1. 下载与校验

- 下载同一目录下所有 `TurboForge-Windows-x64-chinese-part*.zip` 和 `SHA256SUMS.txt`。
- 这是多个**独立 ZIP**，不是 `.z01` 分卷；每包都必须下载、解压。单个 ZIP 不能独立运行。
- 每个 ZIP 最大 1,800,000,000 字节，低于 GitHub Release 单附件必须小于 2 GiB（2,147,483,648 字节）的限制。
- 在下载目录打开 PowerShell，运行 `Get-FileHash .\*chinese-part*.zip -Algorithm SHA256`，逐个核对 `SHA256SUMS.txt` 中的文件名和 SHA-256。缺包或校验不一致时不要启动。

GitHub 限制说明：https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases

## 2. 解压并启动

将全部 ZIP 解压到同一个新的空目录，使文件合并到同一 `TurboForge` 文件夹。不要为每个 part 建立不同安装目录，不要在 ZIP 内直接运行，不要覆盖包含已有模型和任务的旧安装。

例如，在下载目录执行：

```powershell
Get-ChildItem '*chinese-part*.zip' | Sort-Object Name | ForEach-Object {
    Expand-Archive -LiteralPath $_.FullName -DestinationPath '.\installed' -Force
}
```

打开 `installed\TurboForge`，双击 **启动 TurboForge.bat**。英文入口为 **Start TurboForge.bat**。保持命令窗口开启，等待自动打开浏览器；若未打开，使用窗口打印的本机地址。默认端口为 8090，被占用时会另选空闲端口。

首次启动会在安装目录内建立独立 Python 环境，并联网安装 Web 依赖。原始发行包中不含个人 `.env`；使用帮助中心不需要 API Key。可选 AI 助手需由用户自行配置模型服务和凭据，不会携带发布者凭据。

## 3. 系统条件与首次下载

- Windows x64；不支持 Windows ARM64 或 32 位系统。请使用有写权限的普通目录，不要放在 Program Files。
- 无需预先安装 Python、Git、Node.js 或 CUDA Toolkit；包中保留固定 Python 运行时和预构建网页。
- 模型转换与 GPU 验证需要受支持的 NVIDIA 显卡、足够显存和正确驱动；驱动不能由本包替代。RTX 50 系列及 HF GPTQ 路线需要 NVIDIA 580 或更新驱动。
- 首次启动和首次选择转换路线仍需联网。HF GPTQ 的独立环境、PyTorch 等依赖可能下载数 GB；模型另行准备，不包含在包中。本包不是完全离线版。
- 建议至少预留 15 GB 用于基础安装和环境；模型、量化依赖、中间权重及质量对照还需额外空间，不能以 ZIP 大小估算全部需求。

只检查环境、不转换模型：

```powershell
.\"启动 TurboForge.bat" --check
```

## 4. 支持边界

- 先选择模型，再点击“检查模型并继续”，只从界面提供的可用方案中选择；没有通过检查的路线不会被自动替换。
- 小型 Qwen3.5 的 NInfer 路线需要额外适配器；本包复用的上游运行时不包含该适配器。完整原始 HF 模型可选择已接入的 HF GPTQ；不能把 GGUF 直接当作原始 HF 权重。
- MTP/DFlash2 被检测到不代表当前引擎可以执行。无法证明保留或缺少组件的情况会阻止转换；HF 路线需按提示明确关闭推测解码。
- 质量门禁仅验证对应语料上的指标，不保证全部模型能力无损。真实大模型 GPU 转换不属于本次分包启动验收的覆盖范围。
- 使用全新 Windows 虚拟机和所有显卡组合的兼容性不能由当前机器上的测试替代；以实际自检和构建报告为准。

## 5. 包内内容与隐私边界

保留 `src` 后端源码、`web/src` 前端源码和构建配置、`scripts` 启动脚本、`runtime/bin` 引擎及必要 DLL、`runtime/source` 对应转换器源码、`runtime/python` 固定解释器归档、`runtime/web` 网页运行文件、许可证、双语说明和校验清单。网页运行文件不是个人数据，保留它们可避免首次启动安装 Node.js。

不分发 `dist`、`.env`（包括示例环境文件）、`.git`、`.github`、开发缓存、`node_modules`、测试/工作树/旧发布目录、个人配置、模型、任务、日志或聊天缓存。依赖许可证按需保留。`runtime/source/.turboforge-source.json` 仅记录公共上游版本来源，不含用户配置。

运行后新生成的 `.turboforge` 是**安装用户自己的**环境、任务和日志，不是随包携带的发布者数据。请勿将其连同个人凭据、模型重新上传到 GitHub。

## 6. 故障处理

- 缺少脚本、DLL、网页或提示清单损坏：先核对所有 ZIP 的 SHA-256，再向新的空目录完整解压全部分包。
- 首次安装失败：检查网络、代理、磁盘、写入权限及安全软件，查看 `.turboforge/logs` 中的启动日志；不要全局关闭安全防护。
- 显卡或驱动不支持：更新适合本机的官方 NVIDIA 驱动；不要把“网页能打开”理解为“任意模型都能转换”。
- 页面无响应：确认启动窗口仍然开启，使用其打印的端口，不要假定永远是 8090。
- 提交问题前删除日志中的用户名、路径、访问令牌和私人模型信息，不要提交 `.env` 或助手配置。
