# AI 助手诊断与运行异常内容目录

日期：2026-11-14。状态：知识初稿。配套[改造方案](AI_ASSISTANT_REFACTOR_PLAN.md)、[120 项用户问题](AI_ASSISTANT_KNOWLEDGE.md)。

本文固定诊断码取自当前 `src/turboforge/**/*.py` 中 `Diagnostic(...)` 的字符串字面量参数，经 AST 提取后核对。它不等于全部运行异常集合：很多下载/校准/生成失败通过 `ValueError / RuntimeError / OSError` 或第三方返回内容传播，需另按第二部分分类。

## 1. 固定诊断码：含义与操作

严重性以本次后端实际返回的 `severity` 为准；同一条解释不得把 warning 自动升级为 error，也不得把 error 降级为建议。自动选择和识别成功属于信息，不在 UI 中称为“报错”。动态 message、alternatives 中的任务具体数值与建议应在脱敏后保留。

### 输入格式与完整性

| Code | 新手解释 | 下一步 | 来源 |
|---|---|---|---|
| `input-unreadable` | 模型路径或内容读不了 | 核对本机路径、访问权限与文件完整性 | `planner.py` |
| `empty-model` | 目录没有可用的 Safetensors 权重 | 选择完整模型目录，检查下载是否结束 | `adapters/safetensors.py` |
| `duplicate-tensor` | 多个分片声明了重复权重，来源可能混杂 | 按相同 revision 重新核对分片，不直接删除未知张量 | `adapters/safetensors.py` |
| `config-missing` | 缺少架构配置 | 补齐与权重匹配的 config.json | `adapters/safetensors.py` |
| `tokenizer-missing` | 缺少分词器 | 获取同一来源的 tokenizer 资料，不能用随机模型的代替 | `adapters/safetensors.py` |
| `gguf-recognized` | 已识别 GGUF；不是错误 | 查看其真实编码、分片与可用导入路线 | `adapters/gguf.py` |
| `gguf-source-invalid` | 附加 GGUF 无法读取或解析 | 检查具体文件、分片、格式完整性，保留底层错误 | `planner.py` |
| `gguf-secondary-source` | 手动路线应将 GGUF 作为附加来源，并补原始元数据 | 使用自动引导核实来源，或指定原始目录与 gguf 来源字段 | `planner.py` |
| `nvfp4-secondary-source` | NVFP4 应通过量化附加来源导入 | 补匹配原始资料，并设置 quantized 来源 | `planner.py` |
| `nvfp4-name-is-not-evidence` | 名字含 NVFP4 但没有完整格式证据 | 以实际配置、packed 张量和 scale 为准，不改名绕过 | `adapters/safetensors.py` |
| `nvfp4-layout-unverified` | 当前 NVFP4 布局没有通过已支持契约验证 | 获取完整且受支持的检查点/生产者布局 | `adapters/safetensors.py` |
| `nvfp4-information-loss` | NVFP4 已经丢失部分原始数值信息 | 接受当前来源的精度边界；需要原始精度应取得原权重 | `adapters/safetensors.py` |

### 量化、来源与输出

| Code | 新手解释 | 下一步 | 来源 |
|---|---|---|---|
| `no-verified-recipe` | 当前架构/几何没有已接入转换方案 | 查看当前支持列表，换支持模型；不猜一个相近配方 | `planner.py` |
| `target-incompatible` | 指定方案与模型不匹配 | 回可用选项重新选择；AI 不擅自替换明确选择 | `planner.py` |
| `no-runnable-target` | 有配方，但所需来源未提供 | 按缺失列表补齐 ternary/gguf/quantized/imatrix 等资料 | `planner.py` |
| `recipe-source-missing` | 本次配方缺必要附加来源 | 补当前配方明确要求的资料，再预览计划 | `planner.py` |
| `source-missing` | 填写的来源路径不存在 | 重新选择正确文件/目录，确认盘符挂载 | `planner.py` |
| `packed-needs-hf` | 已有 AWQ/GPTQ 不能直接交给 NInfer | 选择 HF packed 保留路线或从高精度重新开始 | `planner.py` |
| `nvfp4-requantization-loss` | 从已有低精度转其他量化存在额外损失 | 优先原始高精度；只有实际支持的近似路线才谈显式同意，不能用开关解除所有布局限制 | `planner.py` |
| `output-overlaps-input` | 输出与原模型目录重叠，存在覆盖风险 | 选择原模型之外的新空目录 | `planner.py` |
| `output-not-empty` | 输出目录已有文件，程序拒绝覆盖 | 新建输出目录，保留已有产物 | `planner.py` |
| `auto-selected-bonsai2` | 已有匹配三值来源，自动选了 Bonsai | 展示实际 recipe 与来源；不是本次训练了三值模型 | `planner.py` |
| `auto-selected-gguf` | 自动沿 GGUF 原精度导入 | 解释精度由输入决定，不承诺重新变 bit | `planner.py` |
| `auto-selected-nvfp4` | 自动选了可用 NVFP4 导入 | 仍需匹配实际 producer 布局，有歧义按页面明确选择 | `planner.py` |
| `auto-selected-gguf-bridge` | 当前几何采用 GGML Q8_0 桥接默认 | 说明原生固定几何内核限制，不推广成全模型默认 Q8 | `planner.py` |
| `auto-selected-lowest-bpw` | 当前候选中按默认规则选较低 bpw | 展示本次候选和质量/资源取舍，不声称最佳质量 | `planner.py` |

### 环境与资源

| Code | 新手解释 | 下一步 | 来源 |
|---|---|---|---|
| `engine-missing` | 选定引擎目录缺少必需可执行文件 | 使用匹配整合包或重新准备当前引擎 | `planner.py` |
| `upstream-missing` | 转换器源码不在指定位置 | 自动准备或在高级配置选择匹配源码目录 | `backends/compiler.py` |
| `upstream-revision-differs` | 实际转换器与已核对版本不同 | 查看 revision 差异，优先匹配版本；不声称未测版本已兼容 | `backends/compiler.py` |
| `upstream-revision-unknown` | 无法确认转换器版本 | 检查来源标记/版本，记录为未知，不自动视作匹配 | `backends/compiler.py` |
| `torch-missing` | 当前转换环境缺 torch | 用一键准备并检查实际 worker 环境 | `backends/compiler.py` |
| `numpy-missing` | 当前转换环境缺 numpy | 用一键准备修复对应环境 | `backends/compiler.py` |
| `torch-cuda-unavailable` | 当前 torch 不能使用 CUDA | 检查实际 Python、Torch CUDA 构建、驱动与设备 | `planner.py` |
| `device-unverified-cpu` | 选择 CPU 转换，但当前没有该路线速度证据 | 可解释预计较慢，不能给出编造耗时，也不代表 CPU 推理已支持 | `planner.py` |
| `ram-profile-unverified` | 此路线没有可靠转换 RAM 峰值模型 | 首次观察实际使用，不把未知填成零或“足够” | `resources.py` |
| `disk-insufficient` | 临时磁盘需求超过可用空间 | 换空闲更多的输出盘；不自动删除用户资料 | `resources.py` |
| `ram-insufficient` | 已建模的系统内存需求超过可用量 | 释放 RAM、换小模型或已验证低内存路线，不能只承诺降 bit 有效 | `resources.py` |
| `gpu-not-detected` | 无法查询 NVIDIA 显存 | 检查显卡/驱动及当前运行环境，区分转换和推理阶段 | `resources.py` |
| `vram-insufficient` | 预计运行显存可能放不下 | 结合上下文/KV/并发、较低 bpw 和其他占用调整；不是已测转换峰值 | `resources.py` |
| `context-capped-gdn-geometry` | 当前适配引擎限制了该几何的上下文 | 显示后端实际有效上限，长文需求换兼容方案，不能仅改数值突破内核 | `planner.py` |

### 权重处理与校准

| Code | 新手解释 | 下一步 | 来源 |
|---|---|---|---|
| `jailbreak-unknown` | 处理方式名称无效 | 选择 none/heretic/abliteration 中当前可用项 | `planner.py` |
| `jailbreak-precision` | 当前精度不支持高精度权重处理 | 取得受支持原始 BF16/FP16/FP32 或选择不处理 | `planner.py` |
| `jailbreak-not-applied-to-imported-weights` | 编辑 HF 基座不会作用到导入量化张量 | 导入已处理来源并选 none，或先处理高精度再量化 | `planner.py` |
| `direction-file-missing` | 指定方向文件不存在 | 重新选择匹配向量，或使用可用自动标定 | `planner.py` |
| `calibration-unavailable` | 自动方向标定缺 transformers 等条件 | 修复正确环境，或提供匹配已有方向 | `planner.py` |
| `heretic-needs-high-precision` | Heretic 需要高精度输入 | 使用匹配高精度源，不对 packed 输入强制执行 | `jailbreak/processors.py` |
| `heretic-needs-safetensors` | 当前输入容器不适合 Heretic 处理 | 选择完整受支持 Safetensors 模型目录 | `jailbreak/processors.py` |
| `heretic-unavailable` | 当前环境没有可用 heretic-llm | 按现有安装说明准备，或选择可用处理方式 | `jailbreak/processors.py` |
| `heretic-needs-desktop-console` | 当前环境无法完成所需桌面交互 | 在支持的 Windows 桌面环境使用高级模式，或选择可无人值守方式 | `jailbreak/processors.py` |
| `heretic-interactive-save` | 需要在外部窗口操作并保存处理权重 | 按提示将完整权重保存到约定目录，等待真实完成 | `jailbreak/processors.py` |
| `abliteration-needs-high-precision` | Abliteration 需要支持的高精度权重 | 换高精度来源或关闭处理 | `jailbreak/processors.py` |
| `abliteration-needs-safetensors` | 当前格式不能由该处理器改写 | 使用完整 HF Safetensors 目录 | `jailbreak/processors.py` |
| `abliteration-unavailable` | 处理依赖不齐 | 检查该 worker 的 torch/numpy 等实际依赖，不只检查系统环境 | `jailbreak/processors.py` |
| `abliteration-calibration-source` | 正在使用指定/默认提示采集方向 | 说明方向来源和处理阶段，不把标定成功等同所有任务质量保证 | `jailbreak/processors.py` |

### 推测组件与 HF 路线

| Code | 新手解释 | 下一步 | 来源 |
|---|---|---|---|
| `speculative-component-missing` | 显式请求了没有检测到的兼容组件 | 选 auto/关闭，或提供支持的匹配来源；不训练/伪造头 | `planner.py` |
| `speculative-would-be-dropped` | 当前路线无法保留输入自带组件 | 更换能保留的路线；不能静默丢头完成 | `planner.py` |
| `speculative-inspection-unverified` | 已有权重头部无法读取，不能确认是否包含 MTP/DFlash2 | 核对同一版本的完整分片并重新检查；不能假定没有推测头继续转换 | `inspector.py` |
| `hf-native-runtime` | 输出走独立 HF 引擎，不是 NInfer | 使用 HF 启动流程；MTP/DFlash2 不执行，新路线未完成 GPU 实机验收 | `backends/hf_quantization.py` |
| `hf-configuration` | HF 参数/来源/布局校验未通过 | 展示该次具体 message，再映射到第二部分对应类别，不只说“改成自动” | `backends/hf_quantization.py` |

最后四项分别来自 `planner.py`（前两项）及 `backends/hf_quantization.py`（后两项）。将其转换为运行条目时补齐源代码引用。

## 2. 不带固定 Diagnostic code 的运行异常

以下分类是**拟新增知识 ID**，不是现有后端统一错误码。实施时保留真实异常原因和 request ID，逐步在 API 边界提供稳定 code；不要靠中文整句匹配推导所有原因。

| 拟定知识 ID / 场景 | 现有来源/典型现象 | 应回答什么与下一步 |
|---|---|---|
| `runtime.model.not_found` | precheck/preparation：本机不存在且非有效仓库 | 支持成对引号及首尾空格；确认路径属于运行服务的电脑，选择存在的目录/文件或有效 HF 仓库 |
| `runtime.model.source_unknown` | 孤立 GGUF/元数据无原始来源，或来源不是完整仓库 | 用户补可核实原始 HF 仓库或含 config.json 与分词器的目录；仅含 GGUF 的目录不合格；无对应尺寸导入路线时改用原始高精度模型，不能只补配置 |
| `runtime.model.incomplete` | 缺配置、分词器、权重/索引/分片 | 列具体缺失项与 revision，从同一来源补齐 |
| `runtime.model.unsupported_geometry` | 下载大权重前拒绝不支持架构/几何 | 当前缺实现而非网络故障，换已支持输入；重复下载无助 |
| `runtime.model.engine_port_required` | 小模型需适配整合包 | 使用带相应适配引擎的版本，不承诺普通上游下载即具备 |
| `runtime.download.access` | HTTP 401/403、受限模型、权限/许可 | 核实合法权限，不索取 Key 或指导绕过 |
| `runtime.download.not_found` | 文件/revision 不存在 | 核对完整链接、明确版本与实际文件，分清 404 来源 |
| `runtime.download.network` | DNS/SSL/代理/超时/中断 | 说明网络阶段，保留安全错误；可复用缓存/续传，但不禁用 TLS 校验 |
| `runtime.download.integrity` | 校验失败或分片缺失 | 重新获取明确损坏项并重新校验，不跳过 SHA-256 |
| `runtime.fs.permission` | 输出、缓存或任务记录不可写 | 核对空间/当前用户权限，选择可写独立目录，不默认提权 |
| `runtime.hf.layout` | GEMV/Marlin/BitBLAS/额外激活/混合位宽等未接入 | 提供标准兼容布局或原始高精度，不冒充 GEMM |
| `runtime.hf.packed_integrity` | qweight/scales/qzeros/g_idx dtype、shape、数值、索引异常 | 检查完整检查点与量化配置，不擅自补值或去掉检查 |
| `runtime.hf.config_conflict` | 不同量化配置文件或方法声明冲突 | 使用同一来源一致检查点，明确冲突字段，不任选一个吞掉错误 |
| `runtime.hf.calibration` | 样本不足、长度超窗、文本缺失、校准失败 | 补有效语料/减少长度或按支持参数调整，不 fallback 到另一算法 |
| `runtime.hf.calibration_eval_overlap` | 评测与校准数据相同或重叠 | 使用独立评分语料，保证记录可追溯，不能关闭校验后宣称质量可靠 |
| `runtime.hf.parameters` | group、backend、并发、context、PPL stride、重复次数等不合法 | 展示实际字段和值与允许范围，定位控件；packed 原 group 可有 -1，不与新校准的 32/64/128 混同 |
| `runtime.hf.environment` | 独立环境缺失、安装失败、CUDA/导入自检失败 | 查看 `.turboforge/hf-env` 实际错误；当前代码固定版本/驱动要求按本包说明，不任意升级已有环境 |
| `runtime.hf.speculation` | HF 不执行头，auto 带头被阻止 | 要执行选支持的 NInfer；只归档需用户明确关闭推测，不能自动丢头 |
| `runtime.weight.interactive` | Heretic 未保存、交互被关、超时 | 核对外部窗口与约定产物，保存成功后按真实流程继续 |
| `runtime.source.imatrix` | 矩阵没有匹配向量 | 用本模型适配矩阵，不把普通文本填成 imatrix |
| `runtime.source.conflict` | 主输入与附加来源互相冲突 | 重新确定唯一路线与对应来源，保留用户明确选择，不静默覆盖 |
| `runtime.convert.process_exit` | 转换器非零退出/崩溃 | 定位 convert.log 首个相关错误，区分资源、格式与引擎；不只复述退出码 |
| `runtime.convert.no_artifact` | 进程结束却没生成预期产物 | 构建失败，不能因退出码 0 就填成功；检查输出/转换器接口 |
| `runtime.convert.oom` | 转换/校准 RAM 或 CUDA OOM | 按阶段分析内存，选支持的低内存配置/小模型，保留真实资源限制 |
| `runtime.speculation.binding` | 头张量/连接/shape/物理范围/续卷校验失败 | 使用匹配完整组件或支持路线，不能跳过校验后显示已保留 |
| `runtime.validate.failure` | GPU smoke 失败、非有限 logits | 查看 smoke/worker 日志与引擎环境；有文件不等于可启动 |
| `runtime.quality.reference` | 对照来源、处理阶段、fingerprint/hash 不符 | 选同阶段真实对照，不能混用越狱前后或产物自身 |
| `runtime.quality.measurement` | PPL 工具缺失/失败、评分 token 不一致 | 质量未知并保留失败，修复测试条件，不填零损失 |
| `runtime.quality.threshold` | quality-rejected | 解释实际 PPL/相对增幅/阈值，选其他方案再验收，不手改状态 |
| `runtime.build.concurrent` | 已有构建/模型服务占用 | 等任务结束，或由用户手动停止明确所属服务，不自动杀未知进程 |
| `runtime.build.record` | unknown build id/记录结构不完整/中断 | 刷新实际状态并检查输出记录，不能把丢失记录认作成功 |
| `runtime.serve.validation_required` | 非本次 validated 不能启动 | 指向本次验证问题；converted 或跳过验证不能直接启动 |
| `runtime.serve.artifact_missing` | 模型已移动/删除、校验异常 | 核对实际产物或按现有流程重建；不能用空文件骗过存在检查 |
| `runtime.serve.port_conflict` | 指定端口已占用 | 确认归属后手动停相应服务，或按当前流程换端口重建计划 |
| `runtime.serve.not_ready` | starting、进程退出、健康检查未过 | 查看 serve.log、加载和显存；不把进程存在当 ready |
| `runtime.serve.request` | HF 不支持 tools/图片/参数，token 总量超限 | 用文本消息与支持参数，缩短输入/输出或更换实际支持服务 |
| `runtime.serve.busy` | HF 并发 1，另一个请求在处理 | 等待/停止自身请求，不承诺增加线程即扩展 GPU 并发 |
| `runtime.serve.generation` | 超时、生成错误、流中错误 | 保留已返回文本并显示未完成，查具体日志，不伪造完整答案 |
| `runtime.ai.config` | 拟新增：URL/模型/Key 缺失或无效 | 打开助手设置三项配置，说明实际生效来源 |
| `runtime.ai.auth_quota` | 拟新增：401/403/429 | 区分鉴权/权限/额度/并发，不把所有 429 都说成余额不足 |
| `runtime.ai.protocol` | 拟新增：404、非法 JSON/SSE、格式不兼容 | 检查接口根地址、模型 ID、文本协议和流式能力 |
| `runtime.ai.timeout` | 拟新增：连接/首包/总超时、取消/断流 | 保留问题草稿；由用户决定重试，不能流中自动重复扣费 |
| `runtime.ai.context` | 拟新增：413/上下文超限、选区过大 | 缩选区/日志/历史，明确舍弃内容；用户可新建会话 |
| `runtime.ai.security` | 拟新增：CSRF/URL/会话权限拒绝 | 说明当前部署策略与允许的安全配置路径，不把防护关掉当通用修复 |

## 3. 目录维护与验收方法

1. 用 AST 收集 `Diagnostic` 的字符串 code，去重后与本文 code 表/未来运行条目集合做差集；新 code 必须补正文与关联页面。不能只 grep 当前前端 `guidance.ts`，它只覆盖部分诊断。
2. `Diagnostic` 的代码引用可能被重构，实施时生成 source_refs，防止手写行号失效。本表列模块名，所有相对位置均位于 `src/turboforge/` 下。
3. 扫描主要边界的 raise/HTTP 错误/任务终态，按第二部分分类；此类异常不是有限静态 code 集，不声称完全覆盖第三方错误。
4. 每张运行帮助卡包含含义、适用阶段、证据、可操作步骤、何时需补资料；不能只有“请重试”。用户现场错误文本与日志先脱敏再引用。
5. 发布时将本表转换为可打包知识条目，验证诊断精确检索与 UI 就地解释；本次只是内容与覆盖范围交付，没有执行 LLM/GPU/浏览器验收。
