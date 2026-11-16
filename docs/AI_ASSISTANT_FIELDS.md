# AI 助手字段与状态知识

日期：2026-11-14。运行帮助由 `build_assistant_knowledge.py` 编译。字段别名与页面对应；新增字面量 Field 未覆盖时生成检查失败。动态处理参数从 `api/meta.py` 的参数声明提取帮助与默认值，不重复维护。

| ID | 字段别名 | 含义与影响 | 默认来源与约束 | 主题 |
|---|---|---|---|---|
| field.model | 模型、输入模型 | 选择完整 HF/NVFP4 目录或 GGUF 文件，也可给明确 HF 仓库；决定后续架构识别与路线 | 指运行服务的电脑；支持复制地址的引号/首尾空格。完整 HF 只填此字段，GGUF 选具体文件；路径存在不等于格式受支持；路径不发送给外部助手 | input |
| field.original | 原始模型来源、原始 HF 目录 | 为已有量化来源补齐匹配配置、分词器或路线要求的高精度张量 | 默认显示；完整 HF 模型应留空；缺失来源时必填。已支持自包含 Qwen3.5-0.8B Q8_0 GGUF 自动生成配置/分词器/模板，无需来源；传统 GGUF 来源目录需 config.json 和 tokenizer.json/tokenizer.model，不能选仅含 GGUF 的目录；支持明确仓库链接；不按文件名猜来源、不依赖开发者本机目录 | input |
| field.gguf | 仓库 GGUF 文件 | 多精度仓库中选择具体文件与分片，输入精度随文件决定 | 用户明确选择，不自动猜精度 | input |
| field.output | 输出目录 | 保存新产物、计划、日志与报告；中间文件可能占更多空间 | 自动分配或用户选择原模型外新空目录；不覆盖源权重 | resources |
| field.target | 目标位宽、量化方案、量化目标 | 直接自选量化目标：方案选项写明位宽、算法与引擎，位宽下拉只是按主权重精度筛选;不再有「使用偏好」替用户暗选方案 | 候选＝当前模型＋引擎已验证配方，未适配位宽不列出故菜单可能较短;自动推荐是可用默认、遵循模型默认策略，不是最优保证;低 bit 不保证更快或质量不变 | quantization |
| field.ternary | 已发布三值 GGUF | 匹配 Bonsai 2 的现成 PQ2_0 来源，不训练新三值模型 | 可选；启用该配方时必需，架构/几何必须匹配 | quantization |
| field.device | 转换设备 | 控制转换计算在 CUDA 或支持的 CPU 路线执行，不等于更换推理引擎 | 来自 meta.defaults.device；CPU 路线未统一测速 | resources |
| field.requantization | 允许近似再量化 | 表达用户对已有量化近似解码再量化损失的明确同意；不会恢复丢失信息 | 默认关闭；不能解除未适配格式/架构等真实限制 | quantization |
| field.jailbreak | 越狱方式 | none 保留现有权重行为；Abliteration/Heretic 改写支持的高精度权重，可能影响任务质量 | 默认 none；导入量化不能叠加处理，Heretic 需交互保存 | calibration |
| field.calibration_file | 校准语料 | AWQ/GPTQ 估计量化统计所用 UTF-8 文本；代表性影响效果和耗时 | 留空由当前固定版本准备 WikiText-2 train；不与评测数据混用 | calibration |
| field.group | 分组大小 | 一组权重共享量化参数，影响元数据开销、统计与兼容性 | 当前新校准默认 128，可选 32/64/128；已有 packed 布局按自身契约 | calibration |
| field.samples | 校准样本数 | 校准采样数量，增加可能增加时间、内存和数据覆盖，不保证单调改善质量 | 当前 HF 默认 128，允许范围由后端校验 | calibration |
| field.sequence | 每样本 token | 每个校准样本的长度，影响激活规模和可用语料 | 当前 HF 默认 512，不能超过模型实际窗口；不足样本不静默换算法 | calibration |
| field.backend | HF 推理后端 | torch 使用兼容计算；auto 由 GPTQModel 选择，部分优化内核需额外编译工具 | 默认 torch；HF 独立环境，不把量化 packed 交 NInfer | serving |
| field.speculative | 推测解码 | 使用兼容已有 MTP/DFlash2 组件；关闭执行仍保留应保留的头 | 默认 auto；缺组件时显式请求拒绝。HF 不执行头，带头 auto 会阻止路线 | speculation |
| field.context | 服务上下文 | 输入、生成与缓存使用的 token 窗口，增大会提高显存压力 | 从运行 profile 取有效值；几何/引擎上限不能仅靠输入大数字突破 | serving |
| field.concurrency | 最大并发 | 同时请求数量，影响缓存/计算资源与吞吐 | 取 meta/runtime 默认；HF 生成当前要求 1 | serving |
| field.port | 服务端口 | 当前本机模型 API 的监听端口，与页面/外部助手服务地址不同 | 默认取 .env 中 TURBOFORGE_AI_ALLOWED_LOCAL_ORIGINS 第一个 origin 的端口（模板 18080）；1–65535 且不被占用；修改按现有计划/脚本流程生效 | serving |
| field.smoke | 转换后冒烟验证 | 真实 GPU 加载并做短生成，提供可运行证据，不替代全面质量评测 | 默认启用；HF 不允许跳过。converted 不能当作 validated | quality |
| field.benchmark | 测试速度、转换后测试速度（质量评测可选） | 运行当前路线支持的真实速度/可选 PPL 测试，增加时间；关闭时未测就是未知 | 由当前用户选择/引导模式默认；门禁依赖对应评测 | performance |
| field.repetitions | 基准重复次数 | 多次测试有助于观察波动，也增加耗时 | 当前默认与范围取 benchmark/profile；HF 有次数限制，不默认无限重跑 | performance |
| field.kv | 基准 KV 精度 | 只影响测试缓存/运行数值与显存，不改变主权重编码 | 自动跟随几何；bf16/rk4v4 的速度与质量需按对应引擎验证 | performance |
| field.eval_text | 评测语料、困惑度文本 | 在指定 UTF-8 文本上评分；HF 新量化可留空自动使用固定 test 集，最多 2048 token | 与校准 train 保持隔离；NInfer 需要手动语料与对照，完整路径默认排除 | quality |
| field.eval_manifest | 困惑度语料 manifest | 使用已支持的评分语料描述而非普通自由文本 | 与困惑度文本按现有契约选择，必须有效且版本一致 | quality |
| field.baseline | 对照模型、基线产物 | 相同来源/分词器/处理阶段的对照用于衡量差异 | NInfer 用户提供已验证对照；HF 当前用本次处理后高精度 CPU FP32 参考；packed 不能拿自己证明无损 | quality |
| field.ppl_gate | 质量门禁 | 要求 PPL 比较通过才允许直接启动；测量失败也不能伪装通过 | 引导模式 HF 新量化默认开启，可明确关闭；NInfer 按用户选择，不能代替代码/数学/长文本评测 | quality |
| field.ppl_threshold | 允许 PPL 增幅 | 相对增幅 `(当前/对照 - 1)×100%` 的用户接受阈值，不是官方安全保证 | 页面常见初值 10%，由用户任务要求决定；不能改 manifest 绕过失败 | quality |
| field.prompt_lengths | 预填充长度列表 | 基准输入 token 长度列表，影响预填充负载与测量可比性 | 来自 profile/benchmark，逗号分隔有效长度，受几何上限约束 | performance |
| field.generation_lengths | 生成长度列表 | 基准输出 token 数列表，影响解码时间与测试口径 | 取当前 profile/用户值，HF 不支持任意多个生成长度 | performance |
| field.ppl_context | 困惑度上下文 | PPL 评分窗口长度，不等于量化位宽或训练长度 | 受模型实际窗口限制；对照必须使用相同有效值 | quality |
| field.ppl_stride | 困惑度步长 | 评分滑窗前进步长，影响重叠与评分数量 | 大于 0 且小于实际 PPL context；对照条件一致，不能重复计数冒充新语料 | quality |
| field.bench_ids | 基准语料 ids | 特定工具使用的 token ID 语料，用来生成可复现负载 | 采用当前兼容格式；平铺语料接受率不代表真实用户请求 | performance |
| field.upstream | NInfer 转换器源码目录 | 包含 tools/convert 的匹配源码，是转换程序来源 | 自动准备或用户选择固定版本；版本不同需要核实兼容性 | maintenance |
| field.engine | NInfer 引擎目录 | 包含 ninfer.exe/ninfer-serve.exe 与 DLL，影响实际加载/生成支持 | 自动准备或选匹配整合包；支持转换不等于引擎支持运行 | maintenance |

## 构建与服务状态

| ID | 名称 | 含义与下一步 | 是否可直接启动 |
|---|---|---|---|
| state.idle | idle / created / starting / planned | 等待、开始或已规划；尚未证明产物可运行，查看下一阶段 | 否 |
| state.preparing | preparing | 下载/安装和准备资料，查看当前准备日志 | 否 |
| state.processing | jailbreaking / jailbroken / converting | 正在权重处理或转换；完成处理不等于完成 GPU 验证 | 否 |
| state.converted | converted | 已生成产物，本次未完成对应 GPU 验证 | 否 |
| state.validating | validating / benchmarking | 正在验证或测量；等待真实结果，不把进度当成绩 | 否 |
| state.validated | validated | 本次构建验证通过；仍须满足任务完成与质量门禁、产物/组件检查等后端条件 | 以实际后端启动门禁为准 |
| state.blocked | blocked / failed / crashed / interrupted | 被规划阻止、构建失败或中断；查看具体诊断/阶段，不能改状态文件绕过 | 否 |
| state.validation_failed | validation-failed | GPU 加载/生成验证失败，检查实际 smoke 日志 | 否 |
| state.quality_rejected | quality-rejected | 当前质量要求未通过或评测失败；查看来源、语料、对照和实际指标 | 否 |
| state.server_starting | server: starting | 进程正在加载，健康检查尚未通过，查看 serve.log | 等待 ready |
| state.server_ready | server: ready | 本地模型健康检查通过，下一步仍要做一次实际文本请求 | 已运行，不等于任务质量保证 |
| state.server_stopped | server: stopped / exited | 未运行或进程退出；确认当前构建验证与服务错误 | 未运行 |
