"""Public static facts derived from executable capability rules, never page text."""
from __future__ import annotations

from ..backends.hf_quantization import (
    AWQ_DISABLED_MODEL_TYPES, AWQ_DISABLED_REASON, GPTQMODEL_VERSION,
    HF_ACCEL_VERSIONS, HF_CORE_VERSIONS, HF_DEFAULT_BACKEND,
)
from ..evaluation import AUTO_EVALUATION_TOKENS, WIKITEXT_REVISION


def capability_entries() -> list[dict]:
    source = ["src/turboforge/backends/hf_quantization.py"]
    return [
        {"id": "capability.local_model_input", "title": "本机模型路径有效却找不到、缺少 config.json 与原始模型来源怎么填",
         "body": "本机路径指运行 TurboForge 服务的这台电脑。点击选择文件夹/选择 GGUF 文件可填入路径，也可手动粘贴；支持 Windows 复制文件地址的成对引号与首尾空格，目录内部空格保留。路径存在不等于模型资料完整或转换路线受支持。完整原始 HF 检查点填在“模型”，其“原始模型来源”应留空；GGUF 填具体文件；已支持的自包含文件自动生成资源，其他路线需匹配的 config.json 与 tokenizer.json/tokenizer.model，可来自独立的本机原始 HF 目录或明确原始仓库。不要把仅有 GGUF 的文件夹再次当作原始来源；GGUF 导入只需元数据时不要求原始 Safetensors。Qwen3.5 dense Q8_0、hidden_size=1024 且内嵌元数据完整的单文件 GGUF 可直接生成 config.json、分词器和聊天模板，不需原始目录；其他来源不明文件不能按文件名猜配置。没有对应导入路线时，改用原始高精度模型作为待处理模型重新检查，不能仅补 config.json 假装支持；不支持的硬件或模型应明确停止。",
         "topic": "input", "aliases": ["本机模型无效", "路径没问题", "本地目录", "找不到config.json", "原始模型来源怎么填", "无需上传不是无需填写"],
         "source_refs": ["src/turboforge/model_sources.py", "src/turboforge/inspector.py", "src/turboforge/precheck.py", "src/turboforge/preparation.py", "web/src/App.tsx", "web/src/components/PathInput.tsx"]},
        {"id": "capability.clean_gguf", "title": "空白环境只有一个 GGUF，原始模型目录从哪里来",
         "body": "发行包不含模型，但已支持的单文件 GGUF 不需要开发者本机原始模型目录。当前 Qwen3.5 dense Q8_0、hidden_size=1024 的完整 GGUF 可从内嵌结构参数、词表、BPE merges、特殊词元及聊天模板在缓存 gguf-resources 中生成前端资源，直接以 GGUF 为唯一权重源转换。选择 GGUF 后原始模型来源留空；旧版误填该 GGUF 所在目录且其中无 config.json 时也可识别此路线。配方 qwen3_5_gguf_standalone 解码后重编码为 Q8_0，非逐字节无损，质量需单独验证；现有适配引擎上下文上限 1024，引导检查后自动应用。缺字段、重复词元或带未验证 MTP 的文件明确停止，不猜配置或静默丢头。其他架构/精度/分片不属于当前独立导入验证范围。首次运行仍需准备依赖与适配引擎；已实测这份 0.8B Q8_0 的生成资源、转换、GPU 加载和文本生成，未实测空白电脑完整部署。",
         "topic": "input", "aliases": ["空白环境只下一个gguf", "干净电脑", "没有原始模型目录", "0.8B GGUF导入", "config.json补齐就能用吗"],
         "source_refs": ["src/turboforge/gguf_standalone.py", "src/turboforge/model_sources.py", "src/turboforge/precheck.py", "src/turboforge/preparation.py", "src/turboforge/bootstrap.py", "src/turboforge/backends/registry.py"]},
        {"id": "capability.auto_quality", "title": "HF 自动质量门禁与独立评测语料",
         "body": f"引导模式 HF 新量化默认启用质量门禁，无需手动提供评测文件。默认使用 WikiText-2 test 固定版本 {WIKITEXT_REVISION}，最多 {AUTO_EVALUATION_TOKENS} token；校准用 train，记录哈希与评分数。参考为本次处理后、量化前权重的 CPU FP32 PPL，需额外 RAM/时间。NInfer 仍需匹配对照产物与语料；packed 导入不能用自身证明质量。通用英文 PPL 不保证所有任务效果。",
         "topic": "quality", "aliases": ["质量门禁 评测文件 自动评测 独立语料"],
         "source_refs": ["src/turboforge/evaluation.py", "src/turboforge/domain.py", "web/src/App.tsx"]},
        {"id": "capability.offline_start", "title": "没网络还能启动吗，离线运行需要什么",
         "body": "首次启动需联网安装缺少的 Python/网页依赖；转换时还需下载未缓存的模型、环境与校准/评测集。已准备完整后可使用本地模型和离线帮助。远程 AI 问答仍需连接所配置服务，不能承诺未准备的电脑断网开箱即用。",
         "topic": "start", "aliases": ["离线 启动 断网 没网络"], "source_refs": ["scripts/start_web.py", "src/turboforge/bootstrap.py"]},
        {"id": "capability.awq_restrictions", "title": "Qwen3.5 AWQ 支持与停用原因",
         "body": f"GPTQModel {GPTQMODEL_VERSION}：{AWQ_DISABLED_REASON} 停用 model_type：{', '.join(AWQ_DISABLED_MODEL_TYPES)}。其他架构也不能据此宣称质量已实测。",
         "topic": "quantization", "aliases": ["Qwen3.5可以做AWQ吗", "qwen3_5 AWQ", "AWQ选项不见了"], "source_refs": source},
        {"id": "capability.hf_backend", "title": "GPTQ AWQ 默认推理后端与环境依赖",
         "body": f"当前默认后端 {HF_DEFAULT_BACKEND.upper()}，由 GPTQModel 为当前设备与布局选择可用内核；不保证每个模型都更快。Torch 为显式兼容选项。依赖版本：GPTQModel {GPTQMODEL_VERSION}，{HF_CORE_VERSIONS}；Windows 加速依赖：{HF_ACCEL_VERSIONS}。脚本自动准备独立环境，附带受支持 Python 版本的 PyPcre wheel，无需用户手工安装 C++ 编译器。",
         "topic": "resources", "aliases": ["GPTQ默认用什么后端", "安装需要自己装编译器吗", "AUTO Torch Triton FLA", "PyPcre torchvision"], "source_refs": source},
    ]


def public_capabilities() -> dict:
    return {entry["id"]: entry["body"] for entry in capability_entries()}
