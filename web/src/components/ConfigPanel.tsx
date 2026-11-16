import { Collapse, Form, Input, InputNumber, Select, Switch } from 'antd';
import type { EnvMeta, ForgeConfig, ForgePlan, JailbreakMeta, Meta, ModelInspection, ParamSchema, Recipe } from '../api/types';
import Field from './Field';
import { Chip, Notice, StepHeader } from './ui';
import { bitsLabel, bpwLabel } from '../utils/format';
import PathInput from './PathInput';

interface ConfigPanelProps {
  stage: number;
  meta: Meta;
  config: ForgeConfig;
  inputPath: string;
  onInputPath: (v: string) => void;
  baseDir: string;
  onBaseDir: (v: string) => void;
  inspection: ModelInspection | null;
  routeInspection: ModelInspection | null;
  plan: ForgePlan | null;
  candidateRecipes: Recipe[] | null;
  disabled: boolean;
  onOutputTouched: () => void;
  onChange: (patch: Partial<ForgeConfig>) => void;
}

function paramRequired(p: ParamSchema, env: EnvMeta): boolean {
  if (p.required) return true;
  if (p.required_when_env_missing) {
    return !env[p.required_when_env_missing as keyof EnvMeta];
  }
  return false;
}

// The geometry-generic GGML Q8_0 bridge is NOT native groupwise quantization;
// it is the only route for small checkpoints whose hidden size the native
// Q4/Q5/Q6/Q8/NVFP4 kernels (compile-time locked to 27B/35B) cannot serve.
const isBridge = (r: Recipe) => r.storage_formats.includes('gguf_q8_0');

function RecipeOption({ recipe }: { recipe: Recipe }) {
  return (
    <div className="opt">
      <span className="opt-name">{recipe.label}</span>
      <span className="opt-meta">
        <span className="mono">{recipe.recipe}</span> · {bitsLabel(recipe.nominal_bits)} bit ·{' '}
        {bpwLabel(recipe.typical_bpw)}
        {recipe.extra_sources.length > 0 && (
          <Chip tone="warn">需要 {recipe.extra_sources.join('/')}</Chip>
        )}
        {recipe.needs_gpu && <Chip tone="info">GPU</Chip>}
        {isBridge(recipe) && <Chip tone="info">通用桥接</Chip>}
      </span>
    </div>
  );
}

export default function ConfigPanel({
  stage,
  meta,
  config,
  inputPath,
  onInputPath,
  baseDir,
  onBaseDir,
  inspection,
  routeInspection,
  plan,
  candidateRecipes,
  disabled,
  onOutputTouched,
  onChange,
}: ConfigPanelProps) {
  const needsBase = inspection?.input_kind === 'nvfp4' || inspection?.input_kind === 'gguf';
  // Prefer the plan's geometry-filtered recipes (the exact gate the planner uses)
  // so the dropdown never offers a target the plan would refuse, e.g. native 27B
  // kernels on a 0.8B checkpoint. Before a plan exists, fall back to the
  // architecture-matched meta list.
  const candidates = candidateRecipes
    ? candidateRecipes
    : routeInspection
      ? meta.recipes.filter(
          (r) =>
            r.input_kinds.includes(routeInspection.input_kind) &&
            r.architectures.includes(routeInspection.architecture),
        )
      : meta.recipes;

  const effectiveName = config.target !== 'auto' ? config.target : plan?.backend ?? null;
  const effectiveRecipe = meta.recipes.find((r) => r.recipe === effectiveName) ?? null;
  // NVFP4/GGUF typed paths are auto-wired into these sources by App; editable
  // fields for them would only invite drift.
  const ternaryInput = inspection?.quantization.ternary_pq2 === true;
  const managedSources = needsBase ? ['quantized', 'gguf', ...(ternaryInput ? ['ternary'] : [])] : [];
  const requiredSources = (effectiveRecipe?.extra_sources ?? []).filter(
    (n) => !managedSources.includes(n),
  );
  const architecture = routeInspection?.architecture ?? '';
  const accelerationSources = architecture.includes('Qwen') ? ['mtp', 'dflash2'] : architecture.includes('Gemma4') ? ['mtp'] : [];
  const allSourceNames = Array.from(new Set([...candidates.flatMap((r) => r.extra_sources), ...accelerationSources]));
  const optionalSourceNames = allSourceNames.filter(
    (n) => !requiredSources.includes(n) && !managedSources.includes(n) && n !== 'ternary',
  );
  const hasTernaryCandidate = candidates.some((r) => r.extra_sources.includes('ternary'));
  const showTernaryField = hasTernaryCandidate && !requiredSources.includes('ternary') && !ternaryInput;

  const onlyBridge = candidates.length > 0 && candidates.every(isBridge);
  const nvfp4Route = inspection?.input_kind === 'nvfp4' || Boolean(config.sources.quantized);
  const serving = plan?.serving ?? null;
  const ceiling = serving?.context_ceiling ?? null;

  const jailbreak: JailbreakMeta =
    meta.choices.jailbreak.find((j) => j.name === config.jailbreak) ?? meta.choices.jailbreak[0];

  const setSource = (name: string, value: string) =>
    onChange({ sources: { ...config.sources, [name]: value } });
  const setParam = (key: string, value: unknown) =>
    onChange({ jailbreak_params: { ...config.jailbreak_params, [key]: value } });
  const setRuntime = (key: string, value: unknown) =>
    onChange({ runtime: { ...config.runtime, [key]: value } });
  const setBenchmark = (key: string, value: unknown) =>
    onChange({ benchmark: { ...config.benchmark, [key]: value } });

  // Benchmark controls fall back to the backend defaults until the user edits
  // them; an untouched control stays absent from the config so the backend keeps
  // its own (geometry-aware) default rather than receiving a pinned value.
  const bd = meta.defaults.benchmark;
  const benchReps = config.benchmark.repetitions ?? bd.repetitions ?? 5;
  const benchKv = config.benchmark.kv_dtype ?? 'auto';
  const benchNPrompt = config.benchmark.n_prompt ?? bd.n_prompt ?? [512, 2048];
  const benchNGen = config.benchmark.n_gen ?? bd.n_gen ?? [128];
  const benchPplCtx = config.benchmark.perplexity_context ?? bd.perplexity_context ?? 4096;
  const benchPplStride = config.benchmark.perplexity_stride ?? bd.perplexity_stride ?? 512;
  const listText = (v: unknown, fallback: number[]) => (Array.isArray(v) ? v : fallback).join(', ');
  const parseList = (s: string) =>
    s.split(',').map((x) => parseInt(x.trim(), 10)).filter((n) => Number.isFinite(n) && n > 0);

  const recipeOption = (r: Recipe) => ({ value: r.recipe, label: r.label, recipe: r });
  const targetGroups = [
    {
      label: '推荐默认',
      options: [{ value: 'auto', label: '自动推荐 · 优先降低权重显存' }],
    },
    {
      label: '原生分组量化（BF16/FP16 → Q4/Q5/Q6/Q8 · 27B/35B 几何）',
      options: candidates
        .filter(
          (r) =>
            r.input_kinds.includes('safetensors') &&
            r.engine !== 'gptqmodel' &&
            !r.input_kinds.includes('nvfp4') &&
            !r.extra_sources.includes('gguf') &&
            !r.extra_sources.includes('ternary') &&
            !isBridge(r),
        )
        .map(recipeOption),
    },
    {
      label: 'AWQ / GPTQ（HF 原生格式与独立引擎）',
      options: candidates.filter(r => r.engine === 'gptqmodel').map(recipeOption),
    },
    {
      label: '通用 GGUF Q8_0 桥接（任意几何 · 小模型如 0.8B）',
      options: candidates.filter(isBridge).map(recipeOption),
    },
    {
      label: 'Bonsai2 三值导入（2 bit）',
      options: candidates.filter((r) => r.extra_sources.includes('ternary')).map(recipeOption),
    },
    {
      label: 'NVFP4 原生导入（4.25 bpw · 含 FP8）',
      options: candidates.filter((r) => r.input_kinds.includes('nvfp4')).map(recipeOption),
    },
    {
      label: 'GGUF 块导入（位宽由文件决定）',
      options: candidates.filter((r) => r.extra_sources.includes('gguf')).map(recipeOption),
    },
  ].filter((g) => g.options.length > 0);

  const renderParam = (p: ParamSchema) => {
    const required = paramRequired(p, meta.env);
    const value = config.jailbreak_params[p.key];
    let control;
    if (p.type === 'int' || p.type === 'float') {
      control = (
        <InputNumber
          style={{ width: '100%' }}
          value={value ?? null}
          placeholder={p.placeholder}
          step={p.type === 'float' ? 0.1 : 1}
          onChange={(v) => setParam(p.key, v)}
        />
      );
    } else if (p.type === 'path') {
      control = <PathInput label={p.label} kind="file" value={value ?? ''} placeholder={p.placeholder} onChange={(v) => setParam(p.key, v)} />;
    } else if (p.type === 'string_list') {
      control = (
        <Input.TextArea
          autoSize={{ minRows: 2, maxRows: 5 }}
          value={Array.isArray(value) ? value.join('\n') : value ?? ''}
          placeholder={p.placeholder || '每行一条'}
          onChange={(e) =>
            setParam(
              p.key,
              e.target.value.split('\n').map((s) => s.trim()).filter(Boolean),
            )
          }
        />
      );
    } else {
      control = (
        <Input
          value={value ?? ''}
          placeholder={p.placeholder}
          allowClear
          onChange={(e) => setParam(p.key, e.target.value)}
        />
      );
    }
    return (
      <Field key={p.key} label={p.label} required={required} tooltip={p.help}>
        {control}
      </Field>
    );
  };

  const basicParams = jailbreak.params.filter((p) => !p.advanced);
  const advancedParams = jailbreak.params.filter((p) => p.advanced);

  const sourceField = (name: string, required: boolean) => (
    <Field
      key={name}
      label={`来源 · ${name}`}
      required={required}
      tooltip={
        required
          ? `当前目标 ${effectiveRecipe?.recipe ?? ''} 需要 --source ${name}=<路径>；配方不会合成这些权重`
          : `可选附加来源 ${name}；提供后可解锁相应目标`
      }
    >
      <PathInput label={`来源 ${name}`} kind={['quantized', 'dflash2'].includes(name) || (name === 'mtp' && !architecture.includes('Gemma4')) ? 'directory' : 'file'}
        value={config.sources[name] ?? ''}
        placeholder={`D:/path/to/${name}`}
        onChange={(value) => setSource(name, value)}
      />
    </Field>
  );

  return (
    <Form layout="vertical" disabled={disabled} requiredMark={false}>
      <div className="step" hidden={stage !== 0}>
        <StepHeader index="" title="输入与输出" />
        <Field
          label="输入模型"
          required
          tooltip="粘贴 Hugging Face 模型目录、NVFP4 检查点目录或 .gguf 文件路径；自动识别架构与精度，原始模型不会被修改"
        >
          <PathInput label="输入模型" kind="model"
            value={inputPath}
            placeholder="D:/models/Qwen3.8-27B 或 D:/models/xxx-Q4_K_M.gguf"
            onChange={onInputPath}
          />
        </Field>
        {needsBase && (
          <>
            <Notice
              tone="info"
              title={
                ternaryInput ? 'PQ2_0 三值权重：自动选择 Bonsai 2 导入' : inspection?.input_kind === 'nvfp4'
                  ? 'NVFP4 检查点：原样导入，位宽不可更改'
                  : 'GGUF：位宽由文件决定，原样导入'
              }
            >
              {ternaryInput ? '已按张量存储格式识别三值 GGUF。请补充匹配的原始 HF 模型目录；基座中的 MTP 如存在会自动带入。当前仅支持已核对的 27B 几何。' : inspection?.input_kind === 'nvfp4'
                ? 'NVFP4 路线把已发布的 4.25 bpw 权重直接转成 NInfer 格式（含 FP8 张量）；要换其他位宽或越狱，请使用原始 BF16/FP16 检查点。'
                : '此高级传统路线需要原始 HF 配置与分词器。仅有完整 Qwen3.5-0.8B Q8_0 GGUF 时，可使用引导模式独立导入，自动生成资源；独立配方会解码后重编码 Q8_0。'}
            </Notice>
            <Field
              label="原始 HF 目录"
              required
              tooltip={
                inspection?.input_kind === 'nvfp4'
                  ? '转换器 --model 需要原始 BF16 目录：提供 BF16 词嵌入、config 与 tokenizer'
                  : '转换器 --model 需要原始 HF 目录的 config.json 与 tokenizer 文件'
              }
            >
              <PathInput label="原始 HF 目录"
                value={baseDir}
                placeholder="D:/models/Qwen3.8-27B（原始 HF 目录）"
                onChange={onBaseDir}
              />
            </Field>
          </>
        )}
        <Field
          label="输出目录"
          required
          tooltip="构建产物写入此目录；必须为空且不同于原始模型位置。已按输入自动生成，可直接修改"
        >
          <PathInput label="输出目录"
            value={config.output_dir}
            placeholder="D:/models/Qwen3.8-27B-ninfer"
            onChange={(value) => {
              onOutputTouched();
              onChange({ output_dir: value });
            }}
          />
        </Field>
      </div>

      <div className="step" hidden={stage !== 1}>
        <StepHeader index="" title="量化目标" note="改变产物" />
        <Field
          label="量化目标"
          required
          tooltip="智能默认：提供已发布三值 GGUF 时优先 Bonsai2（2 bit），否则选已验证的最低 bpw 目标；显式选择会精确使用该配方。仅显示与当前几何兼容的配方"
          extra={
            routeInspection
              ? `已识别 ${routeInspection.architecture}（${routeInspection.input_kind}），仅显示兼容配方`
              : '填写输入路径后自动筛选兼容配方'
          }
        >
          <Select
            showSearch
            value={config.target}
            optionFilterProp="label"
            options={targetGroups}
            onChange={(v) => onChange({ target: v })}
            optionRender={(opt: any) =>
              opt.data.recipe ? <RecipeOption recipe={opt.data.recipe} /> : <span>{opt.data.label}</span>
            }
          />
        </Field>

        {onlyBridge && (
          <Notice tone="info" title="此几何只支持通用 GGUF Q8_0 桥接">
            当前小模型可转换为 8 bit。此引擎尚未支持它的低位宽内核，不能选择 2/4 bit。已有 MTP 权重会一并导入。
          </Notice>
        )}

        {requiredSources.map((name) => sourceField(name, true))}
        {showTernaryField && (
          <Field
            label="已发布三值 GGUF"
            tooltip="可选：提供已发布的 PQ2_0 三值 GGUF（--source ternary）后，智能默认会优先选择 Bonsai2（2 bit）。TurboForge 不会自行训练三值权重"
          >
            <PathInput label="已发布三值 GGUF" kind="file"
              value={config.sources.ternary ?? ''}
              placeholder="D:/models/xxx-PQ2_0.gguf（可选）"
              onChange={(value) => setSource('ternary', value)}
            />
          </Field>
        )}
        {optionalSourceNames.length > 0 && (
          <Collapse
            ghost
            size="small"
            style={{ marginBottom: 12 }}
            items={[
              {
                key: 'src',
                label: <span className="muted">其他可选来源（{optionalSourceNames.length}）</span>,
                children: optionalSourceNames.map((name) => sourceField(name, false)),
              },
            ]}
          />
        )}

        <Field
          label="转换设备"
          required
          tooltip="分组量化在哪执行。CUDA 最快（需 CUDA 版 torch）；CPU 合法但本机无实测计时，预计远慢于 CUDA。直接导入路线（NVFP4/GGUF/三值）不依赖 GPU 量化"
        >
          <Select
            value={config.device}
            options={meta.choices.device.map((d) => ({ value: d.value, label: d.label, d }))}
            onChange={(v) => onChange({ device: v })}
            optionRender={(opt: any) => (
              <div className="opt">
                <span className="opt-name">{opt.data.d.label}</span>
                <span className="opt-meta">{opt.data.d.description}</span>
              </div>
            )}
          />
        </Field>
        {!meta.env.torch_cuda && <Notice tone="info" title="已推荐 CPU 转换">当前 torch 不支持 CUDA。CPU 只负责转换，转换后的模型仍在 NVIDIA 显卡上推理。</Notice>}

        {nvfp4Route && config.jailbreak !== 'none' && (
          <Field
            label="允许近似再量化"
            tooltip="仅当对已量化（NVFP4）权重做修改后再量化时相关：两次独立损失会叠加。开启表示明确接受叠加损失；导入路线 + 越狱仍会被拒绝（编辑无法到达导入张量）"
          >
            <Switch
              checked={config.allow_approximate_requantization}
              onChange={(v) => onChange({ allow_approximate_requantization: v })}
            />
          </Field>
        )}
      </div>

      <div className="step" hidden={stage !== 1}>
        <StepHeader index="" title="越狱" note="可选 · 改变产物" />
        <Field
          label="越狱方式"
          tooltip="不越狱则保持原始权重；只保留主流的 Heretic 与 Abliteration 两种。不支持的组合会被明确拒绝，绝不静默回退"
        >
          <Select
            value={config.jailbreak}
            options={meta.choices.jailbreak.filter(j => !needsBase || j.name === 'none').map((j) => ({
              value: j.name,
              label: `${j.label}（${j.name}）`,
              jb: j,
              disabled: (j.name !== 'none' && (!j.capability.available || Boolean(effectiveRecipe?.extra_sources.some((s) => ['ternary', 'gguf', 'quantized'].includes(s))))),
            }))}
            onChange={(v) => onChange({ jailbreak: v, jailbreak_params: {} })}
            optionRender={(opt: any) => {
              const j: JailbreakMeta = opt.data.jb;
              return (
                <div className="opt">
                  <span className="opt-name">
                    {j.label}
                    {!j.capability.available && <Chip tone="bad">不可用</Chip>}
                  </span>
                  <span className="opt-meta">{j.description}</span>
                </div>
              );
            }}
          />
        </Field>
        {jailbreak.name !== 'none' && !jailbreak.capability.available && (
          <Notice tone="warn" title="该处理器当前不可用">
            {jailbreak.capability.unavailable_reason ?? '依赖缺失'}
          </Notice>
        )}
        {jailbreak.name === 'heretic' && jailbreak.capability.available && (
          <Notice tone="info" title="Heretic 一键越狱：独立窗口运行">
            {`开始后 Heretic 会在独立控制台窗口运行；结束时选择「Save the model to a local folder」，把权重保存到 ${config.output_dir}/intermediate/heretic，TurboForge 检测到后自动继续量化。`}
          </Notice>
        )}
        {jailbreak.capability.requires_high_precision && (
          <Notice tone="info" title="仅支持高精度权重（BF16/FP16/FP32）">
            NVFP4 / GGUF 输入会被拒绝；请提供原始高精度检查点。
          </Notice>
        )}
        {basicParams.map(renderParam)}
        {advancedParams.length > 0 && (
          <Collapse
            ghost
            size="small"
            style={{ marginBottom: 12 }}
            items={[
              {
                key: 'adv',
                label: <span className="muted">高级参数（可选）</span>,
                children: advancedParams.map(renderParam),
              },
            ]}
          />
        )}
      </div>

      <div className="step" hidden={stage !== 2}>
        <StepHeader index="" title="服务参数" note="不改变产物" />
        <Notice tone="info" title="保留默认即可：单用户、自动加速">
          系统按模型选择缓存精度，有可用 MTP / DFlash2 时自动启用。较长上下文与更高并发会占用更多显存；实际速度需转换后测试。
        </Notice>
        <Field label="推测解码" tooltip="自动使用已导入的 MTP / DFlash2。显存紧张或实际速度变慢时，可关闭；不删除产物中的组件。">
          <Select value={config.runtime.speculative ?? 'auto'} options={[{value:'auto', label:'自动加速（推荐）'}, {value:'none', label:'关闭 · 减少推测解码开销'}]} onChange={(value) => setRuntime('speculative', value)} />
        </Field>
        <Field
          label="服务上下文"
          tooltip="ninfer-serve 的 --max-context / --kv-capacity。超出几何上限会被自动截断；计划与启动器始终使用截断后的值"
          extra={
            ceiling
              ? `此几何服务上限 ${ceiling}；请求超出会被自动截断到上限`
              : serving
                ? '最大可用长度仍受模型配置与空闲显存限制'
                : '选定目标后显示几何上限'
          }
        >
          <InputNumber
            style={{ width: '100%' }}
            min={1}
            step={256}
            max={ceiling ?? undefined}
            value={ceiling ? Math.min(config.runtime.max_context ?? 32768, ceiling) : config.runtime.max_context ?? 32768}
            onChange={(v) => setRuntime('max_context', v)}
          />
        </Field>
        <Collapse ghost size="small" items={[{key:"service-advanced", label:"高级服务设置（并发、端口）", children:<>
        <Field
          label="最大并发"
          tooltip="同时服务的请求数（--max-concurrency）。单卡消费级建议 1；提高会增加显存占用"
        >
          <InputNumber
            style={{ width: '100%' }}
            min={1}
            value={config.runtime.max_concurrency ?? meta.defaults.runtime.max_concurrency ?? 1}
            onChange={(v) => setRuntime('max_concurrency', v)}
          />
        </Field>
        <Field
          label="服务端口"
          tooltip="run.bat 中 ninfer-serve 的监听端口（仅绑定 127.0.0.1）"
        >
          <InputNumber
            style={{ width: '100%' }}
            min={1}
            max={65535}
            value={config.runtime.port ?? meta.defaults.runtime.port}
            onChange={(v) => setRuntime('port', v)}
          />
        </Field>
        </>}]} />
      </div>

      <div className="step" hidden={stage !== 2}>
        <StepHeader index="" title="验证与实测" note="可选" />
        <Field
          label="转换后冒烟验证"
          tooltip="用 ninfer.exe 实际加载产物做一次短生成，确认可读可跑；关闭则跳过验证（产物仍写入）。绝不把未运行写成已通过"
        >
          <Switch
            checked={config.run_validation}
            onChange={(v) => onChange({ run_validation: v })}
          />
        </Field>
        <Field
          label="转换后测试速度（质量评测可选）"
          extra="不填写下方测试文本时，不运行质量评测。速度来自微基准，实际任务表现可能不同。"
          tooltip="开启后用 ninfer_bench / ninfer-perplexity 实际运行并填入真实数字；关闭时所有指标保持「未实测」，绝不估算"
        >
          <Switch checked={config.run_benchmark} onChange={(v) => onChange({ run_benchmark: v })} />
        </Field>
        {config.run_benchmark && (
          <>
            <Field label="基准重复次数" tooltip="ninfer_bench 重复次数（-r）；越多越稳但越慢">
              <InputNumber
                style={{ width: '100%' }}
                min={1}
                value={benchReps}
                onChange={(v) => setBenchmark('repetitions', v)}
              />
            </Field>
            <Field
              label="基准 KV 精度"
              tooltip="仅影响实测，不改变产物。自动＝跟随几何（小 GDN 桥用 bf16 提速 MTP，原生 27B/35B 用 rk4v4 省显存）；显式选择会覆盖自动值"
            >
              <Select
                value={benchKv}
                options={[
                  { value: 'auto', label: '自动（跟随几何）' },
                  { value: 'bf16', label: 'bf16 · MTP 更快' },
                  { value: 'rk4v4', label: 'rk4v4 · 更省显存' },
                ]}
                onChange={(v) => setBenchmark('kv_dtype', v === 'auto' ? null : v)}
              />
            </Field>
            <Field
              label="困惑度文本"
              tooltip="用于 ninfer-perplexity 的文本文件；留空则不测困惑度。量化损失需要同时提供基线产物"
            >
              <Input
                data-ai-ignore value={config.benchmark.perplexity_text ?? ''}
                placeholder="D:/corpus/sample.txt"
                allowClear
                onChange={(e) => setBenchmark('perplexity_text', e.target.value)}
              />
            </Field>
            <p className="muted" style={{ marginTop: 0, marginBottom: 12, fontSize: 12, lineHeight: 1.6 }}>
              投机解码自动跟随导入的组件：
              {serving?.spec
                ? ` ${serving.spec}（draft ${serving.draft_tokens}）`
                : ' 无（未检测到 MTP/DFlash2）'}
              。实测命令与拒绝原因都会写入报告，可复现。
            </p>
            <Collapse
              ghost
              size="small"
              style={{ marginBottom: 12 }}
              items={[
                {
                  key: 'bench-adv',
                  label: <span className="muted">高级实测参数（可选）</span>,
                  children: (
                    <>
                      <Field
                        label="预填充长度列表"
                        tooltip="ninfer_bench -p，逗号分隔（如 512,2048）；超出几何上限会自动截断"
                      >
                        <Input
                          value={listText(config.benchmark.n_prompt, benchNPrompt)}
                          placeholder="512, 2048"
                          allowClear
                          onChange={(e) => setBenchmark('n_prompt', parseList(e.target.value))}
                        />
                      </Field>
                      <Field label="生成长度列表" tooltip="ninfer_bench -n，逗号分隔（如 128）">
                        <Input
                          value={listText(config.benchmark.n_gen, benchNGen)}
                          placeholder="128"
                          allowClear
                          onChange={(e) => setBenchmark('n_gen', parseList(e.target.value))}
                        />
                      </Field>
                      <Field label="困惑度上下文" tooltip="ninfer-perplexity --context">
                        <InputNumber
                          style={{ width: '100%' }}
                          min={1}
                          value={benchPplCtx}
                          onChange={(v) => setBenchmark('perplexity_context', v)}
                        />
                      </Field>
                      <Field label="困惑度步长" tooltip="ninfer-perplexity --stride（语料模式）">
                        <InputNumber
                          style={{ width: '100%' }}
                          min={1}
                          value={benchPplStride}
                          onChange={(v) => setBenchmark('perplexity_stride', v)}
                        />
                      </Field>
                      <Field
                        label="困惑度语料 manifest"
                        tooltip="与「困惑度文本」二选一；提供后按语料滑窗测困惑度"
                      >
                        <Input
                          data-ai-ignore value={config.benchmark.perplexity_corpus ?? ''}
                          placeholder="D:/corpus/manifest.json"
                          allowClear
                          onChange={(e) => setBenchmark('perplexity_corpus', e.target.value)}
                        />
                      </Field>
                      <Field
                        label="基线产物"
                        tooltip="另一个 .ninfer（如原始精度产物）；与产物在同一语料/上下文测困惑度，差值即量化损失"
                      >
                        <Input
                          data-ai-ignore value={config.benchmark.baseline_artifact ?? ''}
                          placeholder="D:/models/baseline.ninfer"
                          allowClear
                          onChange={(e) => setBenchmark('baseline_artifact', e.target.value)}
                        />
                      </Field>
                      <Field
                        label="基准语料 ids"
                        tooltip="ninfer_bench --corpus 的 ids 文件；留空用内置默认。注意：内置 tiled 语料上的投机接受率无意义，解码 tok/s 仍有效"
                      >
                        <Input
                          data-ai-ignore value={config.benchmark.corpus ?? ''}
                          placeholder="D:/llm/ninfer-upstream/bench/fixtures/bench_corpus.ids"
                          allowClear
                          onChange={(e) => setBenchmark('corpus', e.target.value)}
                        />
                      </Field>
                    </>
                  ),
                },
              ]}
            />
          </>
        )}
        <p className="muted" style={{ marginTop: 4, marginBottom: 0, fontSize: 12, lineHeight: 1.6 }}>
          转换完成后会自动用 ninfer 实际加载产物做冒烟验证；MTP / DFlash2 组件如已检测到会自动随附，无需配置。
        </p>
      </div>
    </Form>
  );
}
