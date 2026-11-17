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
          <Chip tone="warn">Needs {recipe.extra_sources.join('/')}</Chip>
        )}
        {recipe.needs_gpu && <Chip tone="info">GPU</Chip>}
        {isBridge(recipe) && <Chip tone="info">Generic bridge</Chip>}
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
      label: 'Recommended default',
      options: [{ value: 'auto', label: 'Auto recommendation · prefers lower weight VRAM' }],
    },
    {
      label: 'Native groupwise quantization (BF16/FP16 → Q4/Q5/Q6/Q8 · 27B/35B geometry)',
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
      label: 'AWQ / GPTQ (HF native format and separate engine)',
      options: candidates.filter(r => r.engine === 'gptqmodel').map(recipeOption),
    },
    {
      label: 'Generic GGUF Q8_0 bridge (any geometry · small models such as 0.8B)',
      options: candidates.filter(isBridge).map(recipeOption),
    },
    {
      label: 'Bonsai2 ternary import (2 bit)',
      options: candidates.filter((r) => r.extra_sources.includes('ternary')).map(recipeOption),
    },
    {
      label: 'NVFP4 native import (4.25 bpw · includes FP8)',
      options: candidates.filter((r) => r.input_kinds.includes('nvfp4')).map(recipeOption),
    },
    {
      label: 'GGUF block import (bit width decided by the file)',
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
          placeholder={p.placeholder || 'One entry per line'}
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
      label={`Source · ${name}`}
      required={required}
      tooltip={
        required
          ? `The current target ${effectiveRecipe?.recipe ?? ''} needs --source ${name}=<path>; the recipe does not synthesize these weights`
          : `Optional extra source ${name}; supplying it unlocks the corresponding targets`
      }
    >
      <PathInput label={`Source ${name}`} kind={['quantized', 'dflash2'].includes(name) || (name === 'mtp' && !architecture.includes('Gemma4')) ? 'directory' : 'file'}
        value={config.sources[name] ?? ''}
        placeholder={`D:/path/to/${name}`}
        onChange={(value) => setSource(name, value)}
      />
    </Field>
  );

  return (
    <Form layout="vertical" disabled={disabled} requiredMark={false}>
      <div className="step" hidden={stage !== 0}>
        <StepHeader index="" title="Input and output" />
        <Field
          label="Input model"
          required
          tooltip="Paste a Hugging Face model directory, an NVFP4 checkpoint directory or a .gguf file path; architecture and precision are identified automatically and the original model is never modified"
        >
          <PathInput label="Input model" kind="model"
            value={inputPath}
            placeholder="D:/models/Qwen3.8-27B or D:/models/xxx-Q4_K_M.gguf"
            onChange={onInputPath}
          />
        </Field>
        {needsBase && (
          <>
            <Notice
              tone="info"
              title={
                ternaryInput ? 'PQ2_0 ternary weights: Bonsai 2 import selected automatically' : inspection?.input_kind === 'nvfp4'
                  ? 'NVFP4 checkpoint: imported verbatim; bit width cannot change'
                  : 'GGUF: bit width decided by the file, imported verbatim'
              }
            >
              {ternaryInput ? 'Ternary GGUF recognized by its tensor storage format. Supply the matching original HF model directory; MTP in the base is carried over automatically when present. Only the verified 27B geometry is currently supported.' : inspection?.input_kind === 'nvfp4'
                ? 'The NVFP4 route converts published 4.25 bpw weights directly into the NInfer format (including FP8 tensors); for another bit width or abliteration, use the original BF16/FP16 checkpoint.'
                : 'This advanced legacy route needs the original HF config and tokenizer. With only a complete Qwen3.5-0.8B Q8_0 GGUF, use guided mode for standalone import with automatically generated resources; the standalone recipe decodes and re-encodes as Q8_0.'}
            </Notice>
            <Field
              label="Original HF directory"
              required
              tooltip={
                inspection?.input_kind === 'nvfp4'
                  ? 'The converter --model needs the original BF16 directory: it supplies the BF16 word embedding, config and tokenizer'
                  : 'The converter --model needs config.json and tokenizer files from the original HF directory'
              }
            >
              <PathInput label="Original HF directory"
                value={baseDir}
                placeholder="D:/models/Qwen3.8-27B (original HF directory)"
                onChange={onBaseDir}
              />
            </Field>
          </>
        )}
        <Field
          label="Output directory"
          required
          tooltip="Build artifacts are written to this directory; it must be empty and different from the original model location. Generated automatically from the input; you can edit it directly"
        >
          <PathInput label="Output directory"
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
        <StepHeader index="" title="Quantization target" note="Changes the artifact" />
        <Field
          label="Quantization target"
          required
          tooltip="Smart default: with a published ternary GGUF supplied, Bonsai2 (2 bit) is preferred; otherwise the lowest-bpw verified target is chosen. An explicit selection uses exactly that recipe. Only recipes compatible with the current geometry are shown"
          extra={
            routeInspection
              ? `Identified ${routeInspection.architecture} (${routeInspection.input_kind}); only compatible recipes are shown`
              : 'Compatible recipes are filtered automatically after entering the input path'
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
          <Notice tone="info" title="This geometry only supports the generic GGUF Q8_0 bridge">
            This small model can be converted to 8 bit. The engine does not yet support lower-bit kernels for it, so 2/4 bit cannot be selected. Existing MTP weights are imported along with it.
          </Notice>
        )}

        {requiredSources.map((name) => sourceField(name, true))}
        {showTernaryField && (
          <Field
            label="Published ternary GGUF"
            tooltip="Optional: after supplying a published PQ2_0 ternary GGUF (--source ternary), the smart default prefers Bonsai2 (2 bit). TurboForge never trains ternary weights itself"
          >
            <PathInput label="Published ternary GGUF" kind="file"
              value={config.sources.ternary ?? ''}
              placeholder="D:/models/xxx-PQ2_0.gguf (optional)"
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
                label: <span className="muted">Other optional sources ({optionalSourceNames.length})</span>,
                children: optionalSourceNames.map((name) => sourceField(name, false)),
              },
            ]}
          />
        )}

        <Field
          label="Conversion device"
          required
          tooltip="Where groupwise quantization runs. CUDA is fastest (needs a CUDA build of torch); CPU is legal but has no measured timing on this machine and is expected to be far slower than CUDA. Direct import routes (NVFP4/GGUF/ternary) do not depend on GPU quantization"
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
        {!meta.env.torch_cuda && <Notice tone="info" title="CPU conversion recommended">The current torch does not support CUDA. The CPU only handles conversion; the converted model still runs inference on the NVIDIA GPU.</Notice>}

        {nvfp4Route && config.jailbreak !== 'none' && (
          <Field
            label="Allow approximate requantization"
            tooltip="Only relevant when modifying already-quantized (NVFP4) weights and quantizing again: two independent losses stack. Enabling it explicitly accepts the stacked loss; import routes + abliteration are still refused (edits cannot reach imported tensors)"
          >
            <Switch
              checked={config.allow_approximate_requantization}
              onChange={(v) => onChange({ allow_approximate_requantization: v })}
            />
          </Field>
        )}
      </div>

      <div className="step" hidden={stage !== 1}>
        <StepHeader index="" title="Abliteration" note="Optional · changes the artifact" />
        <Field
          label="Abliteration method"
          tooltip="No abliteration keeps the original weights; only the mainstream Heretic and Abliteration methods are kept. Unsupported combinations are explicitly refused, never silently downgraded"
        >
          <Select
            value={config.jailbreak}
            options={meta.choices.jailbreak.filter(j => !needsBase || j.name === 'none').map((j) => ({
              value: j.name,
              label: `${j.label} (${j.name})`,
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
                    {!j.capability.available && <Chip tone="bad">Unavailable</Chip>}
                  </span>
                  <span className="opt-meta">{j.description}</span>
                </div>
              );
            }}
          />
        </Field>
        {jailbreak.name !== 'none' && !jailbreak.capability.available && (
          <Notice tone="warn" title="This processor is currently unavailable">
            {jailbreak.capability.unavailable_reason ?? 'Missing dependencies'}
          </Notice>
        )}
        {jailbreak.name === 'heretic' && jailbreak.capability.available && (
          <Notice tone="info" title="Heretic one-click abliteration: runs in a separate window">
            {`After starting, Heretic runs in a separate console window; at the end choose 'Save the model to a local folder' and save the weights to ${config.output_dir}/intermediate/heretic. TurboForge detects them and continues quantization automatically.`}
          </Notice>
        )}
        {jailbreak.capability.requires_high_precision && (
          <Notice tone="info" title="Only high-precision weights supported (BF16/FP16/FP32)">
            NVFP4 / GGUF inputs are refused; provide the original high-precision checkpoint.
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
                label: <span className="muted">Advanced parameters (optional)</span>,
                children: advancedParams.map(renderParam),
              },
            ]}
          />
        )}
      </div>

      <div className="step" hidden={stage !== 2}>
        <StepHeader index="" title="Serving parameters" note="Does not change the artifact" />
        <Notice tone="info" title="Keeping the defaults is fine: single user, automatic acceleration">
          The system picks cache precision per model and enables MTP / DFlash2 automatically when available. Longer context and higher concurrency use more VRAM; actual speed must be tested after conversion.
        </Notice>
        <Field label="Speculative decoding" tooltip="Uses imported MTP / DFlash2 automatically. Turn it off when VRAM is tight or actual speed drops; components in the artifact are not deleted.">
          <Select value={config.runtime.speculative ?? 'auto'} options={[{value:'auto', label:'Automatic acceleration (recommended)'}, {value:'none', label:'Off · reduces speculative-decoding overhead'}]} onChange={(value) => setRuntime('speculative', value)} />
        </Field>
        <Field
          label="Serving context"
          tooltip="ninfer-serve --max-context / --kv-capacity. Values above the geometry ceiling are capped automatically; the plan and launcher always use the capped value"
          extra={
            ceiling
              ? `Serving ceiling for this geometry: ${ceiling}; larger requests are capped automatically`
              : serving
                ? 'The maximum usable length is still limited by the model config and free VRAM'
                : 'The geometry ceiling appears after selecting a target'
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
        <Collapse ghost size="small" items={[{key:"service-advanced", label:"Advanced serving settings (concurrency, port)", children:<>
        <Field
          label="Max concurrency"
          tooltip="Number of requests served simultaneously (--max-concurrency). 1 is recommended for a single consumer GPU; higher values increase VRAM usage"
        >
          <InputNumber
            style={{ width: '100%' }}
            min={1}
            value={config.runtime.max_concurrency ?? meta.defaults.runtime.max_concurrency ?? 1}
            onChange={(v) => setRuntime('max_concurrency', v)}
          />
        </Field>
        <Field
          label="Serving port"
          tooltip="The ninfer-serve listen port in run.bat (binds 127.0.0.1 only)"
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
        <StepHeader index="" title="Validation and measurement" note="Optional" />
        <Field
          label="Post-conversion smoke test"
          tooltip="Loads the artifact with ninfer.exe for one short real generation to confirm it is readable and runnable; turning it off skips validation (the artifact is still written). An unrun check is never reported as passed"
        >
          <Switch
            checked={config.run_validation}
            onChange={(v) => onChange({ run_validation: v })}
          />
        </Field>
        <Field
          label="Post-conversion speed benchmark (quality evaluation optional)"
          extra="Without the test text below, quality evaluation does not run. Speed comes from a microbenchmark; real-task performance may differ."
          tooltip="When enabled, ninfer_bench / ninfer-perplexity actually run and fill in real numbers; when off, every metric stays 'Not measured' and is never estimated"
        >
          <Switch checked={config.run_benchmark} onChange={(v) => onChange({ run_benchmark: v })} />
        </Field>
        {config.run_benchmark && (
          <>
            <Field label="Benchmark repetitions" tooltip="ninfer_bench repetition count (-r); more is steadier but slower">
              <InputNumber
                style={{ width: '100%' }}
                min={1}
                value={benchReps}
                onChange={(v) => setBenchmark('repetitions', v)}
              />
            </Field>
            <Field
              label="Benchmark KV precision"
              tooltip="Affects measurement only, not the artifact. Auto = follows the geometry (small GDN bridges use bf16 to speed up MTP; native 27B/35B uses rk4v4 to save VRAM); an explicit choice overrides the automatic value"
            >
              <Select
                value={benchKv}
                options={[
                  { value: 'auto', label: 'Auto (follows geometry)' },
                  { value: 'bf16', label: 'bf16 · faster MTP' },
                  { value: 'rk4v4', label: 'rk4v4 · saves VRAM' },
                ]}
                onChange={(v) => setBenchmark('kv_dtype', v === 'auto' ? null : v)}
              />
            </Field>
            <Field
              label="Perplexity text"
              tooltip="Text file for ninfer-perplexity; leave empty to skip perplexity. Quantization loss also needs a baseline artifact"
            >
              <Input
                data-ai-ignore value={config.benchmark.perplexity_text ?? ''}
                placeholder="D:/corpus/sample.txt"
                allowClear
                onChange={(e) => setBenchmark('perplexity_text', e.target.value)}
              />
            </Field>
            <p className="muted" style={{ marginTop: 0, marginBottom: 12, fontSize: 12, lineHeight: 1.6 }}>
              Speculative decoding follows the imported components automatically: 
              {serving?.spec
                ? ` ${serving.spec} (draft ${serving.draft_tokens})`
                : ' none (no MTP/DFlash2 detected)'}
              . Measured commands and refusal reasons are written into the report and are reproducible.
            </p>
            <Collapse
              ghost
              size="small"
              style={{ marginBottom: 12 }}
              items={[
                {
                  key: 'bench-adv',
                  label: <span className="muted">Advanced measurement parameters (optional)</span>,
                  children: (
                    <>
                      <Field
                        label="Prefill length list"
                        tooltip="ninfer_bench -p, comma-separated (e.g. 512,2048); values above the geometry ceiling are capped automatically"
                      >
                        <Input
                          value={listText(config.benchmark.n_prompt, benchNPrompt)}
                          placeholder="512, 2048"
                          allowClear
                          onChange={(e) => setBenchmark('n_prompt', parseList(e.target.value))}
                        />
                      </Field>
                      <Field label="Generation length list" tooltip="ninfer_bench -n, comma-separated (e.g. 128)">
                        <Input
                          value={listText(config.benchmark.n_gen, benchNGen)}
                          placeholder="128"
                          allowClear
                          onChange={(e) => setBenchmark('n_gen', parseList(e.target.value))}
                        />
                      </Field>
                      <Field label="Perplexity context" tooltip="ninfer-perplexity --context">
                        <InputNumber
                          style={{ width: '100%' }}
                          min={1}
                          value={benchPplCtx}
                          onChange={(v) => setBenchmark('perplexity_context', v)}
                        />
                      </Field>
                      <Field label="Perplexity stride" tooltip="ninfer-perplexity --stride (corpus mode)">
                        <InputNumber
                          style={{ width: '100%' }}
                          min={1}
                          value={benchPplStride}
                          onChange={(v) => setBenchmark('perplexity_stride', v)}
                        />
                      </Field>
                      <Field
                        label="Perplexity corpus manifest"
                        tooltip="Alternative to 'Perplexity text'; when supplied, perplexity is measured with a sliding corpus window"
                      >
                        <Input
                          data-ai-ignore value={config.benchmark.perplexity_corpus ?? ''}
                          placeholder="D:/corpus/manifest.json"
                          allowClear
                          onChange={(e) => setBenchmark('perplexity_corpus', e.target.value)}
                        />
                      </Field>
                      <Field
                        label="Baseline artifact"
                        tooltip="Another .ninfer (e.g. an original-precision artifact); perplexity is measured on the same corpus/context as the artifact, and the difference is the quantization loss"
                      >
                        <Input
                          data-ai-ignore value={config.benchmark.baseline_artifact ?? ''}
                          placeholder="D:/models/baseline.ninfer"
                          allowClear
                          onChange={(e) => setBenchmark('baseline_artifact', e.target.value)}
                        />
                      </Field>
                      <Field
                        label="Benchmark corpus ids"
                        tooltip="The ids file for ninfer_bench --corpus; leave empty for the built-in default. Note: speculative acceptance rates on the built-in tiled corpus are meaningless; decode tok/s remains valid"
                      >
                        <Input
                          data-ai-ignore value={config.benchmark.corpus ?? ''}
                          placeholder="Leave empty to use the built-in default"
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
          After conversion, ninfer automatically loads the artifact for a real smoke test; detected MTP / DFlash2 components are carried automatically, no configuration needed.
        </p>
      </div>
    </Form>
  );
}
