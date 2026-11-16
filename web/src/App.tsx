import { useEffect, useRef, useState } from 'react';
import { App as AntApp, Button, Collapse, Skeleton, Steps, Select, Switch, InputNumber } from 'antd';
import { SettingOutlined } from '@ant-design/icons';
import { Chip, Notice, Panel } from './components/ui';
import { api } from './api/client';
import type {
  BackendCapability,
  BuildStatus,
  ForgeConfig,
  ForgePlan,
  Meta,
  ModelInspection,
  Recipe,
  Report,
} from './api/types';
import AppHeader from './components/AppHeader';
import ConfigPanel from './components/ConfigPanel';
import InspectionCard from './components/InspectionCard';
import PlanPreview from './components/PlanPreview';
import BuildPanel from './components/BuildPanel';
import WorkflowSummary from './components/WorkflowSummary';
import PathInput from './components/PathInput';
import Field from './components/Field';
import AssistantProvider from './components/assistant/AssistantProvider';
import StoragePanel from './components/StoragePanel';

const INITIAL_CONFIG: ForgeConfig = {
  input_path: '',
  output_dir: '',
  target: 'auto',
  jailbreak: 'none',
  jailbreak_params: {},
  runtime: {},
  allow_approximate_requantization: false,
  sources: {},
  device: 'cuda',
  run_validation: true,
  run_benchmark: false,
  benchmark: {},
};

// Drop empty optional values so the backend never sees a blank source/param.
function cleanConfig(c: ForgeConfig): ForgeConfig {
  const keep = (v: unknown) =>
    v !== '' && v !== null && v !== undefined && !(Array.isArray(v) && v.length === 0);
  return {
    ...c,
    sources: Object.fromEntries(Object.entries(c.sources).filter(([, v]) => keep(v))),
    jailbreak_params: Object.fromEntries(Object.entries(c.jailbreak_params).filter(([, v]) => keep(v))),
    benchmark: Object.fromEntries(Object.entries(c.benchmark).filter(([, v]) => keep(v))),
  };
}

function suggestOutput(input: string): string {
  const trimmed = input.trim().replace(/[\\/]+$/, '');
  if (!trimmed) return '';
  if (trimmed.toLowerCase().endsWith('.gguf')) {
    const cut = Math.max(trimmed.lastIndexOf('/'), trimmed.lastIndexOf('\\'));
    const dir = cut >= 0 ? trimmed.slice(0, cut) : '';
    const stem = trimmed.slice(cut + 1, -5);
    return dir ? `${dir}/${stem}-ninfer` : `${stem}-ninfer`;
  }
  return `${trimmed}-ninfer`;
}

interface SavedBuild { id: string; config: ForgeConfig; inputPath: string; baseDir: string; startMs: number }
function readSavedBuild(): SavedBuild | null {
  try {
    const value: SavedBuild | null = JSON.parse(sessionStorage.getItem('turboforge.build') ?? 'null');
    return value && typeof value.id === 'string' && value.config?.input_path ? value : null;
  } catch { return null; }
}

export default function App() {
  const { message } = AntApp.useApp();
  const [saved] = useState(readSavedBuild);
  const [advanced, setAdvanced] = useState(false);
  const [bitChoice, setBitChoice] = useState<number | 'auto' | 'source'>('auto');
  const [originalModel, setOriginalModel] = useState('');
  const [serverInUse, setServerInUse] = useState(false);
  const [serverPollError, setServerPollError] = useState<string | null>(null);
  const [activeServers, setActiveServers] = useState<Array<{build_id:string;status:string;base_url:string}>>([]);
  useEffect(() => {
    let cancelled = false;
    const refresh = () => api.servers().then(result => { if (!cancelled) {setActiveServers(result.servers);setServerInUse(result.servers.length > 0);setServerPollError(null);} }).catch(error => {if (!cancelled) setServerPollError((error as Error).message);});
    refresh(); const timer = setInterval(refresh,2500);
    return () => {cancelled=true;clearInterval(timer);};
  }, []);
  const stopActiveServers = async () => {
    try { const result = await api.servers('DELETE');setActiveServers(result.servers);setServerInUse(result.servers.length > 0); }
    catch (error) {message.error((error as Error).message);}
  };

  const [qualityEnabled, setQualityEnabled] = useState(false);
  const checkSequence = useRef(0);
  const [checking, setChecking] = useState(false);
  const [checked, setChecked] = useState<Awaited<ReturnType<typeof api.precheck>> | null>(null);
  const [stage, setStage] = useState(saved ? 3 : 0);
  const [baseError, setBaseError] = useState<string | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);
  const plannedConfig = useRef('');

  const [meta, setMeta] = useState<Meta | null>(null);
  const [metaError, setMetaError] = useState<string | null>(null);
  const [config, setConfig] = useState<ForgeConfig>(saved?.config ?? INITIAL_CONFIG);

  // What the user types, kept separate from the normalized backend config.
  const [inputPath, setInputPath] = useState(saved?.inputPath ?? new URLSearchParams(window.location.search).get('model') ?? '');
  const [baseDir, setBaseDir] = useState(saved?.baseDir ?? '');
  const outputTouched = useRef(Boolean(saved));
  const managedTernary = useRef('');

  const [inspection, setInspection] = useState<ModelInspection | null>(null);
  const [inspectLoading, setInspectLoading] = useState(false);
  const [inspectError, setInspectError] = useState<string | null>(null);
  const [baseInspection, setBaseInspection] = useState<ModelInspection | null>(null);

  const [plan, setPlan] = useState<ForgePlan | null>(null);
  const [capabilities, setCapabilities] = useState<BackendCapability[]>([]);
  // Geometry-filtered recipes from the same gate the plan uses; null until the
  // first plan arrives, so ConfigPanel falls back to the full meta list.
  const [planRecipes, setPlanRecipes] = useState<Recipe[] | null>(null);
  const [planLoading, setPlanLoading] = useState(false);
  const [planError, setPlanError] = useState<string | null>(null);

  const [buildId, setBuildId] = useState<string | null>(saved?.id ?? null);
  const [building, setBuilding] = useState(Boolean(saved));
  const [buildStatus, setBuildStatus] = useState<BuildStatus | null>(null);
  const [log, setLog] = useState('');
  const [startMs, setStartMs] = useState<number | null>(saved?.startMs ?? null);
  const [performance, setPerformance] = useState<Report | null>(null);
  const [quality, setQuality] = useState<Report | null>(null);

  const patch = (p: Partial<ForgeConfig>) => setConfig((c) => ({ ...c, ...p }));

  const chooseModel = (value: string) => {
    checkSequence.current++; setChecking(false); setOriginalModel('');
    sessionStorage.removeItem('turboforge.build'); setBuildId(null); setBuildStatus(null); setPlan(null); setLog(''); setPerformance(null); setQuality(null); setBitChoice('auto'); setQualityEnabled(false);
    setInputPath(value); setChecked(null); setStage(0); setBaseDir('');
    setConfig(c => ({...c, jailbreak:'none', jailbreak_params:{}, target:'auto', sources:{}, benchmark:{}, quantization:{}, runtime:{...c.runtime, speculative:'auto'}}));
  };
  const checkModel = async () => {
    const sequence = ++checkSequence.current;
    setChecking(true); setChecked(null);
    try {
      const result = await api.precheck(inputPath.trim(), originalModel);
      if (sequence !== checkSequence.current) return;
      setChecked(result);
      if (result.ready) { setStage(1); setBitChoice('auto'); setQualityEnabled(false); setBuildId(null); setBuildStatus(null); setPlan(null); setPerformance(null); setQuality(null); sessionStorage.removeItem('turboforge.build'); patch({jailbreak:'none', jailbreak_params:{}, target:result.requires_explicit_target ? '' : 'auto', sources:{}, benchmark:{}, quantization:{}, runtime:{...config.runtime,speculative:'auto',...(result.recommended_max_context ? {max_context:result.recommended_max_context} : {})}, run_benchmark:true}); }
    } catch (error) { if (sequence === checkSequence.current) setChecked({ready:false, message:`模型检查未完成：${(error as Error).message}`}); }
    finally { if (sequence === checkSequence.current) setChecking(false); }
  };
  const chooseOriginal = async () => {
    try { const picked = await api.pickPath('directory'); if (picked.path) chooseModel(picked.path); }
    catch (error) { message.error((error as Error).message); }
  };

  const needsBase = inspection?.input_kind === 'nvfp4' || inspection?.input_kind === 'gguf';

  // Load the single source of truth once, and seed the serving defaults from it
  // so the UI controls and the plan start from the same values (no drift). The
  // benchmark stays empty until edited so the backend keeps its geometry-aware
  // defaults (e.g. KV dtype) rather than receiving a pinned value.
  useEffect(() => {
    api
      .meta()
      .then((m) => {
        setMeta({...m,recipes:[...m.recipes,...(m.hf_recipes ?? [])]});
        setConfig((c) => ({ ...c, device: saved ? c.device : m.defaults.device,
          upstream_dir: c.upstream_dir ?? m.defaults.upstream_dir, ninfer_dir: c.ninfer_dir ?? m.defaults.ninfer_dir,
          runtime: { ...m.defaults.runtime, ...c.runtime } }));
      })
      .catch((e) => setMetaError((e as Error).message));
  }, []);

  // Auto-inspect whatever the user typed (HF dir, NVFP4 checkpoint, GGUF file).
  useEffect(() => {
    const path = inputPath.trim();
    if (/^[\w.-]+\/[\w.-]+$/.test(path) || path.startsWith('https://huggingface.co/')) {
      setInspection(null); setInspectError(null); setInspectLoading(false);
      return;
    }
    if (!path) {
      setInspection(null);
      setInspectError(null);
      setInspectLoading(false);
      return;
    }
    let cancelled = false;
    setInspectLoading(true);
    const t = setTimeout(async () => {
      try {
        const res = await api.inspect(path);
        if (!cancelled) {
          setInspection(res.inspection);
          setInspectError(null);
        }
      } catch (e) {
        if (!cancelled) {
          setInspection(null);
          setInspectError((e as Error).message);
        }
      } finally {
        if (!cancelled) setInspectLoading(false);
      }
    }, 500);
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
  }, [inputPath]);

  // NVFP4/GGUF routes keep the original HF dir as --model; inspect it too so
  // recipe candidates reflect the architecture the converter will actually see.
  useEffect(() => {
    if (!needsBase || !baseDir.trim()) {
      setBaseInspection(null);
      setBaseError(null);
      return;
    }
    let cancelled = false;
    const t = setTimeout(async () => {
      try {
        const res = await api.inspect(baseDir.trim(), inspection?.input_kind === 'gguf');
        if (!cancelled) { setBaseInspection(res.inspection); setBaseError(null); }
      } catch (error) {
        if (!cancelled) { setBaseInspection(null); setBaseError((error as Error).message); }
      }
    }, 500);
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
  }, [baseDir, needsBase, inspection?.input_kind]);

  // Normalize the typed path into the verified backend contract: the HF dir is
  // always input_path (--model); a typed NVFP4/GGUF becomes a named --source.
  useEffect(() => {
    const typed = inputPath.trim();
    const oldTernary = managedTernary.current;
    const ternaryInput = inspection?.quantization.ternary_pq2 === true;
    managedTernary.current = ternaryInput ? typed : '';
    setConfig((c) => {
      const sources = { ...c.sources };
      delete sources.quantized;
      delete sources.gguf;
      if (oldTernary && sources.ternary === oldTernary) delete sources.ternary;
      let nextInput = typed;
      if (needsBase && inspection) {
        if (inspection.input_kind === 'nvfp4') sources.quantized = typed;
        if (inspection.input_kind === 'gguf') {
          if (ternaryInput) sources.ternary = typed;
          else sources.gguf = typed;
        }
        if (baseDir.trim()) nextInput = baseDir.trim();
      }
      if (c.input_path === nextInput && JSON.stringify(c.sources) === JSON.stringify(sources)) {
        return c;
      }
      return { ...c, input_path: nextInput, sources };
    });
  }, [inspection, inputPath, baseDir, needsBase]);

  // Suggest an output dir until the user edits it themselves.
  useEffect(() => {
    if (outputTouched.current) return;
    const suggested = suggestOutput(inputPath);
    setConfig((c) => (c.output_dir === suggested ? c : { ...c, output_dir: suggested }));
  }, [inputPath]);

  // Auto-plan on any config change (debounced), so route/resources/risks
  // stay live without the user clicking "Plan" over and over.
  useEffect(() => {
    if (!advanced) {
      setPlan(null); setPlanLoading(false); setPlanError(null);
      return;
    }
    if (saved && buildId === saved.id && config.output_dir === saved.config.output_dir) {
      // A resumed job already owns this directory. Its recorded plan is the
      // authoritative preview; replanning it would falsely report non-empty output.
      setPlanLoading(false);
      return;
    }
    if (!config.input_path.trim() || !config.output_dir.trim()) {
      setPlan(null);
      setCapabilities([]);
      setPlanRecipes(null);
      setPlanError(null);
      setPlanLoading(false);
      return;
    }
    let cancelled = false;
    setPlanLoading(true);
    const t = setTimeout(async () => {
      try {
        const res = await api.plan(cleanConfig(config));
        if (!cancelled) {
          setPlan(res.plan);
          setCapabilities(res.capabilities);
          setPlanRecipes(res.recipes);
          setPlanError(null);
          plannedConfig.current = JSON.stringify(cleanConfig(config));
        }
      } catch (e) {
        if (!cancelled) {
          setPlan(null);
          setPlanRecipes(null);
          setPlanError((e as Error).message);
        }
      } finally {
        if (!cancelled) setPlanLoading(false);
      }
    }, 600);
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
  }, [config, advanced]);

  // Live build polling: status + converter log, then reports on completion.
  useEffect(() => {
    if (!buildId || !building) return;
    let cancelled = false;
    let inFlight = false;
    const tick = async () => {
      if (inFlight || cancelled) return;
      inFlight = true;
      try {
        const st = await api.status(buildId);
        if (cancelled) return;
        setBuildStatus(st);
        setPollError(null);
        if (st.manifest?.plan) setPlan(st.manifest.plan);
        const lg = await api.log(buildId);
        if (!cancelled) setLog(lg.log);
        if (st.finished) {
          const [p, q] = await Promise.all([
            api.report(buildId, 'performance').catch(() => null),
            api.report(buildId, 'quality').catch(() => null),
          ]);
          if (cancelled) return;
          setPerformance(p);
          setQuality(q);
          setBuilding(false);
        }
      } catch (error) {
        if (!cancelled) setPollError(`进度连接中断：${(error as Error).message}。正在重新连接，请勿重复开始转换。`);
        if (!cancelled && (error as Error).message === 'unknown build id') {
          setBuilding(false);
          setPollError('本机服务已重启，任务记录不在内存中。请查看输出目录的 manifest.json 和 logs；更换空目录后可重新构建。');
          sessionStorage.removeItem('turboforge.build');
        }
      } finally {
        inFlight = false;
      }
    };
    tick();
    const iv = setInterval(tick, 1500);
    return () => {
      cancelled = true;
      clearInterval(iv);
    };
  }, [buildId, building]);

  const onStart = async () => {
        if (!canBuild) return;
        setBuilding(true);
        try {
          const res = await api.build(cleanConfig(config));
          sessionStorage.setItem('turboforge.build', JSON.stringify({id:res.build_id, config:cleanConfig(config), inputPath, baseDir, startMs:Date.now()}));
          setBuildId(res.build_id);
          setBuildStatus(null);
          setLog('');
          setPerformance(null);
          setQuality(null);
          setStartMs(Date.now());
          setBuilding(true);
        } catch (e) {
          setBuilding(false);
          message.error((e as Error).message);
        }
  };

  const onAutoStart = async () => {
    if (!inputPath.trim() || building || serverInUse) return;
    setBuilding(true); setPollError(null); setStage(3);
    try {
      const options: Record<string, unknown> = {target:config.target, jailbreak:config.jailbreak, runtime:{speculative:config.runtime.speculative ?? 'auto',max_context:config.runtime.max_context ?? 32768,max_concurrency:config.runtime.max_concurrency ?? 1}, run_benchmark:config.run_benchmark,
        benchmark:qualityEnabled ? config.benchmark : {}, quantization:config.quantization ?? {}, sources:Object.fromEntries(Object.entries(config.sources).filter(([,value]) => value.trim())),
        ...(originalModel.trim() ? {original_model:originalModel.trim()} : {})};
      const result = await api.autoBuild(inputPath.trim(), options);
      setPlan(null); setBuildStatus(null); setLog(''); setPerformance(null); setQuality(null); setStartMs(Date.now());
      setBuildId(result.build_id);
      sessionStorage.setItem('turboforge.build', JSON.stringify({id:result.build_id, config, inputPath, baseDir:'', startMs:Date.now()}));
    } catch (error) {
      setBuilding(false); setStage(checked?.ready ? 1 : 0); message.error((error as Error).message);
    }
  };

  const canBuild = Boolean(plan?.executable) && !building && !planLoading && !inspectLoading && !inspectError
    && (!needsBase || Boolean(baseInspection)) && plannedConfig.current === JSON.stringify(cleanConfig(config))
    && !(buildStatus?.finished && buildStatus.output_dir === config.output_dir);
  const routeInspection = needsBase ? baseInspection : inspection;

  const titles = ['选择模型', '量化与越狱', '加速与验证', '转换并使用'];
  const inputReady = Boolean(inspection?.usable && config.output_dir.trim() && !inspectLoading && !inspectError && (!needsBase || baseInspection?.usable));
  const nextReady = stage === 0 ? inputReady : Boolean(plan?.executable && !planLoading);
  const go = (value: number) => { if (serverInUse && value < 3) {message.warning('请先停止模型服务，再开始新的转换');return;} setStage(!advanced && value < 3 ? 0 : value); window.scrollTo({ top: 0, behavior: 'smooth' }); };
  const displayPlan = buildStatus?.manifest?.plan ?? plan;
  const actualConfig = buildStatus?.manifest?.config ?? buildStatus?.prepared_config ?? (!advanced ? { ...config, output_dir: '' } : config);
  const actualInspection = buildStatus?.manifest?.source ?? routeInspection ?? inspection;
  const companionSources = Object.fromEntries(Object.entries(config.sources).filter(([name]) => name in (checked?.optional_sources ?? {})));
  const hfSelected = config.target.startsWith('hf_');
  const selectTarget = (target: string) => {
    const hf = target.startsWith('hf_');
    let benchmark = {...config.benchmark};
    if (hf) delete benchmark.baseline_artifact;
    if (hf !== hfSelected || target === 'hf_packed_import' || config.target === 'hf_packed_import') {
      const recommendedQuality = hf && target !== 'hf_packed_import';
      setQualityEnabled(recommendedQuality);
      benchmark = recommendedQuality ? {auto_quality:true,max_ppl_relative_increase:.1} : {};
    }
    patch({target, sources:hf ? {} : companionSources, benchmark, quantization:hf ? config.quantization : {},
      run_benchmark:hf ? true : config.run_benchmark,
      runtime:hf ? {...config.runtime,speculative:config.runtime.speculative ?? 'auto',max_context:Math.min(config.runtime.max_context ?? 2048,2048),max_concurrency:1} : config.runtime});
  };
  const selectedSpec = String(config.runtime.speculative ?? 'auto');
  const specAvailable = hfSelected ? checked?.detected_components?.length ? ['none'] : ['auto','none'] : Array.from(new Set([...(checked?.speculation ?? ['auto','none']), ...['mtp','dflash2'].filter(key => config.sources[key]?.trim())]));
  const specSelectionValid = specAvailable.includes(selectedSpec);


  return (
    <AssistantProvider context={{buildId,modelIdentity:inputPath,summary:{architecture:actualInspection?.architecture,
      input_kind:actualInspection?.input_kind,weight_precision:actualInspection?.weight_precision,target:config.target,
      engine:hfSelected ? 'gptqmodel' : 'ninfer',speculative:selectedSpec,stage:buildStatus?.status ?? stage,
      diagnostics:displayPlan?.diagnostics ?? []}}}>
    <div className="shell">
      <AppHeader meta={meta} />
      {serverPollError && meta && stage !== 3 && <Notice tone="warn" title="模型服务状态暂时无法连接">正在重新连接：{serverPollError}</Notice>}
      {activeServers.length > 0 && (stage !== 3 || activeServers.some(server => server.build_id !== buildId)) && <Notice title="本页面服务已有模型正在运行"><p>开始新转换前先释放显存；即使切换任务，也可在这里停止服务。</p><Button onClick={stopActiveServers}>停止当前模型并释放显存</Button></Notice>}
      <div className="wizard-heading">
        <h1>选择量化与加速方案，生成可验证的本地模型</h1>
        <p>选择待处理模型；量化模型无法识别原始来源时，还需填写原始模型目录或仓库链接。脚本自动准备依赖、引擎和输出位置，完成转换与 GPU 验证；原模型始终保留。</p>
        <Steps current={advanced ? stage : stage === 3 ? 2 : stage} items={(advanced ? titles : ['选择模型', '确认处理方式', '转换并使用']).map((title) => ({ title }))} />
      </div>
      <main className="workspace wizard-workspace">
        <div className="col">
          <Panel title={advanced ? `${stage + 1}. ${titles[stage]}` : stage === 0 ? '1. 先选择模型' : stage === 1 ? '2. 确认处理方式' : '3. 转换并使用'} icon={<SettingOutlined />}>
            {metaError && <Notice tone="bad" title="无法连接本机服务">{metaError}<br /><Button onClick={() => window.location.reload()}>重新连接</Button></Notice>}
            {stage === 0 && <Notice title="先选模型，再确认可用的处理方式">选择本机模型，或输入 Hugging Face 的「组织/模型名」、仓库链接。首次需要联网下载；已准备好的文件自动复用，下载进度会显示在页面。</Notice>}
            {meta ? <>
              {stage === 0 && !advanced && <>
                <Field label="模型" required tooltip="HF / NVFP4 选目录，GGUF 选文件；也可输入组织/模型名。">
                  <PathInput label="输入模型" kind="model" value={inputPath} onChange={chooseModel} placeholder="选择本机模型，或填写 Hugging Face 仓库链接" />
                </Field>
                {checked && !checked.ready && <Notice tone="warn" title={checked.action === 'provide_original' ? '需补充原始模型来源' : checked.action === 'choose_original' ? '请改用原始模型进行转换' : '模型检查未通过'}>{checked.message}
                  {checked.action === 'provide_original' && <p>在下方选择本机原始模型目录，或填写原始模型仓库链接，然后点击「检查模型并继续」。点击「选择文件夹」可自动填入目录路径，也可以手动填写。</p>}
                  {checked.action === 'choose_original' && <p><Button onClick={chooseOriginal}>改用原始模型文件夹</Button></p>}
                </Notice>}
                <Field label="原始模型来源" required={checked?.action === 'provide_original'} extra="待处理模型为完整原始 HF 模型时请留空。已支持的完整 GGUF 可自动生成配置与分词器，不需填写来源。其他 GGUF / NVFP4 在提示缺少来源时填写：选择含 config.json 与分词器文件的原始 HF 目录，或粘贴其仓库链接；仅含 GGUF 的目录不能作为原始来源。"><PathInput label="原始模型来源" value={originalModel} onChange={value => {checkSequence.current++; setChecking(false);setOriginalModel(value);setChecked(null);}} placeholder="本机原始模型目录路径，或 Hugging Face 仓库链接" /></Field>
                {!!checked?.input_candidates?.length && <Field label="仓库 GGUF 文件" required><Select placeholder="选择具体精度文件" options={checked.input_candidates.map(item => ({value:item.reference,label:item.name}))} onChange={chooseModel} /></Field>}
                <div className="wizard-actions"><Button disabled={building || checking} onClick={() => {setAdvanced(true); patch({jailbreak:'none', jailbreak_params:{}});}}>高级手动配置</Button><Button type="primary" disabled={!inputPath.trim() || building} loading={checking} onClick={checkModel}>检查模型并继续</Button></div>
              </>}
              {!advanced && stage === 1 && checked?.ready && <>
                <p className="mono path-wrap" data-ai-ignore>{inputPath}</p>
                <Notice tone="ok" title={checked.remote ? '已选择模型仓库' : '模型检查完成'}>{checked.message}</Notice>
                {(checked.architecture_kind && checked.architecture_kind !== 'unknown' || checked.gpu) && <div className="chip-row">
                  {checked.architecture_kind === 'moe' && <Chip tone="ok">MoE 混合专家</Chip>}
                  {checked.architecture_kind === 'dense' && <Chip tone="ok">Dense 稠密</Chip>}
                  {checked.gpu && <Chip mono>{checked.gpu.generation} · {Math.round(checked.gpu.total_bytes / 1024 ** 3)} GB</Chip>}
                  {checked.gpu && <Chip>{checked.scheme_order === 'native-format' ? '本卡原生格式已排到前面' : '本卡无原生低比特格式，按位宽排序'}</Chip>}
                </div>}
                {checked.runtime_engine && <Notice tone={checked.runtime_engine.compat_check ? 'warn' : 'info'} title={`后续推理建议 · ${checked.runtime_engine.label}`}>
                  {checked.runtime_engine.compat_check && <p><strong>{checked.runtime_engine.compat_check}</strong></p>}
                  <p>{checked.runtime_engine.reason}</p>
                  <p className="muted">要求：{checked.runtime_engine.requirements}</p>
                  <p className="muted">格式：{checked.runtime_engine.format} · 支持模型：{checked.runtime_engine.models}</p>
                  <p className="muted">备选：{checked.runtime_engine.alternative}</p>
                  <p className="muted">后续步骤：{checked.runtime_engine.steps}</p>
                  <p className="muted">{checked.runtime_engine.advisory} <a href={checked.runtime_engine.url} target="_blank" rel="noreferrer">官方仓库 {checked.runtime_engine.url}</a></p>
                </Notice>}
                <h3>量化方式与位宽</h3>
                <p className="muted">下方直接列出当前模型＋引擎已验证的全部量化方案，位宽、算法与引擎都写在选项里，由你自选;「自动推荐」只遵循该模型默认策略，不会替你改目标。可选位宽仅来自这些已适配配方，未验证的位宽不列出，所以菜单可能较短。</p>
                <Field label="目标位宽"><Select value={bitChoice} options={[{value:'auto',label:checked.provided_sources?.length ? '按来源精度导入' : '全部位宽 · 自动推荐（保留当前模型默认质量策略）'}, ...Array.from(new Set((checked.targets ?? []).filter(t => t.available && t.bits !== null).map(t => t.bits as number))).sort((a,b) => a-b).map(bits => ({value:bits,label:`${bits} bit · 主权重精度`})), ...((checked.targets ?? []).some(t => t.available && t.bits === null) ? [{value:'source',label:'保持来源位宽'}] : [])]} onChange={value => {
                  setBitChoice(value); selectTarget(checked.requires_explicit_target && value === 'auto' ? '' : value === 'auto' ? 'auto' : checked.targets?.find(t => t.available && t.bits === (value === 'source' ? null : value))?.value ?? '');
                }} /></Field>
                <Field label="量化方案"><Select placeholder="选择与来源布局匹配的方案" value={config.target || undefined} options={[...(bitChoice === 'auto' && !checked.requires_explicit_target ? [{value:'auto',label:'自动推荐'}] : []), ...(checked.targets ?? []).filter(t => bitChoice === 'auto' || t.bits === (bitChoice === 'source' ? null : bitChoice)).map(t => ({value:t.value,label:t.label + (t.bits !== null ? ` · ${t.bits}bit` : ' · 来源位宽') + (t.engine === 'ninfer' ? ' · NInfer' : ' · HF') + (t.gpu_note ? ` · ${t.gpu_note}` : '') + (t.missing_sources?.length ? '（需补充来源）' : ''),disabled:!t.available}))]} onChange={selectTarget} /></Field>
                {checked.requires_explicit_target && <Notice title="请明确选择量化路线">按检查点布局和目标格式选择方案；程序不会猜测生产工具或自动切换推理引擎。</Notice>}
                {(checked.targets?.find(t => t.value === config.target)?.required_sources ?? []).filter(name => !checked.provided_sources?.includes(name)).map(name => <Field key={name} label={checked.source_fields?.[name]?.label ?? name} required><PathInput label={`来源 ${name}`} kind={checked.source_fields?.[name]?.kind ?? 'file'} value={config.sources[name] ?? ''} onChange={value => patch({sources:{...config.sources,[name]:value}})} /></Field>)}
                <p className="muted">只列出此模型可用的位宽。嵌入、输出头等可能保留更高精度；主位宽、整模型 bpw 与文件大小不同。</p>
                {!!checked.unsupported?.length && <Collapse ghost items={[{key:'unsupported', label:'格式与位宽的支持范围', children:checked.unsupported.map(item => <p key={item.name}>{item.reason}</p>)}]} />}
                {hfSelected && <Collapse ghost items={[{key:'hf-options',label:'调整校准与推理后端（可选，默认自动准备）',children:<>
                  <Notice title="HF 原生 AWQ/GPTQ 路线">保留原生打包格式，使用独立 GPTQModel 引擎。首次自动准备 CUDA 13 环境，需要 NVIDIA 580+ 驱动。本机 Qwen3.5-0.8B 的 GPTQ 4-bit 已通过有限质量验收；其他模型与位宽以本次验证报告为准。</Notice>
                  {config.target !== 'hf_packed_import' && <>
                    <Field label="校准语料"><PathInput label="量化校准文本" kind="file" value={String(config.quantization?.calibration_file ?? '')} onChange={value => patch({quantization:{...config.quantization,calibration_file:value}})} placeholder="可选；留空自动下载 WikiText-2" /></Field>
                    <Field label="分组大小"><Select value={Number(config.quantization?.group_size ?? 128)} options={[32,64,128].map(value => ({value,label:String(value)}))} onChange={value => patch({quantization:{...config.quantization,group_size:value}})} /></Field>
                    <Collapse ghost items={[{key:'hf-calibration',label:'调整校准样本与长度（可选）',children:<>
                      <Field label="校准样本数"><InputNumber min={1} max={4096} value={Number(config.quantization?.samples ?? 128)} onChange={value => {if(value !== null) patch({quantization:{...config.quantization,samples:value}});}} /></Field>
                      <Field label="每样本 token"><InputNumber min={32} max={4096} value={Number(config.quantization?.sequence_length ?? 512)} onChange={value => {if(value !== null) patch({quantization:{...config.quantization,sequence_length:value}});}} /></Field>
                    </>}]} />
                  </>}
                  <Field label="HF 推理后端"><Select value={String(config.quantization?.backend ?? 'auto')} options={[{value:'auto',label:'自动优化（推荐，首次编译后缓存）'},{value:'torch',label:'Torch 兼容后端'}]} onChange={value => patch({quantization:{...config.quantization,backend:value}})} /></Field>
                  <p className="muted">校准以 FP16、batch=1 执行，样本不重叠，至少需要 {Number(config.quantization?.samples ?? 128)*Number(config.quantization?.sequence_length ?? 512)} token；校准与评测请使用独立语料。激活放 CPU、已处理层卸载到磁盘，会增加 RAM、磁盘和转换时间；低位宽仍需质量验收。</p>
                </>}]} />}
                <h3>权重处理</h3>
                <Field label="越狱方式"><Select value={config.jailbreak} options={[{value:'none', label:'保持现有行为（推荐，已有越狱模型也选此项）'}, ...(checked.can_edit_weights ? [{value:'abliteration',label:'Abliteration · 自动校准拒绝方向'}] : [])]} onChange={value => patch({jailbreak:value,jailbreak_params:{}})} /></Field>
                <p className="muted">越狱会改变模型行为，本身不保证提速；Heretic 的交互处理仍在高级配置中。</p>
                <h3>解码加速</h3>
                <Field label="推测解码"><Select value={config.runtime.speculative ?? 'auto'} options={Array.from(new Set([...specAvailable, selectedSpec])).map(value => ({value, disabled:!specAvailable.includes(value), label:!specAvailable.includes(value) ? `${value.toUpperCase()}（${hfSelected ? 'HF 路线不执行推测头' : '缺少组件来源'}）` : ({auto:'自动使用兼容组件',none:'关闭推测解码',mtp:config.sources.mtp ? '外部 MTP（转换时验证）' : 'MTP · 已检测到组件',dflash2:config.sources.dflash2 ? '外部 DFlash2（转换时验证）' : 'DFlash2 · 已检测到组件'} as Record<string,string>)[value]}))} onChange={value => patch({runtime:{...config.runtime,speculative:value}})} /></Field>
                <p className="muted">{hfSelected ? 'HF 路线无损归档自带 MTP/DFlash2，但当前服务不执行这些头。要保留推测加速请选择 NInfer；仅需归档时请明确关闭推测。' : '自带 MTP/DFlash2 自动保留；转换后校验实际权重绑定与共享输出头。只封装已有组件，不训练缺失的头。'}</p>
                {!hfSelected && !!Object.keys(checked.optional_sources ?? {}).length && <Collapse ghost items={[{key:'companions',label:'添加已有的兼容组件（可选）',children:Object.entries(checked.optional_sources ?? {}).map(([name,field]) => <Field key={name} label={field.label}><PathInput label={`来源 ${name}`} kind={field.kind} value={config.sources[name] ?? ''} onChange={value => patch({sources:{...config.sources,[name]:value}})} /></Field>)}]} />}
                {!specSelectionValid && <Notice tone="warn">{hfSelected ? '此模型自带推测头，当前 HF 引擎不能执行。请选择 NInfer 路线保留加速，或明确关闭推测后仅保留头文件。不会自动降级。' : '所选推测组件缺少来源，请补全文件，或明确改选自动/关闭。'}
                  {hfSelected && checked.targets?.some(target => target.engine === 'ninfer' && target.available) && <p><Button onClick={() => {setBitChoice('auto');selectTarget('auto');patch({runtime:{...config.runtime,speculative:'auto'}});}}>改用 NInfer，自动保留并使用兼容推测头</Button></p>}
                </Notice>}
                <h3>性能与质量验收</h3>
                <Field label="测试速度"><Switch checked={config.run_benchmark} disabled={qualityEnabled} onChange={value => patch({run_benchmark:value})} /></Field>
                <Field label="质量门禁"><Switch checked={qualityEnabled} disabled={config.target === 'hf_packed_import'} onChange={value => {setQualityEnabled(value);patch({run_benchmark:true, benchmark:value ? {...config.benchmark,...(hfSelected && !config.benchmark.perplexity_text ? {auto_quality:true} : {}),max_ppl_relative_increase:0.1} : {}});}} /></Field>
                {qualityEnabled ? <>
                  {hfSelected ? <Notice title="自动准备独立质量评测">默认下载固定版本 WikiText-2 test，取开头最多 2048 token，与 train 校准分开。自动使用本次处理后、量化前的权重，在 CPU 上评分；需要额外 RAM 和时间。通用英文语料不替代你的中文、代码等任务验证。</Notice> : <Notice tone="warn">此路线需要同一处理阶段的高精度 NInfer 对照文件。程序不会拿低精度产物自证无损；没有对照时只能确认可运行和所测速率。</Notice>}
                  <Field label="评测语料" required={!hfSelected}><PathInput label="质量评测语料" kind="file" value={String(config.benchmark.perplexity_text ?? '')} onChange={value => {const benchmark = {...config.benchmark}; delete benchmark.perplexity_text; delete benchmark.auto_quality; patch({benchmark:{...benchmark,...(value ? {perplexity_text:value} : hfSelected ? {auto_quality:true} : {})}});}} placeholder={hfSelected ? '可选；留空自动准备独立测试集' : '选择 UTF-8 文本'} /></Field>
                  {!hfSelected && <Field label="对照模型" required><PathInput label="对照 NInfer 模型" kind="file" value={String(config.benchmark.baseline_artifact ?? '')} onChange={value => patch({benchmark:{...config.benchmark,baseline_artifact:value}})} placeholder="同一处理阶段的高精度 .ninfer" /></Field>}
                  <Field label="允许 PPL 增幅"><InputNumber min={0} max={100} suffix="%" value={Number(config.benchmark.max_ppl_relative_increase ?? .1)*100} onChange={value => patch({benchmark:{...config.benchmark,max_ppl_relative_increase:Number(value)/100}})} /></Field>
                  <p className="muted">对照模型应使用同一版本、分词器及越狱处理阶段的权重。两者按同一语料和上下文测 PPL；失败或超过阈值则阻止直接启动。此结果不替代代码、数学等任务评测。</p>
                </> : <Notice tone="warn" title="此次不验证量化质量">将验证 GPU 可运行，并按选择测速；质量保持未验证状态。若要比较量化损失，请开启质量门禁。</Notice>}
                <div className="wizard-actions"><Button disabled={building} onClick={() => setStage(0)}>重新选择模型</Button><Button type="primary" disabled={building || serverInUse || !config.target || !specSelectionValid || (checked.targets?.find(t => t.value === config.target)?.required_sources ?? []).some(name => !checked.provided_sources?.includes(name) && !config.sources[name]?.trim()) || (qualityEnabled && ((!config.benchmark.perplexity_text && !config.benchmark.auto_quality) || (!hfSelected && !config.benchmark.baseline_artifact)))} loading={building} onClick={onAutoStart}>{qualityEnabled ? '转换并验收质量' : '转换并验证可运行'}</Button></div>

              </>}

              {advanced && stage < 3 && <Button onClick={() => {setAdvanced(false); chooseModel(inputPath);}}>返回引导模式（推荐）</Button>}
              {advanced && stage < 3 && <ConfigPanel stage={stage} meta={meta} config={config} inputPath={inputPath} onInputPath={chooseModel}
                baseDir={baseDir} onBaseDir={setBaseDir} inspection={inspection} routeInspection={routeInspection}
                plan={plan} candidateRecipes={planRecipes} disabled={building}
                onOutputTouched={() => { outputTouched.current = true; }} onChange={patch} />}
              {stage === 0 && advanced && baseError && needsBase && <Notice tone="bad" title="原始 HF 目录还未就绪">{baseError}</Notice>}
              {stage === 0 && advanced && inspectError && <Notice tone="bad" title="无法识别模型">{inspectError}</Notice>}
              {stage === 0 && advanced && <Collapse ghost items={[{ key: 'setup', label: '高级路径设置（自动模式无需填写）', children: <>
                <p>本页面在当前电脑运行。首次使用可双击项目根目录的「启动 TurboForge.bat」，自动检查并安装项目依赖。</p>
                <Field label="NInfer 转换器源码目录" tooltip="选择包含 tools/convert 的 ninfer-all 源码目录。">
                  <PathInput label="转换器源码目录" value={config.upstream_dir ?? ''} onChange={(value) => patch({upstream_dir:value})} />
                </Field>
                <Field label="NInfer 引擎目录" tooltip="选择包含 ninfer.exe、ninfer-serve.exe 与依赖 DLL 的文件夹；小模型需要对应的适配版。">
                  <PathInput label="引擎目录" value={config.ninfer_dir ?? ''} onChange={(value) => patch({ninfer_dir:value})} />
                </Field>
                <p>缺少转换器：从 <a href="https://github.com/iamwavecut/ninfer-all" target="_blank" rel="noreferrer">NInfer 官方仓库</a>下载源码及匹配的 Windows 发布包，再选择上面的目录。量化模型、引擎版本与 GPU 必须兼容。</p>
                <p>自动越狱标定需 transformers；Heretic 需 heretic-llm，且保存模型时需要在独立窗口操作。</p>
              </> }]} />}
              {advanced && stage === 1 && <Notice title="量化与速度如何选择">NInfer 的三值与 GSQ Q2 需要现成来源；HF GPTQ 可校准为 2/3/4/8 bit。不同路线使用不同引擎，低位宽不保证更快或保持质量。</Notice>}
              {stage === 3 && <>
                <Notice title={advanced ? '配置确认后，点击下方开始转换' : building ? '自动任务正在运行' : '查看结果并启动使用'}>保持本机服务运行。流程会自动完成越狱（如选择）、量化、GPU 验证与实测（如开启）；通过后可直接在本页启动模型 API。</Notice>
                {checked?.runtime_engine && <Notice tone="info" title={`产物运行路线 · 配套引擎，另可选 ${checked.runtime_engine.label}`}>
                  <p>本页「启动模型服务」与 scripts/run.bat 用配套引擎运行本次产物（.ninfer→NInfer，AWQ/GPTQ→HF）。{checked.runtime_engine.label} 需单独下载：<a href={checked.runtime_engine.url} target="_blank" rel="noreferrer">{checked.runtime_engine.url}</a>。{checked.runtime_engine.steps}</p>
                </Notice>}
                {pollError && <Notice tone="warn">{pollError}</Notice>}
                <BuildPanel canBuild={canBuild} building={building} buildStatus={buildStatus} log={log} startMs={startMs}
                  performance={performance} quality={quality} onServerActive={setServerInUse} onStart={onStart} buildId={buildId} />
              </>}
              {(advanced || stage === 3) && <div className="wizard-actions">
                <Button disabled={stage === 0 || building || serverInUse} onClick={() => go(stage - 1)}>上一步</Button>
                <span className="muted">{stage < 3 ? `下一步：${titles[stage + 1]}` : serverInUse ? '请先停止服务，再开始新转换' : '原始模型不会被覆盖'}</span>
                {stage < 3 && <Button type="primary" disabled={!nextReady || building} loading={inspectLoading || planLoading} onClick={() => go(stage + 1)}>下一步</Button>}
              </div>}
            </> : !metaError && <Skeleton active paragraph={{rows:5}} />}
          </Panel>
          {stage > 0 && <Collapse ghost items={[{key:'technical', label:'查看完整技术计划与诊断', children:<PlanPreview plan={displayPlan} capabilities={capabilities} loading={planLoading} error={planError} /> }]} />}
          <StoragePanel disabled={building || serverInUse} />
        </div>
        <aside className="col wizard-aside">
          {!advanced && stage === 1 ? <Panel title="本次处理路线"><p>{inspection?.weight_precision ?? '仓库模型'} → {config.jailbreak === 'none' ? '保留原行为' : 'Abliteration'} → {checked?.targets?.find(t => t.value === config.target)?.label ?? '自动推荐量化'} → {!specSelectionValid ? '需确认推测设置' : hfSelected ? 'HF 基础解码' : ({auto:'自动选择兼容解码',none:'基础解码',mtp:'MTP 推测解码',dflash2:'DFlash2 推测解码'} as Record<string,string>)[selectedSpec]}</p><p>{qualityEnabled ? '验收：GPU 加载、速度实测、同语料 PPL 门禁' : '验收：GPU 加载' + (config.run_benchmark ? '与速度实测；质量未实测' : '；速度与质量未实测')}</p><p className="muted">不会擅自替换你指定的量化方案。原始模型保留，所需依赖自动准备。</p></Panel> : <WorkflowSummary concise={!advanced && stage < 3} plan={displayPlan} inspection={actualInspection} config={actualConfig} loading={planLoading || inspectLoading || buildStatus?.status === 'preparing'} />}
          {stage === 0 && inspection && <Collapse ghost items={[{key:'inspection', label:'查看模型识别详情', children:<InspectionCard inspection={inspection} loading={inspectLoading} error={inspectError} meta={meta} /> }]} />}
          {advanced && planError && <Notice tone="bad" title="配置未就绪">{planError}</Notice>}
        </aside>
      </main>
    </div>
    </AssistantProvider>
  );
}
