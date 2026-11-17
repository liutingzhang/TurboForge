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
    } catch (error) { if (sequence === checkSequence.current) setChecked({ready:false, message:`Model check incomplete: ${(error as Error).message}`}); }
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
        if (!cancelled) setPollError(`Progress connection interrupted: ${(error as Error).message}. Reconnecting; do not start the conversion again.`);
        if (!cancelled && (error as Error).message === 'unknown build id') {
          setBuilding(false);
          setPollError('The local service restarted and the task record is no longer in memory. Check manifest.json and logs in the output directory; you can rebuild after choosing a new empty directory.');
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

  const titles = ['Select model', 'Quantization & abliteration', 'Acceleration & validation', 'Convert & use'];
  const inputReady = Boolean(inspection?.usable && config.output_dir.trim() && !inspectLoading && !inspectError && (!needsBase || baseInspection?.usable));
  const nextReady = stage === 0 ? inputReady : Boolean(plan?.executable && !planLoading);
  const go = (value: number) => { if (serverInUse && value < 3) {message.warning('Stop the model service before starting a new conversion');return;} setStage(!advanced && value < 3 ? 0 : value); window.scrollTo({ top: 0, behavior: 'smooth' }); };
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
      {serverPollError && meta && stage !== 3 && <Notice tone="warn" title="Model service status temporarily unreachable">Reconnecting: {serverPollError}</Notice>}
      {activeServers.length > 0 && (stage !== 3 || activeServers.some(server => server.build_id !== buildId)) && <Notice title="A model from this page is already running"><p>Free VRAM before starting a new conversion; even after switching tasks, you can stop the service here.</p><Button onClick={stopActiveServers}>Stop the current model and free VRAM</Button></Notice>}
      <div className="wizard-heading">
        <h1>Choose a quantization and acceleration plan, produce a verifiable local model</h1>
        <p>Select the model to process; when a quantized model's original source cannot be identified, also provide the original model directory or repository link. The script prepares dependencies, engine and output location automatically, then completes conversion and GPU validation; the original model is always kept.</p>
        <Steps current={advanced ? stage : stage === 3 ? 2 : stage} items={(advanced ? titles : ['Select model', 'Confirm the plan', 'Convert & use']).map((title) => ({ title }))} />
      </div>
      <main className="workspace wizard-workspace">
        <div className="col">
          <Panel title={advanced ? `${stage + 1}. ${titles[stage]}` : stage === 0 ? '1. Select a model first' : stage === 1 ? '2. Confirm the plan' : '3. Convert & use'} icon={<SettingOutlined />}>
            {metaError && <Notice tone="bad" title="Cannot connect to the local service">{metaError}<br /><Button onClick={() => window.location.reload()}>Reconnect</Button></Notice>}
            {stage === 0 && <Notice title="Select a model first, then confirm the available plans">Select a local model, or enter a Hugging Face org/model name or repository link. The first run needs network downloads; prepared files are reused automatically and download progress is shown on the page.</Notice>}
            {meta ? <>
              {stage === 0 && !advanced && <>
                <Field label="Model" required tooltip="Choose a directory for HF / NVFP4, a file for GGUF; you can also enter an org/model name.">
                  <PathInput label="Input model" kind="model" value={inputPath} onChange={chooseModel} placeholder="Select a local model, or enter a Hugging Face repository link" />
                </Field>
                {checked && !checked.ready && <Notice tone="warn" title={checked.action === 'provide_original' ? 'Original model source needed' : checked.action === 'choose_original' ? 'Use the original model for conversion' : 'Model check failed'}>{checked.message}
                  {checked.action === 'provide_original' && <p>Select the local original model directory below, or enter the original model repository link, then click 'Check model and continue'. Clicking 'Select folder' fills the directory path automatically; you can also enter it manually.</p>}
                  {checked.action === 'choose_original' && <p><Button onClick={chooseOriginal}>Use the original model folder</Button></p>}
                </Notice>}
                <Field label="Original model source" required={checked?.action === 'provide_original'} extra="Leave empty when the model to process is a complete original HF model. Supported complete GGUFs generate config and tokenizer automatically and need no source. For other GGUF / NVFP4, fill this in when a missing source is flagged: select the original HF directory containing config.json and tokenizer files, or paste its repository link; a directory containing only GGUF cannot serve as the original source."><PathInput label="Original model source" value={originalModel} onChange={value => {checkSequence.current++; setChecking(false);setOriginalModel(value);setChecked(null);}} placeholder="Local original model directory path, or a Hugging Face repository link" /></Field>
                {!!checked?.input_candidates?.length && <Field label="Repository GGUF file" required><Select placeholder="Select the specific precision file" options={checked.input_candidates.map(item => ({value:item.reference,label:item.name}))} onChange={chooseModel} /></Field>}
                <div className="wizard-actions"><Button disabled={building || checking} onClick={() => {setAdvanced(true); patch({jailbreak:'none', jailbreak_params:{}});}}>Advanced manual configuration</Button><Button type="primary" disabled={!inputPath.trim() || building} loading={checking} onClick={checkModel}>Check model and continue</Button></div>
              </>}
              {!advanced && stage === 1 && checked?.ready && <>
                <p className="mono path-wrap" data-ai-ignore>{inputPath}</p>
                <Notice tone="ok" title={checked.remote ? 'Model repository selected' : 'Model check complete'}>{checked.message}</Notice>
                {(checked.architecture_kind && checked.architecture_kind !== 'unknown' || checked.gpu) && <div className="chip-row">
                  {checked.architecture_kind === 'moe' && <Chip tone="ok">MoE mixture of experts</Chip>}
                  {checked.architecture_kind === 'dense' && <Chip tone="ok">Dense</Chip>}
                  {checked.gpu && <Chip mono>{checked.gpu.generation} · {Math.round(checked.gpu.total_bytes / 1024 ** 3)} GB</Chip>}
                  {checked.gpu && <Chip>{checked.scheme_order === 'native-format' ? 'Native formats for this GPU ranked first' : 'No native low-bit format for this GPU; sorted by bit width'}</Chip>}
                </div>}
                {checked.runtime_engine && <Notice tone={checked.runtime_engine.compat_check ? 'warn' : 'info'} title={`Inference engine recommendation · ${checked.runtime_engine.label}`}>
                  {checked.runtime_engine.compat_check && <p><strong>{checked.runtime_engine.compat_check}</strong></p>}
                  <p>{checked.runtime_engine.reason}</p>
                  <p className="muted">Requirements: {checked.runtime_engine.requirements}</p>
                  <p className="muted">Format: {checked.runtime_engine.format} · supported models: {checked.runtime_engine.models}</p>
                  <p className="muted">Alternative: {checked.runtime_engine.alternative}</p>
                  <p className="muted">Next steps: {checked.runtime_engine.steps}</p>
                  <p className="muted">{checked.runtime_engine.advisory} <a href={checked.runtime_engine.url} target="_blank" rel="noreferrer">Official repository {checked.runtime_engine.url}</a></p>
                </Notice>}
                <h3>Quantization method and bit width</h3>
                <p className="muted">All quantization plans verified for the current model + engine are listed below, with bit width, algorithm and engine written into each option for you to choose; 'Auto recommendation' only follows this model's default policy and never changes the target for you. Selectable bit widths come only from these adapted recipes; unverified widths are not listed, so the menu may be short.</p>
                <Field label="Target bit width"><Select value={bitChoice} options={[{value:'auto',label:checked.provided_sources?.length ? 'Import at source precision' : 'All bit widths · auto recommendation (keeps the model default quality policy)'}, ...Array.from(new Set((checked.targets ?? []).filter(t => t.available && t.bits !== null).map(t => t.bits as number))).sort((a,b) => a-b).map(bits => ({value:bits,label:`${bits} bit · main-weight precision`})), ...((checked.targets ?? []).some(t => t.available && t.bits === null) ? [{value:'source',label:'Keep source bit width'}] : [])]} onChange={value => {
                  setBitChoice(value); selectTarget(checked.requires_explicit_target && value === 'auto' ? '' : value === 'auto' ? 'auto' : checked.targets?.find(t => t.available && t.bits === (value === 'source' ? null : value))?.value ?? '');
                }} /></Field>
                <Field label="Quantization plan"><Select placeholder="Select the plan matching the source layout" value={config.target || undefined} options={[...(bitChoice === 'auto' && !checked.requires_explicit_target ? [{value:'auto',label:'Auto recommendation'}] : []), ...(checked.targets ?? []).filter(t => bitChoice === 'auto' || t.bits === (bitChoice === 'source' ? null : bitChoice)).map(t => ({value:t.value,label:t.label + (t.bits !== null ? ` · ${t.bits}bit` : ' · source bit width') + (t.engine === 'ninfer' ? ' · NInfer' : ' · HF') + (t.gpu_note ? ` · ${t.gpu_note}` : '') + (t.missing_sources?.length ? ' (source needed)' : ''),disabled:!t.available}))]} onChange={selectTarget} /></Field>
                {checked.requires_explicit_target && <Notice title="Choose a quantization route explicitly">Choose the plan by checkpoint layout and target format; the program never guesses the producer tool or switches the inference engine automatically.</Notice>}
                {(checked.targets?.find(t => t.value === config.target)?.required_sources ?? []).filter(name => !checked.provided_sources?.includes(name)).map(name => <Field key={name} label={checked.source_fields?.[name]?.label ?? name} required><PathInput label={`Source ${name}`} kind={checked.source_fields?.[name]?.kind ?? 'file'} value={config.sources[name] ?? ''} onChange={value => patch({sources:{...config.sources,[name]:value}})} /></Field>)}
                <p className="muted">Only bit widths available for this model are listed. Embeddings, output heads etc. may keep higher precision; main bit width, whole-model bpw and file size are different things.</p>
                {!!checked.unsupported?.length && <Collapse ghost items={[{key:'unsupported', label:'Format and bit-width support scope', children:checked.unsupported.map(item => <p key={item.name}>{item.reason}</p>)}]} />}
                {hfSelected && <Collapse ghost items={[{key:'hf-options',label:'Adjust calibration and inference backend (optional; prepared automatically by default)',children:<>
                  <Notice title="HF native AWQ/GPTQ route">Keeps the native packed format and uses the separate GPTQModel engine. The first run prepares a CUDA 13 environment automatically and needs an NVIDIA 580+ driver. GPTQ 4-bit for Qwen3.5-0.8B passed limited quality acceptance on this machine; other models and bit widths follow this run's validation report.</Notice>
                  {config.target !== 'hf_packed_import' && <>
                    <Field label="Calibration corpus"><PathInput label="Quantization calibration text" kind="file" value={String(config.quantization?.calibration_file ?? '')} onChange={value => patch({quantization:{...config.quantization,calibration_file:value}})} placeholder="Optional; leave empty to download WikiText-2 automatically" /></Field>
                    <Field label="Group size"><Select value={Number(config.quantization?.group_size ?? 128)} options={[32,64,128].map(value => ({value,label:String(value)}))} onChange={value => patch({quantization:{...config.quantization,group_size:value}})} /></Field>
                    <Collapse ghost items={[{key:'hf-calibration',label:'Adjust calibration samples and length (optional)',children:<>
                      <Field label="Calibration samples"><InputNumber min={1} max={4096} value={Number(config.quantization?.samples ?? 128)} onChange={value => {if(value !== null) patch({quantization:{...config.quantization,samples:value}});}} /></Field>
                      <Field label="Tokens per sample"><InputNumber min={32} max={4096} value={Number(config.quantization?.sequence_length ?? 512)} onChange={value => {if(value !== null) patch({quantization:{...config.quantization,sequence_length:value}});}} /></Field>
                    </>}]} />
                  </>}
                  <Field label="HF inference backend"><Select value={String(config.quantization?.backend ?? 'auto')} options={[{value:'auto',label:'Auto optimization (recommended; cached after the first compile)'},{value:'torch',label:'Torch compatible backend'}]} onChange={value => patch({quantization:{...config.quantization,backend:value}})} /></Field>
                  <p className="muted">Calibration runs in FP16 with batch=1 and non-overlapping samples, needing at least {Number(config.quantization?.samples ?? 128)*Number(config.quantization?.sequence_length ?? 512)} tokens; use separate corpora for calibration and evaluation. Activations stay on the CPU and processed layers offload to disk, increasing RAM, disk and conversion time; low bit widths still need quality acceptance.</p>
                </>}]} />}
                <h3>Weight processing</h3>
                <Field label="Abliteration method"><Select value={config.jailbreak} options={[{value:'none', label:'Keep existing behavior (recommended; also choose this for already-abliterated models)'}, ...(checked.can_edit_weights ? [{value:'abliteration',label:'Abliteration · calibrates the refusal direction automatically'}] : [])]} onChange={value => patch({jailbreak:value,jailbreak_params:{}})} /></Field>
                <p className="muted">Abliteration changes model behavior and does not itself guarantee speedup; the interactive Heretic processing remains in advanced configuration.</p>
                <h3>Decode acceleration</h3>
                <Field label="Speculative decoding"><Select value={config.runtime.speculative ?? 'auto'} options={Array.from(new Set([...specAvailable, selectedSpec])).map(value => ({value, disabled:!specAvailable.includes(value), label:!specAvailable.includes(value) ? `${value.toUpperCase()} (${hfSelected ? 'the HF route does not run speculative heads' : 'component source missing'})` : ({auto:'Use compatible components automatically',none:'Speculative decoding off',mtp:config.sources.mtp ? 'External MTP (validated at conversion)' : 'MTP · component detected',dflash2:config.sources.dflash2 ? 'External DFlash2 (validated at conversion)' : 'DFlash2 · component detected'} as Record<string,string>)[value]}))} onChange={value => patch({runtime:{...config.runtime,speculative:value}})} /></Field>
                <p className="muted">{hfSelected ? 'The HF route archives bundled MTP/DFlash2 losslessly, but the current service does not run these heads. To keep speculative acceleration choose NInfer; for archiving only, explicitly turn speculation off.' : 'Bundled MTP/DFlash2 is retained automatically; after conversion the actual weight bindings and shared output head are verified. Only existing components are packaged; missing heads are never trained.'}</p>
                {!hfSelected && !!Object.keys(checked.optional_sources ?? {}).length && <Collapse ghost items={[{key:'companions',label:'Add existing compatible components (optional)',children:Object.entries(checked.optional_sources ?? {}).map(([name,field]) => <Field key={name} label={field.label}><PathInput label={`Source ${name}`} kind={field.kind} value={config.sources[name] ?? ''} onChange={value => patch({sources:{...config.sources,[name]:value}})} /></Field>)}]} />}
                {!specSelectionValid && <Notice tone="warn">{hfSelected ? 'This model ships speculative heads that the current HF engine cannot run. Choose the NInfer route to keep acceleration, or explicitly turn speculation off and keep only the head files. No automatic downgrade.' : 'The selected speculative component lacks a source; supply the file, or explicitly switch to auto/off.'}
                  {hfSelected && checked.targets?.some(target => target.engine === 'ninfer' && target.available) && <p><Button onClick={() => {setBitChoice('auto');selectTarget('auto');patch({runtime:{...config.runtime,speculative:'auto'}});}}>Switch to NInfer; keep and use compatible speculative heads automatically</Button></p>}
                </Notice>}
                <h3>Performance and quality acceptance</h3>
                <Field label="Benchmark speed"><Switch checked={config.run_benchmark} disabled={qualityEnabled} onChange={value => patch({run_benchmark:value})} /></Field>
                <Field label="Quality gate"><Switch checked={qualityEnabled} disabled={config.target === 'hf_packed_import'} onChange={value => {setQualityEnabled(value);patch({run_benchmark:true, benchmark:value ? {...config.benchmark,...(hfSelected && !config.benchmark.perplexity_text ? {auto_quality:true} : {}),max_ppl_relative_increase:0.1} : {}});}} /></Field>
                {qualityEnabled ? <>
                  {hfSelected ? <Notice title="Independent quality evaluation prepared automatically">By default a pinned WikiText-2 test revision is downloaded, taking up to the first 2048 tokens, separate from train calibration. The weights after this run's processing and before quantization are scored on the CPU automatically; extra RAM and time are needed. A general English corpus does not replace validation of your own tasks such as code or math.</Notice> : <Notice tone="warn">This route needs a high-precision NInfer baseline file from the same processing stage. The program never uses a low-precision artifact to prove itself lossless; without a baseline it can only confirm runnability and the measured speed.</Notice>}
                  <Field label="Evaluation corpus" required={!hfSelected}><PathInput label="Quality evaluation corpus" kind="file" value={String(config.benchmark.perplexity_text ?? '')} onChange={value => {const benchmark = {...config.benchmark}; delete benchmark.perplexity_text; delete benchmark.auto_quality; patch({benchmark:{...benchmark,...(value ? {perplexity_text:value} : hfSelected ? {auto_quality:true} : {})}});}} placeholder={hfSelected ? 'Optional; leave empty to prepare an independent test set automatically' : 'Select a UTF-8 text file'} /></Field>
                  {!hfSelected && <Field label="Baseline model" required><PathInput label="Baseline NInfer model" kind="file" value={String(config.benchmark.baseline_artifact ?? '')} onChange={value => patch({benchmark:{...config.benchmark,baseline_artifact:value}})} placeholder="A high-precision .ninfer from the same processing stage" /></Field>}
                  <Field label="Allowed PPL increase"><InputNumber min={0} max={100} suffix="%" value={Number(config.benchmark.max_ppl_relative_increase ?? .1)*100} onChange={value => patch({benchmark:{...config.benchmark,max_ppl_relative_increase:Number(value)/100}})} /></Field>
                  <p className="muted">The baseline model should use weights from the same revision, tokenizer and abliteration stage. Both are measured for PPL on the same corpus and context; failure or exceeding the threshold blocks direct start. This result does not replace task evaluations such as code or math.</p>
                </> : <Notice tone="warn" title="Quantization quality not validated this run">GPU runnability will be validated and speed measured if selected; quality stays unvalidated. To compare quantization loss, enable the quality gate.</Notice>}
                <div className="wizard-actions"><Button disabled={building} onClick={() => setStage(0)}>Reselect model</Button><Button type="primary" disabled={building || serverInUse || !config.target || !specSelectionValid || (checked.targets?.find(t => t.value === config.target)?.required_sources ?? []).some(name => !checked.provided_sources?.includes(name) && !config.sources[name]?.trim()) || (qualityEnabled && ((!config.benchmark.perplexity_text && !config.benchmark.auto_quality) || (!hfSelected && !config.benchmark.baseline_artifact)))} loading={building} onClick={onAutoStart}>{qualityEnabled ? 'Convert and accept quality' : 'Convert and validate runnability'}</Button></div>

              </>}

              {advanced && stage < 3 && <Button onClick={() => {setAdvanced(false); chooseModel(inputPath);}}>Back to guided mode (recommended)</Button>}
              {advanced && stage < 3 && <ConfigPanel stage={stage} meta={meta} config={config} inputPath={inputPath} onInputPath={chooseModel}
                baseDir={baseDir} onBaseDir={setBaseDir} inspection={inspection} routeInspection={routeInspection}
                plan={plan} candidateRecipes={planRecipes} disabled={building}
                onOutputTouched={() => { outputTouched.current = true; }} onChange={patch} />}
              {stage === 0 && advanced && baseError && needsBase && <Notice tone="bad" title="The original HF directory is not ready yet">{baseError}</Notice>}
              {stage === 0 && advanced && inspectError && <Notice tone="bad" title="Cannot identify the model">{inspectError}</Notice>}
              {stage === 0 && advanced && <Collapse ghost items={[{ key: 'setup', label: 'Advanced path settings (not needed in auto mode)', children: <>
                <p>This page runs on the current PC. On first use, double-click 'Start TurboForge.bat' in the project root to check and install project dependencies automatically.</p>
                <Field label="NInfer converter source directory" tooltip="Select the ninfer-all source directory containing tools/convert.">
                  <PathInput label="Converter source directory" value={config.upstream_dir ?? ''} onChange={(value) => patch({upstream_dir:value})} />
                </Field>
                <Field label="NInfer engine directory" tooltip="Select the folder containing ninfer.exe, ninfer-serve.exe and the dependency DLLs; small models need the matching adapted build.">
                  <PathInput label="Engine directory" value={config.ninfer_dir ?? ''} onChange={(value) => patch({ninfer_dir:value})} />
                </Field>
                <p>Missing converter: download the source and matching Windows release from the <a href="https://github.com/iamwavecut/ninfer-all" target="_blank" rel="noreferrer">official NInfer repository</a>, then select the directories above. The quantized model, engine version and GPU must be compatible.</p>
                <p>Automatic abliteration calibration needs transformers; Heretic needs heretic-llm, and saving the model is done in a separate window.</p>
              </> }]} />}
              {advanced && stage === 1 && <Notice title="How to choose quantization and speed">NInfer ternary and GSQ Q2 need ready-made sources; HF GPTQ can calibrate to 2/3/4/8 bit. Different routes use different engines; lower bit width does not guarantee more speed or unchanged quality.</Notice>}
              {stage === 3 && <>
                <Notice title={advanced ? 'After confirming the config, click below to start conversion' : building ? 'The automatic task is running' : 'View results and start using'}>Keep the local service running. The flow completes abliteration (if selected), quantization, GPU validation and measurement (if enabled) automatically; once passed, you can start the model API directly on this page.</Notice>
                {checked?.runtime_engine && <Notice tone="info" title={`Artifact runtime route · companion engine, optionally ${checked.runtime_engine.label}`}>
                  <p>'Start model API' on this page and scripts/run.bat run this artifact with the companion engine (.ninfer→NInfer, AWQ/GPTQ→HF). {checked.runtime_engine.label} must be downloaded separately: <a href={checked.runtime_engine.url} target="_blank" rel="noreferrer">{checked.runtime_engine.url}</a>. {checked.runtime_engine.steps}</p>
                </Notice>}
                {pollError && <Notice tone="warn">{pollError}</Notice>}
                <BuildPanel canBuild={canBuild} building={building} buildStatus={buildStatus} log={log} startMs={startMs}
                  performance={performance} quality={quality} onServerActive={setServerInUse} onStart={onStart} buildId={buildId} />
              </>}
              {(advanced || stage === 3) && <div className="wizard-actions">
                <Button disabled={stage === 0 || building || serverInUse} onClick={() => go(stage - 1)}>Previous</Button>
                <span className="muted">{stage < 3 ? `Next: ${titles[stage + 1]}` : serverInUse ? 'Stop the service before starting a new conversion' : 'The original model is never overwritten'}</span>
                {stage < 3 && <Button type="primary" disabled={!nextReady || building} loading={inspectLoading || planLoading} onClick={() => go(stage + 1)}>Next</Button>}
              </div>}
            </> : !metaError && <Skeleton active paragraph={{rows:5}} />}
          </Panel>
          {stage > 0 && <Collapse ghost items={[{key:'technical', label:'View the full technical plan and diagnostics', children:<PlanPreview plan={displayPlan} capabilities={capabilities} loading={planLoading} error={planError} /> }]} />}
          <StoragePanel disabled={building || serverInUse} />
        </div>
        <aside className="col wizard-aside">
          {!advanced && stage === 1 ? <Panel title="This processing route"><p>{inspection?.weight_precision ?? 'Repository model'} → {config.jailbreak === 'none' ? 'Keep original behavior' : 'Abliteration'} → {checked?.targets?.find(t => t.value === config.target)?.label ?? 'Auto-recommended quantization'} → {!specSelectionValid ? 'Speculation settings need confirmation' : hfSelected ? 'HF basic decoding' : ({auto:'Compatible decoding selected automatically',none:'Basic decoding',mtp:'MTP speculative decoding',dflash2:'DFlash2 speculative decoding'} as Record<string,string>)[selectedSpec]}</p><p>{qualityEnabled ? 'Acceptance: GPU load, measured speed, same-corpus PPL gate' : 'Acceptance: GPU load' + (config.run_benchmark ? ' and measured speed; quality not measured' : '; speed and quality not measured')}</p><p className="muted">Your specified quantization plan is never substituted. The original model is kept and required dependencies are prepared automatically.</p></Panel> : <WorkflowSummary concise={!advanced && stage < 3} plan={displayPlan} inspection={actualInspection} config={actualConfig} loading={planLoading || inspectLoading || buildStatus?.status === 'preparing'} />}
          {stage === 0 && inspection && <Collapse ghost items={[{key:'inspection', label:'View model identification details', children:<InspectionCard inspection={inspection} loading={inspectLoading} error={inspectError} meta={meta} /> }]} />}
          {advanced && planError && <Notice tone="bad" title="Config not ready">{planError}</Notice>}
        </aside>
      </main>
    </div>
    </AssistantProvider>
  );
}
