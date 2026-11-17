import type { ForgeConfig, ForgePlan, ModelInspection } from '../api/types';
import { humanBytes } from '../utils/format';
import { diagnosticText } from '../utils/guidance';
import { Fact, Notice, Panel } from './ui';

export default function WorkflowSummary({ plan, inspection, config, loading, concise = false }: {
  plan: ForgePlan | null; inspection: ModelInspection | null; config: ForgeConfig; loading: boolean; concise?: boolean;
}) {
  const recipe = plan?.steps.find((s) => s.name === 'quantize+compile');
  const problems = plan?.diagnostics.filter((d) => d.severity !== 'info') ?? [];
  if (concise) return <Panel title="What happens next">
    <p>1. Check whether the model can be processed.</p><p>2. Confirm only the applicable processing plan.</p><p>3. Prepare, convert and validate automatically; start using it when done.</p>
    <p className="muted">The original model is kept. Output location and dependencies need no advance input.</p>
  </Panel>;
  return <Panel title="This conversion · live preview">
    {!inspection && <Notice title="Start by providing a model">Select a local model folder or GGUF file, or enter a Hugging Face repository link. Required files are prepared automatically; missing sources are flagged explicitly.</Notice>}
    {inspection && <div className="facts">
      <Fact label="Model" value={inspection.architecture} wide />
      <Fact label="Input precision" value={inspection.weight_precision === 'metadata_only' ? 'Config & tokenizer' : inspection.weight_precision} />
      <Fact label="Input size" value={humanBytes(inspection.tensor_bytes)} />
      {recipe && <Fact label="Quantization plan" value={recipe.details.nominal_bits ? `${recipe.details.nominal_bits} bit · ${plan?.backend?.includes('ternary') ? 'ready-made ternary weight import' : 'converted via a compatible recipe'}` : 'Keeps the source file quantization precision'} wide />}
      <Fact label="Abliteration" value={config.jailbreak === 'none' ? 'None' : config.jailbreak} />
      <Fact label="Conversion device" value={config.device === 'cpu' ? 'CPU (GPU for inference)' : 'GPU / CUDA'} />
      {plan?.serving && <>
        <Fact label="Decode acceleration" value={plan.serving.spec ? `${plan.serving.spec.toUpperCase()} · packaged automatically` : 'Basic decoding'} />
        <Fact label="Cache precision" value={plan.serving.kv_dtype} />
        <Fact label="Context / concurrency" value={`${plan.serving.max_context} token / ${plan.serving.max_concurrency}`} wide />
      </>}
      {plan?.resources && <>
        <Fact label="Estimated artifact" value={humanBytes(plan.resources.estimated_output_bytes)} />
        <Fact label="Estimated serving VRAM" value={humanBytes(plan.resources.peak_vram_bytes)} />
        <Fact label="Estimated conversion RAM" value={humanBytes(plan.resources.peak_ram_bytes)} />
        <Fact label="Available system RAM" value={humanBytes(plan.resources.available_ram_bytes)} />
      </>}
    </div>}
    {loading && <p className="muted" role="status">Checking compatibility and resources…</p>}
    {problems.filter(d => d.severity === 'error').slice(0, 1).map(d => <Notice key={d.code} tone="warn" title="Complete this step first">{diagnosticText(d)}</Notice>)}
    {plan && !loading && plan.executable && <Notice tone="ok" title="Config check passed">This config has an executable path; whether it is directly usable follows the GPU validation in the build results, and speed follows actual tests.</Notice>}
    <p className="muted">Weights, cache and speculative components share VRAM. Estimates are not actual peaks, and lower bit width does not promise more speed; cache-compression and fast-prefill quality must be validated per task.</p>
    <div className="mono path-wrap">Output: {config.output_dir || 'generated after selecting a model'}</div>
  </Panel>;
}
