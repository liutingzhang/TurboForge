// Mirrors the backend JSON contracts (turboforge.domain / api.meta) exactly.
// Keeping these in one place is what prevents frontend/backend drift.

export type Severity = 'info' | 'warning' | 'error';

export interface Diagnostic {
  code: string;
  message: string;
  severity: Severity;
  alternatives: string[];
}

export interface Choice {
  value: string;
  label: string;
  description: string;
}

export type ParamType = 'path' | 'int' | 'float' | 'string' | 'string_list';

export interface ParamSchema {
  key: string;
  label: string;
  type: ParamType;
  required: boolean;
  required_when_env_missing: string | null;
  help: string;
  placeholder: string;
  default: unknown;
  advanced: boolean;
}

export interface ProcessorCapability {
  name: string;
  supported_architectures: string[];
  input_precisions: string[];
  supports_nvfp4: boolean;
  requires_high_precision: boolean;
  requires_calibration: boolean;
  supports_chunking: boolean;
  min_ram_bytes: number | null;
  min_vram_bytes: number | null;
  output_precision: string;
  can_quantize_afterwards: boolean;
  implementation: string;
  available: boolean;
  unavailable_reason: string | null;
}

export interface JailbreakMeta {
  name: string;
  label: string;
  description: string;
  capability: ProcessorCapability;
  params: ParamSchema[];
}

export interface Recipe {
  engine?: 'ninfer' | 'gptqmodel';
  recipe: string;
  label: string;
  input_kinds: string[];
  architectures: string[];
  storage_formats: string[];
  nominal_bits: number | null;
  typical_bpw: number | null;
  extra_sources: string[];
  components: string | null;
  needs_cuda: boolean;
  needs_gpu: boolean;
  notes: string[];
}

export interface UpstreamMeta {
  root: string | null;
  found: boolean;
  revision: string | null;
  revision_matches_pin: boolean;
  torch_available: boolean;
  numpy_available: boolean;
  diagnostics: Diagnostic[];
}

export interface GpuMeta {
  name: string;
  total_bytes: number;
  free_bytes: number;
}

export interface EnvMeta {
  torch_cuda: boolean;
  torch: boolean;
  transformers: boolean;
  numpy: boolean;
  safetensors: boolean;
  flask: boolean;
}

export interface Defaults {
  upstream_dir: string;
  ninfer_dir: string;
  target: string;
  jailbreak: string;
  device: string;
  allow_approximate_requantization: boolean;
  run_validation: boolean;
  run_benchmark: boolean;
  runtime: Record<string, unknown>;
  benchmark: Record<string, unknown>;
}

export interface Meta {
  hf_recipes?: Recipe[];
  engine: { root: string; available: boolean };
  pinned_revision: string;
  choices: {
    jailbreak: JailbreakMeta[];
    device: Choice[];
    input_kind: { value: string; label: string }[];
  };
  recipes: Recipe[];
  env: EnvMeta;
  upstream: UpstreamMeta;
  gpu: GpuMeta | null;
  defaults: Defaults;
}

export interface TensorSummary {
  name: string;
  dtype: string;
  shape: number[];
  byte_size: number;
}

export interface ModelInspection {
  path: string;
  input_kind: string;
  architecture: string;
  model_type: string;
  weight_precision: string;
  tensor_count: number;
  tensor_bytes: number;
  element_count: number;
  shard_count: number;
  dtype_counts: Record<string, number>;
  tensor_samples: TensorSummary[];
  quantization: Record<string, unknown>;
  tokenizer_files: string[];
  speculative_components: Record<string, any>;
  license: string | null;
  files: Record<string, unknown>[];
  diagnostics: Diagnostic[];
  config: Record<string, unknown>;
  usable: boolean;
  source_path: string;
}

export interface ResourceEstimate {
  input_bytes: number;
  estimated_output_bytes: number | null;
  temporary_disk_bytes: number;
  peak_ram_bytes: number | null;
  peak_vram_bytes: number | null;
  available_disk_bytes: number;
  available_ram_bytes: number | null;
  available_vram_bytes: number | null;
  assumptions: string[];
  diagnostics: Diagnostic[];
}

export interface PlanStep {
  name: string;
  implementation: string;
  status: string;
  details: Record<string, any>;
}

// Serving profile derived by the backend (turboforge.serving). These are serving
// choices, not conversion choices: the artifact bytes are identical regardless.
// The UI shows them read-only so the preview matches the generated launcher.
export interface ServingProfile {
  kv_dtype: string;
  spec: string | null;
  draft_tokens: number | null;
  max_context: number;
  context_ceiling: number | null;
  max_concurrency: number;
  port: number;
  embedding_q4: boolean;
  launcher_flags: string[];
}

export interface BackendCapability {
  name: string;
  target: string;
  input_kinds: string[];
  architectures: string[];
  nominal_bits: number | null;
  typical_bpw: number | null;
  implementation: string;
  available: boolean;
  unavailable_reason: string | null;
  verified_conversion: boolean;
  notes: string[];
}

export interface ForgeConfig {
  quantization?: Record<string, unknown>;
  upstream_dir?: string;
  ninfer_dir?: string;
  input_path: string;
  output_dir: string;
  target: string;
  jailbreak: string;
  jailbreak_params: Record<string, any>;
  runtime: Record<string, any>;
  allow_approximate_requantization: boolean;
  source_revision?: string | null;
  sources: Record<string, string>;
  device: string;
  run_validation: boolean;
  run_benchmark: boolean;
  benchmark: Record<string, any>;
}

export interface ForgePlan {
  config: ForgeConfig;
  inspection: ModelInspection;
  steps: PlanStep[];
  diagnostics: Diagnostic[];
  resources: ResourceEstimate | null;
  backend: string | null;
  processor: string | null;
  serving: ServingProfile | null;
  executable: boolean;
}

export interface InspectRecipe {
  recipe: string;
  label: string;
  nominal_bits: number | null;
  typical_bpw: number | null;
  extra_sources: string[];
  needs_gpu: boolean;
}

export interface InspectResponse {
  inspection: ModelInspection;
  recipes: InspectRecipe[];
}

export interface PlanResponse {
  plan: ForgePlan;
  capabilities: BackendCapability[];
  recipes: Recipe[];
}

export interface BuildArtifact {
  path: string;
  bytes: number;
  sha256?: string;
}

export interface HistoryEntry {
  at: string;
  status: string;
  detail: string;
}

export interface Manifest {
  speculative_retention?: {
    mtp?: {retained:boolean;binding_count:number};
    dflash2?: {retained:boolean;binding_count:number};
    status?: string; runtime_acceleration?: string;
  };
  schema_version: number;
  build_id: string;
  status: string;
  history: HistoryEntry[];
  created_at?: string;
  updated_at?: string;
  config?: ForgeConfig;
  source?: ModelInspection;
  plan?: ForgePlan;
  quality_loss?: Record<string, unknown>;
  artifacts?: Record<string, BuildArtifact>;
  measurements?: any;
  converter?: any;
  plan_diagnostics?: Diagnostic[];
}

export interface BuildStatus {
  setup_log?: string[];
  prepared_config?: ForgeConfig;
  build_id: string;
  finished: boolean;
  status: string;
  output_dir: string;
  error: string | null;
  manifest: Manifest | null;
}

export interface ModelServer {
  status: 'stopped' | 'starting' | 'ready' | 'exited';
  base_url: string;
  log_path: string;
}

export type MetricStatus = 'measured' | 'configured' | 'refused' | 'not_measured';

export interface MetricEntry {
  status: MetricStatus;
  value: any;
  source?: string;
  command?: string[];
  note?: string;
}

export interface Report {
  gate?: {status:string; relative_increase?:number; maximum_relative_increase?:number; note?:string};
  recorded_at: string;
  policy: string;
  environment?: Record<string, unknown>;
  separation?: string;
  jailbreak_processor?: string;
  context_tokens?: MetricEntry;
  metrics: Record<string, MetricEntry>;
  runs?: Record<string, any>;
}


export interface PrecheckTarget {
  engine: 'ninfer' | 'gptqmodel';
  value: string; label: string; bits: number | null; bpw: number | null;
  formats: string[]; available: boolean; required_sources: string[]; missing_sources: string[];
  gpu_note?: string | null;
}
export interface GpuInfo {
  name: string; generation: string; compute_cap: number; total_bytes: number; driver: string;
}
// Advisory MoE engine selection by GPU (precheck._moe_engine_recommendation).
// Display only: TurboForge still emits .ninfer; Strata/Basalt are separate
// community engines with their own formats, never integrated here.
export interface MoeEngineRecommendation {
  engine: 'strata' | 'basalt' | 'fusion-kvmem';
  label: string; reason: string; compat_check?: string; requirements: string; format: string;
  models: string; alternative: string; url: string; steps: string; advisory: string;
}
export interface PrecheckResult {
  recommended_max_context?: number;
  detected_components?: string[];
  architecture_kind?: string;
  gpu?: GpuInfo | null;
  scheme_order?: 'native-format' | 'bits-per-weight';
  runtime_engine?: MoeEngineRecommendation | null;
  moe_engine?: MoeEngineRecommendation | null;
  ready: boolean; message: string; can_edit_weights?: boolean; remote?: boolean; action?: string;
  targets?: PrecheckTarget[]; speculation?: string[]; requires_explicit_target?: boolean;
  provided_sources?: string[]; source_fields?: Record<string, {label:string;kind:'file'|'directory'}>;
  optional_sources?: Record<string, {label:string;kind:'file'|'directory'|'model'}>;
  input_candidates?: Array<{name:string;reference:string}>;
  unsupported?: Array<{name:string;reason:string}>; quality_note?: string;
}
