import type { ForgeConfig, ForgePlan, ModelInspection } from '../api/types';
import { humanBytes } from '../utils/format';
import { diagnosticText } from '../utils/guidance';
import { Fact, Notice, Panel } from './ui';

export default function WorkflowSummary({ plan, inspection, config, loading, concise = false }: {
  plan: ForgePlan | null; inspection: ModelInspection | null; config: ForgeConfig; loading: boolean; concise?: boolean;
}) {
  const recipe = plan?.steps.find((s) => s.name === 'quantize+compile');
  const problems = plan?.diagnostics.filter((d) => d.severity !== 'info') ?? [];
  if (concise) return <Panel title="接下来会发生什么">
    <p>1. 检查模型能否处理。</p><p>2. 只确认适用的处理方式。</p><p>3. 自动准备、转换并验证，完成后启动使用。</p>
    <p className="muted">原始模型保留。输出位置与依赖无需提前填写。</p>
  </Panel>;
  return <Panel title="本次转换 · 实时预览">
    {!inspection && <Notice title="从提供模型开始">选择本机模型文件夹、GGUF 文件，或填写 Hugging Face 仓库链接。所需文件会自动准备，缺少来源时会明确提示。</Notice>}
    {inspection && <div className="facts">
      <Fact label="模型" value={inspection.architecture} wide />
      <Fact label="输入精度" value={inspection.weight_precision === 'metadata_only' ? '配置与分词器' : inspection.weight_precision} />
      <Fact label="输入体积" value={humanBytes(inspection.tensor_bytes)} />
      {recipe && <Fact label="量化方案" value={recipe.details.nominal_bits ? `${recipe.details.nominal_bits} bit · ${plan?.backend?.includes('ternary') ? '现成三值权重导入' : '按兼容配方转换'}` : '保持来源文件的量化精度'} wide />}
      <Fact label="越狱" value={config.jailbreak === 'none' ? '不处理' : config.jailbreak} />
      <Fact label="转换设备" value={config.device === 'cpu' ? 'CPU（推理用 GPU）' : 'GPU / CUDA'} />
      {plan?.serving && <>
        <Fact label="解码加速" value={plan.serving.spec ? `${plan.serving.spec.toUpperCase()} · 自动封装` : '基础解码'} />
        <Fact label="缓存精度" value={plan.serving.kv_dtype} />
        <Fact label="上下文 / 并发" value={`${plan.serving.max_context} token / ${plan.serving.max_concurrency}`} wide />
      </>}
      {plan?.resources && <>
        <Fact label="预计产物" value={humanBytes(plan.resources.estimated_output_bytes)} />
        <Fact label="预计服务显存" value={humanBytes(plan.resources.peak_vram_bytes)} />
        <Fact label="预计转换内存" value={humanBytes(plan.resources.peak_ram_bytes)} />
        <Fact label="可用系统内存" value={humanBytes(plan.resources.available_ram_bytes)} />
      </>}
    </div>}
    {loading && <p className="muted" role="status">正在检查兼容性与资源…</p>}
    {problems.filter(d => d.severity === 'error').slice(0, 1).map(d => <Notice key={d.code} tone="warn" title="请先完成这一步">{diagnosticText(d)}</Notice>)}
    {plan && !loading && plan.executable && <Notice tone="ok" title="配置检查通过">此配置具备可执行路径；是否可直接使用以构建结果中的 GPU 验证为准，速度以实际测试为准。</Notice>}
    <p className="muted">权重、缓存与推测组件共同占用显存。预计值不代表实际峰值，也不承诺越低位宽越快；缓存压缩与快速预填充的质量需按任务验证。</p>
    <div className="mono path-wrap">输出：{config.output_dir || '选择模型后自动生成'}</div>
  </Panel>;
}
