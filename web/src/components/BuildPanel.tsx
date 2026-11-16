import { useEffect, useState } from 'react';
import { Button, Steps, Table, Timeline, Typography, Space, App, Collapse } from 'antd';
import { RocketOutlined } from '@ant-design/icons';
import type { BuildStatus, MetricEntry, Report, ModelServer } from '../api/types';
import { api } from '../api/client';
import { elapsedLabel, formatTime, humanBytes, metricValueText } from '../utils/format';
import { Chip, Fact, Notice, Panel } from './ui';
import ReportViewer from './ReportViewer';

interface Props {
  buildId: string | null;
  canBuild: boolean;
  building: boolean;
  buildStatus: BuildStatus | null;
  log: string;
  startMs: number | null;
  performance: Report | null;
  quality: Report | null;
  onStart: () => void;
  onServerActive?: (active:boolean) => void;
}

const STAGES = ['计划', '越狱', '转换', '验证', '基准', '完成'];
const STAGE_INDEX: Record<string, number> = {
  created: 0,
  planned: 0,
  jailbreaking: 1,
  jailbroken: 1,
  converting: 2,
  converted: 2,
  validating: 3,
  validated: 3,
  'validation-failed': 3,
  benchmarking: 4,
};
const ERROR_STATUSES = new Set(['blocked', 'failed', 'crashed', 'validation-failed', 'quality-rejected', 'interrupted']);
const OK_FINAL = new Set(['validated', 'converted']);
const STATUS_LABEL: Record<string, string> = {idle:'等待开始', starting:'正在开始', planned:'已规划',
  interrupted:'上次任务被中断，请重新开始', preparing:'自动准备依赖与模型文件', 'quality-rejected':'质量门禁未通过，禁止直接启动',
  jailbreaking:'正在标定与越狱', jailbroken:'权重编辑完成', converting:'正在转换', converted:'转换完成（未验证）',
  validating:'正在验证 GPU 加载与生成', validated:'验证通过', benchmarking:'正在测试速度',
  blocked:'配置被阻止', failed:'构建失败', crashed:'构建异常', 'validation-failed':'GPU 验证失败'};

function stageState(status: string, finished: boolean, history: { status: string }[]) {
  let reached = 0;
  for (const h of history) reached = Math.max(reached, STAGE_INDEX[h.status] ?? 0);
  if (finished && OK_FINAL.has(status)) reached = 5;
  const isError = ERROR_STATUSES.has(status);
  if (isError && STAGE_INDEX[status] !== undefined) reached = STAGE_INDEX[status];
  return STAGES.map((_, i) => {
    if (isError && i === reached) return 'error' as const;
    if (i < reached) return 'finish' as const;
    if (i === reached) return finished ? ('finish' as const) : ('process' as const);
    return 'wait' as const;
  });
}

const HEADLINE: Record<string, string> = {
  measured: '实测',
  configured: '配置',
  refused: '拒绝',
  not_measured: '未实测',
};

function metricNode(entry?: MetricEntry | null) {
  const status = entry?.status ?? 'not_measured';
  const measured = status === 'measured' && entry?.value !== null && entry?.value !== undefined;
  if (measured) {
    return (
      <span className="mono" style={{ color: 'var(--ok)' }}>
        {metricValueText(entry!.value)}
      </span>
    );
  }
  return <Chip>{HEADLINE[status] ?? status}</Chip>;
}

export default function BuildPanel({
  buildId,
  canBuild,
  building,
  buildStatus,
  log,
  startMs,
  performance,
  quality,
  onStart,
  onServerActive,
}: Props) {
  const { message } = App.useApp();
  const [server, setServer] = useState<ModelServer | null>(null);
  const [serverBusy, setServerBusy] = useState(false);
  const serverActive = server?.status === 'starting' || server?.status === 'ready';
  useEffect(() => { onServerActive?.(serverActive); }, [serverActive,onServerActive]);
  useEffect(() => {
    let cancelled = false;
    setServer(null);
    if (buildId) api.server(buildId).then((state) => { if (!cancelled) setServer(state); }).catch(() => {});
    return () => { cancelled = true; };
  }, [buildId]);
  useEffect(() => {
    if (!buildId || !serverActive) return;
    let cancelled = false;
    const timer = setInterval(() => { api.server(buildId).then((state) => { if (!cancelled) setServer(state); })
      .catch((error) => { if (!cancelled) message.error((error as Error).message); }); }, 2500);
    return () => { cancelled = true; clearInterval(timer); };
  }, [buildId, serverActive, message]);
  const controlServer = async (method: 'POST' | 'DELETE') => {
    if (!buildId) return;
    setServerBusy(true);
    try { setServer(await api.server(buildId, method)); }
    catch (error) { message.error((error as Error).message); }
    finally { setServerBusy(false); }
  };
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    if (!building) return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [building]);

  const manifest = buildStatus?.manifest ?? null;
  const status = buildStatus?.status ?? 'idle';
  const history = manifest?.history ?? [];
  const stages = stageState(status, buildStatus?.finished ?? false, history);
  const executionConfig = manifest?.config ?? buildStatus?.prepared_config;
  const visibleStages = STAGES.map((title, index) => ({ title, index })).filter(({index}) =>
    (index !== 1 || (executionConfig && executionConfig.jailbreak !== 'none')) &&
    (index !== 3 || executionConfig?.run_validation !== false) &&
    (index !== 4 || executionConfig?.run_benchmark));
  const artifacts = Object.entries(manifest?.artifacts ?? {}).map(([name, a]) => ({ name, ...a }));
  const measurements = manifest?.measurements;
  const retainedHeads = [manifest?.speculative_retention?.mtp?.retained ? 'MTP' : '',manifest?.speculative_retention?.dflash2?.retained ? 'DFlash2' : ''].filter(Boolean);
  const isError = ERROR_STATUSES.has(status);
  const dotClass = building ? 'running' : isError ? 'bad' : OK_FINAL.has(status) ? 'ok' : '';

  return (
    <Panel
      title="一键转换"
      icon={<RocketOutlined />}
      extra={
        <Button
          type="primary"
          icon={<RocketOutlined />}
          disabled={!canBuild || building}
          loading={building}
          onClick={onStart}
        >
          {building ? '转换中…' : '开始转换'}
        </Button>
      }
    >
      <div className="status-line" data-ai-id="build.status" data-ai-title="构建状态" style={{ marginBottom: 14 }}>
        <span className={'status-dot ' + dotClass} />
        <span>{STATUS_LABEL[status] ?? status}</span>
        {building && <span className="muted">· {elapsedLabel(startMs, now)}</span>}
      </div>

      {!buildStatus && (
        <p className="muted" style={{ margin: 0, fontSize: 12.5, lineHeight: 1.6 }}>
          计划可执行后，点击「开始转换」。整个过程（越狱 → 量化转换 → 验证 → 实测）会实时显示在下方；
          原始模型不会被修改，产物写入输出目录。
        </p>
      )}
      {buildStatus?.status === 'preparing' && <Notice title="脚本正在自动准备，请保持本机服务运行">
        <p>首次可能下载数 GiB 的引擎或模型文件。已校验的缓存会复用；网络中断后可再次开始，下载将续传。</p>
        <div className="console" style={{maxHeight:220}}>{buildStatus.setup_log?.slice(-8).join('\n') || '正在检查依赖…'}</div>
      </Notice>}

      {buildStatus && (
        <div className="fade">
          <Steps
            size="small"
            style={{ marginBottom: 16 }}
            items={visibleStages.map(({title, index}) => ({ title, status: stages[index] }))}
          />

          {quality?.gate && <div data-ai-id="quality.gate" data-ai-title="质量门禁"><Notice tone={quality.gate.status === 'passed' ? 'ok' : 'bad'} title={quality.gate.status === 'passed' ? '质量门禁通过（限所测语料）' : '质量门禁未通过'}>{quality.gate.relative_increase !== undefined ? `PPL 相对增幅 ${(quality.gate.relative_increase*100).toFixed(2)}%` : quality.gate.note}</Notice></div>}
          {buildStatus.finished && OK_FINAL.has(status) && <Notice title="本次验收范围">可运行：{status === 'validated' ? 'GPU 验证通过' : '未验证'}；质量：{quality?.gate?.status === 'passed' ? '所测语料门禁通过' : quality?.gate ? '门禁未通过' : '未通过门禁验证'}；加速结论：{performance?.metrics?.decode_tok_s?.status === 'measured' ? '已测速；是否更快仍需同条件对照' : '未实测'}。</Notice>}
          {!!retainedHeads.length && <Notice tone="ok" title="推测头保留校验通过">已保留 {retainedHeads.join('、')} 的实际权重绑定并接通输出投影。{manifest?.plan?.serving?.spec ? `当前启用 ${manifest.plan.serving.spec.toUpperCase()}。` : '当前未启用推测解码。'}组件接通不等于实测提速，速度以对应配置的已有/实际测量为准。</Notice>}
          {manifest?.speculative_retention?.status === 'retained' && <Notice title="推测头已无损归档">自带头与组件文件已按原始字节保留；当前 HF 服务不执行 MTP/DFlash2，不能把文件保留标成已经加速。</Notice>}
          {buildStatus.error && <Notice tone="bad" title="构建异常">{buildStatus.error}</Notice>}
          {isError && !buildStatus.error && (
            <Notice tone="bad" title={`构建${status === 'blocked' ? '被阻止' : '失败'}：${status}`}>
              {manifest?.plan_diagnostics?.length
                ? manifest.plan_diagnostics.map((d) => d.message).join('；')
                : [...history].reverse().find((h) => ERROR_STATUSES.has(h.status) && h.detail !== 'performance/quality reports written')?.detail ?? '见下方时间线与日志'}
            </Notice>
          )}
          {buildStatus.finished && OK_FINAL.has(status) && (
            <Notice tone="ok" title="构建完成">
              产物已保存到 {buildStatus.output_dir}。如需重新构建，请回到第一步；自动模式会分配新目录，高级模式需选择空目录。
            </Notice>
          )}
          {buildStatus.finished && status === 'validated' && <Notice tone="info" title="下一步：启动并使用模型">
            <p>点击启动后等待模型加载。状态显示就绪，即可把下面的地址填入支持 OpenAI API 的应用。服务运行期间会占用显存。</p>
            <Space data-ai-id="build.server" data-ai-title="启动与停止模型服务">
              <Button type="primary" loading={serverBusy} disabled={serverActive} onClick={() => controlServer('POST')}>启动模型服务</Button>
              <Button disabled={!serverActive || serverBusy} onClick={() => controlServer('DELETE')}>停止并释放显存</Button>
              <span>{server?.status === 'ready' ? '服务已就绪' : server?.status === 'starting' ? '正在加载模型…' : server?.status === 'exited' ? '服务已退出，请查看 serve.log' : '尚未启动'}</span>
            </Space>
            {server && <Typography.Paragraph copyable={{text:server.base_url}} className="mono" style={{marginTop:12}}>{server.base_url}</Typography.Paragraph>}
            <p>也可双击输出目录中的 scripts/run.bat 启动；详细日志位于 logs/serve.log。</p>
          </Notice>}
          {buildStatus.finished && status === 'converted' && <Notice tone="warn" title="转换完成，尚未验证可运行">此次关闭了 GPU 验证，页面不会标为可直接使用。可用 scripts/run.bat 手动验证，或回到第一步选择新输出目录并开启验证。</Notice>}

          {history.length > 0 && (
            <Collapse ghost size="small" items={[{key:'history', label:'查看执行记录与诊断', children:<Timeline
              style={{ marginBottom: 16 }}
              items={history.map((h) => ({
                color: ERROR_STATUSES.has(h.status) ? 'red' : OK_FINAL.has(h.status) ? 'green' : 'blue',
                children: (
                  <div>
                    <Chip mono>{h.status}</Chip> <span>{h.detail}</span>
                    <span className="muted" style={{ marginLeft: 8, fontSize: 12 }}>
                      {formatTime(h.at)}
                    </span>
                  </div>
                ),
              }))}
            />}]} />
          )}

          <Collapse ghost size="small" items={[{key:'logs', label:'查看转换日志（实时）', children:<div className="console">{log || '（等待转换器输出…）'}</div>}]} />

          {artifacts.length > 0 && (
            <Collapse ghost size="small" items={[{key:'artifacts', label:'查看产物与文件位置', children:<Table
                size="small"
                pagination={false}
                rowKey="name"
                dataSource={artifacts}
                columns={[
                  { title: '名称', dataIndex: 'name', width: 150 },
                  { title: '路径', dataIndex: 'path', render: (v: string) => <span className="mono">{v}</span> },
                  { title: '大小', dataIndex: 'bytes', width: 100, render: (v: number) => humanBytes(v) },
                  {
                    title: 'SHA256',
                    dataIndex: 'sha256',
                    width: 120,
                    render: (v?: string) => (v ? <span className="mono muted">{v.slice(0, 12)}…</span> : '—'),
                  },
                ]}
              />}]} />
          )}

          {measurements && (
            <>
              <div className="subhead">测量摘要（诚实标注）</div>
              <div className="facts">
                <Fact label="生成 tok/s" value={metricNode(measurements.performance?.decode_tok_s)} />
                <Fact label="预填充 tok/s" value={metricNode(measurements.performance?.prefill_tok_s)} />
                <Fact label="困惑度" value={metricNode(measurements.quality?.perplexity)} />
                <Fact label="量化损失" value={metricNode(measurements.quality?.quantization_loss)} />
              </div>
            </>
          )}

          <Collapse ghost size="small" items={[{key:'reports', label:'查看完整性能与质量报告', children:<ReportViewer performance={performance} quality={quality} />}]} />
        </div>
      )}
    </Panel>
  );
}
