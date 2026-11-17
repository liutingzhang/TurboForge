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

const STAGES = ['Plan', 'Abliteration', 'Convert', 'Validate', 'Benchmark', 'Done'];
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
const STATUS_LABEL: Record<string, string> = {idle:'Waiting to start', starting:'Starting', planned:'Planned',
  interrupted:'The previous task was interrupted; start again', preparing:'Automatically preparing dependencies and model files', 'quality-rejected':'Quality gate failed; direct start is forbidden',
  jailbreaking:'Calibrating and abliterating', jailbroken:'Weight editing complete', converting:'Converting', converted:'Conversion complete (not validated)',
  validating:'Validating GPU load and generation', validated:'Validation passed', benchmarking:'Benchmarking speed',
  blocked:'Config blocked', failed:'Build failed', crashed:'Build error', 'validation-failed':'GPU validation failed'};

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
  measured: 'Measured',
  configured: 'Configured',
  refused: 'Refused',
  not_measured: 'Not measured',
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
      title="One-click conversion"
      icon={<RocketOutlined />}
      extra={
        <Button
          type="primary"
          icon={<RocketOutlined />}
          disabled={!canBuild || building}
          loading={building}
          onClick={onStart}
        >
          {building ? 'Converting…' : 'Start conversion'}
        </Button>
      }
    >
      <div className="status-line" data-ai-id="build.status" data-ai-title="Build status" style={{ marginBottom: 14 }}>
        <span className={'status-dot ' + dotClass} />
        <span>{STATUS_LABEL[status] ?? status}</span>
        {building && <span className="muted">· {elapsedLabel(startMs, now)}</span>}
      </div>

      {!buildStatus && (
        <p className="muted" style={{ margin: 0, fontSize: 12.5, lineHeight: 1.6 }}>
          Once the plan is executable, click 'Start conversion'. The whole process (abliteration → quantization/conversion → validation → measurement) is shown live below; 
          the original model is never modified; artifacts are written to the output directory.
        </p>
      )}
      {buildStatus?.status === 'preparing' && <Notice title="The script is preparing automatically; keep the local service running">
        <p>The first run may download several GiB of engine or model files. Verified caches are reused; after a network interruption, start again and downloads resume.</p>
        <div className="console" style={{maxHeight:220}}>{buildStatus.setup_log?.slice(-8).join('\n') || 'Checking dependencies…'}</div>
      </Notice>}

      {buildStatus && (
        <div className="fade">
          <Steps
            size="small"
            style={{ marginBottom: 16 }}
            items={visibleStages.map(({title, index}) => ({ title, status: stages[index] }))}
          />

          {quality?.gate && <div data-ai-id="quality.gate" data-ai-title="Quality gate"><Notice tone={quality.gate.status === 'passed' ? 'ok' : 'bad'} title={quality.gate.status === 'passed' ? 'Quality gate passed (tested corpus only)' : 'Quality gate failed'}>{quality.gate.relative_increase !== undefined ? `Relative PPL increase ${(quality.gate.relative_increase*100).toFixed(2)}%` : quality.gate.note}</Notice></div>}
          {buildStatus.finished && OK_FINAL.has(status) && <Notice title="Acceptance scope of this run">Runnable: {status === 'validated' ? 'GPU validation passed' : 'not validated'}; quality: {quality?.gate?.status === 'passed' ? 'gate passed on the tested corpus' : quality?.gate ? 'gate failed' : 'not gate-validated'}; acceleration: {performance?.metrics?.decode_tok_s?.status === 'measured' ? 'speed measured; whether it is faster still needs a same-condition baseline' : 'not measured'}.</Notice>}
          {!!retainedHeads.length && <Notice tone="ok" title="Speculative-head retention check passed">Retained the actual weight bindings of {retainedHeads.join(', ')} and connected them to the output projection. {manifest?.plan?.serving?.spec ? `Currently enabled: ${manifest.plan.serving.spec.toUpperCase()}.` : 'Speculative decoding is currently disabled.'}Connected components do not equal measured speedup; speed follows existing/actual measurements for the corresponding config.</Notice>}
          {manifest?.speculative_retention?.status === 'retained' && <Notice title="Speculative heads archived losslessly">Bundled heads and component files are kept byte-for-byte; the current HF service does not run MTP/DFlash2, so file retention must not be labeled as acceleration.</Notice>}
          {buildStatus.error && <Notice tone="bad" title="Build error">{buildStatus.error}</Notice>}
          {isError && !buildStatus.error && (
            <Notice tone="bad" title={`Build ${status === 'blocked' ? 'blocked' : 'failed'}: ${status}`}>
              {manifest?.plan_diagnostics?.length
                ? manifest.plan_diagnostics.map((d) => d.message).join('; ')
                : [...history].reverse().find((h) => ERROR_STATUSES.has(h.status) && h.detail !== 'performance/quality reports written')?.detail ?? 'See the timeline and logs below'}
            </Notice>
          )}
          {buildStatus.finished && OK_FINAL.has(status) && (
            <Notice tone="ok" title="Build complete">
              Artifacts saved to {buildStatus.output_dir}. To rebuild, go back to step 1; auto mode assigns a new directory, advanced mode needs an empty directory.
            </Notice>
          )}
          {buildStatus.finished && status === 'validated' && <Notice tone="info" title="Next: start and use the model">
            <p>After clicking start, wait for the model to load. Once the status shows ready, put the address below into any OpenAI-API-compatible app. The service occupies VRAM while running.</p>
            <Space data-ai-id="build.server" data-ai-title="Start and stop the model service">
              <Button type="primary" loading={serverBusy} disabled={serverActive} onClick={() => controlServer('POST')}>Start model API</Button>
              <Button disabled={!serverActive || serverBusy} onClick={() => controlServer('DELETE')}>Stop and free VRAM</Button>
              <span>{server?.status === 'ready' ? 'Service ready' : server?.status === 'starting' ? 'Loading model…' : server?.status === 'exited' ? 'Service exited; check serve.log' : 'Not started'}</span>
            </Space>
            {server && <Typography.Paragraph copyable={{text:server.base_url}} className="mono" style={{marginTop:12}}>{server.base_url}</Typography.Paragraph>}
            <p>You can also double-click scripts/run.bat in the output directory to start; detailed logs are in logs/serve.log.</p>
          </Notice>}
          {buildStatus.finished && status === 'converted' && <Notice tone="warn" title="Conversion complete; runnability not yet validated">GPU validation was turned off for this run, so the page will not mark it directly usable. Validate manually with scripts/run.bat, or go back to step 1, choose a new output directory and enable validation.</Notice>}

          {history.length > 0 && (
            <Collapse ghost size="small" items={[{key:'history', label:'View execution history and diagnostics', children:<Timeline
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

          <Collapse ghost size="small" items={[{key:'logs', label:'View conversion log (live)', children:<div className="console">{log || '(waiting for converter output…)'}</div>}]} />

          {artifacts.length > 0 && (
            <Collapse ghost size="small" items={[{key:'artifacts', label:'View artifacts and file locations', children:<Table
                size="small"
                pagination={false}
                rowKey="name"
                dataSource={artifacts}
                columns={[
                  { title: 'Name', dataIndex: 'name', width: 150 },
                  { title: 'Path', dataIndex: 'path', render: (v: string) => <span className="mono">{v}</span> },
                  { title: 'Size', dataIndex: 'bytes', width: 100, render: (v: number) => humanBytes(v) },
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
              <div className="subhead">Measurement summary (honestly labeled)</div>
              <div className="facts">
                <Fact label="Generation tok/s" value={metricNode(measurements.performance?.decode_tok_s)} />
                <Fact label="Prefill tok/s" value={metricNode(measurements.performance?.prefill_tok_s)} />
                <Fact label="Perplexity" value={metricNode(measurements.quality?.perplexity)} />
                <Fact label="Quantization loss" value={metricNode(measurements.quality?.quantization_loss)} />
              </div>
            </>
          )}

          <Collapse ghost size="small" items={[{key:'reports', label:'View full performance and quality reports', children:<ReportViewer performance={performance} quality={quality} />}]} />
        </div>
      )}
    </Panel>
  );
}
