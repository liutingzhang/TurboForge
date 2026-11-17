import { Collapse, Empty, Table, Tabs } from 'antd';
import type { MetricEntry, MetricStatus, Report } from '../api/types';
import { metricValueText } from '../utils/format';
import { Chip, Fact, Notice } from './ui';

const STATUS_META: Record<MetricStatus, { label: string; tone: 'ok' | 'info' | 'warn' | 'default' }> = {
  measured: { label: 'Measured', tone: 'ok' },
  configured: { label: 'Configured', tone: 'info' },
  refused: { label: 'Refused', tone: 'warn' },
  not_measured: { label: 'Not measured', tone: 'default' },
};

function StatusChip({ status }: { status: MetricStatus }) {
  const m = STATUS_META[status] ?? STATUS_META.not_measured;
  return <Chip tone={m.tone}>{m.label}</Chip>;
}

function MetricTable({ report }: { report: Report }) {
  const rows = Object.entries(report.metrics).map(([name, entry]) => ({ name, ...entry }));
  return (
    <Table
      size="small"
      pagination={false}
      rowKey="name"
      dataSource={rows}
      columns={[
        { title: 'Metric', dataIndex: 'name', width: 200, render: (v: string) => <span className="mono">{v}</span> },
        {
          title: 'Status',
          dataIndex: 'status',
          width: 90,
          render: (v: MetricStatus) => <StatusChip status={v} />,
        },
        {
          title: 'Value',
          dataIndex: 'value',
          render: (v: unknown, row: MetricEntry) =>
            row.status === 'measured' || row.status === 'configured' ? (
              <span className="mono">{metricValueText(v)}</span>
            ) : (
              <span className="muted">—</span>
            ),
        },
        {
          title: 'Notes / source',
          dataIndex: 'note',
          render: (v: string | undefined, row: MetricEntry) => (
            <span className="muted" style={{ fontSize: 12 }}>
              {v ?? ''}
              {row.source && <div className="mono dim">{row.source}</div>}
            </span>
          ),
        },
      ]}
    />
  );
}

function ReportBody({ report }: { report: Report | null }) {
  if (!report) {
    return (
      <Empty
        image={Empty.PRESENTED_IMAGE_SIMPLE}
        description="The report has not been generated yet (written after the build completes; metrics that did not run stay 'Not measured')"
      />
    );
  }
  const runs = Object.entries(report.runs ?? {});
  const env = Object.entries(report.environment ?? {});
  return (
    <div className="fade">
      <Notice tone="info">{report.policy}</Notice>
      {report.separation && <Notice tone="warn">{report.separation}</Notice>}
      {env.length > 0 && (
        <div className="facts">
          {env.map(([k, v]) => (
            <Fact key={k} label={k} value={String(v)} mono />
          ))}
        </div>
      )}
      <MetricTable report={report} />
      {runs.length > 0 && (
        <Collapse
          ghost
          size="small"
          style={{ marginTop: 8 }}
          items={[
            {
              key: 'runs',
              label: <span className="muted">Tool run records (reproducible, {runs.length})</span>,
              children: runs.map(([name, run]) => (
                <div key={name} style={{ marginBottom: 12 }}>
                  <div className="mono" style={{ color: 'var(--info)' }}>
                    {name} · exit {run.returncode} · {run.seconds}s
                  </div>
                  <div className="console" style={{ maxHeight: 160, marginTop: 6 }}>
                    {(run.command ?? []).join(' ')}
                    {run.stderr_tail ? `\n--- stderr ---\n${run.stderr_tail}` : ''}
                  </div>
                </div>
              )),
            },
          ]}
        />
      )}
    </div>
  );
}

interface Props {
  performance: Report | null;
  quality: Report | null;
}

export default function ReportViewer({ performance, quality }: Props) {
  return (
    <Tabs
      size="small"
      items={[
        { key: 'perf', label: 'Performance report', children: <div data-ai-id="report.performance" data-ai-title="Performance report"><ReportBody report={performance} /></div> },
        { key: 'qual', label: 'Quality report', children: <div data-ai-id="quality.gate" data-ai-title="Quality report"><ReportBody report={quality} /></div> },
      ]}
    />
  );
}
