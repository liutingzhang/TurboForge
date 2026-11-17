import { Collapse, Skeleton, Table } from 'antd';
import { NodeIndexOutlined } from '@ant-design/icons';
import type { BackendCapability, ForgePlan } from '../api/types';
import { humanBytes } from '../utils/format';
import { Chip, Diag, EmptyHint, Fact, Meter, Notice, Panel, Subhead } from './ui';

interface Props {
  plan: ForgePlan | null;
  capabilities: BackendCapability[];
  loading: boolean;
  error: string | null;
}

const STEP_TITLE: Record<string, string> = {
  jailbreak: 'Abliteration',
  'quantize+compile': 'Quantize + compile',
  speculative: 'MTP/DFlash2 carried',
  validate: 'Validate',
};

const STEP_STATUS: Record<string, { label: string; tone: 'ember' | 'bad' | 'default'; cls: string }> = {
  planned: { label: 'Planned', tone: 'ember', cls: 'is-planned' },
  blocked: { label: 'Blocked', tone: 'bad', cls: 'is-blocked' },
  skipped: { label: 'Skipped', tone: 'default', cls: 'is-skipped' },
};

function ResourceMeter({ label, est, avail }: { label: string; est: number | null; avail: number | null }) {
  if (est === null) {
    return <Meter label={label} hint="Not modeled · no invented numbers" ratio={0} />;
  }
  const ratio = avail && avail > 0 ? est / avail : 0;
  const over = avail !== null && est > avail;
  return (
    <Meter
      label={label}
      hint={`Estimated ${humanBytes(est)} / available ${humanBytes(avail)}`}
      ratio={ratio}
      over={over}
    />
  );
}

export default function PlanPreview({ plan, capabilities, loading, error }: Props) {
  const compileStep = plan?.steps.find((s) => s.name === 'quantize+compile');
  const details = compileStep?.details ?? {};
  const r = plan?.resources ?? null;
  const order = { error: 0, warning: 1, info: 2 } as const;
  const diags = plan
    ? [...plan.diagnostics].sort((a, b) => order[a.severity] - order[b.severity])
    : [];

  return (
    <Panel
      title="Build plan"
      icon={<NodeIndexOutlined />}
      extra={
        plan && <Chip tone={plan.executable ? 'ok' : 'bad'}>{plan.executable ? 'Executable' : 'Blocked'}</Chip>
      }
    >
      {error && <Notice tone="bad" title="Plan generation failed">{error}</Notice>}
      {!error && loading && <Skeleton active paragraph={{ rows: 4 }} />}
      {!error && !loading && !plan && (
        <EmptyHint icon={<NodeIndexOutlined />}>
          The plan regenerates after config changes: route, compatibility, resources and risks at a glance
        </EmptyHint>
      )}
      {!error && !loading && plan && (
        <div className="fade">
          <div className="facts">
            <Fact label="Backend recipe" value={plan.backend ?? '(none)'} mono />
            <Fact label="Abliteration processor" value={plan.processor ?? 'none'} mono />
          </div>

          <div className="route">
            {plan.steps.map((s, i) => {
              const m = STEP_STATUS[s.status] ?? STEP_STATUS.planned;
              return (
                <div key={i} className={'route-step ' + m.cls}>
                  <span className="route-dot" />
                  <div className="route-main">
                    <div className="route-name">
                      {STEP_TITLE[s.name] ?? s.name}
                      <Chip tone={m.tone}>{m.label}</Chip>
                    </div>
                    <div className="route-impl">{s.implementation}</div>
                  </div>
                </div>
              );
            })}
          </div>

          {compileStep && (
            <>
              <Subhead>Conversion target</Subhead>
              <div className="facts">
                <Fact label="Target" value={details.label} wide />
                <Fact label="Nominal bits" value={(details.nominal_bits ?? 'Source-defined') + ' bit'} mono />
                <Fact label="Typical bpw" value={details.typical_bpw ?? 'Source-defined'} mono />
                <Fact label="Device" value={details.device} mono />
                <Fact label="Components" value={details.components ?? 'text'} mono />
                <Fact
                  label="Storage formats"
                  wide
                  value={
                    <span className="chip-row">
                      {(details.storage_formats ?? []).map((f: string) => (
                        <Chip key={f} mono>
                          {f}
                        </Chip>
                      ))}
                    </span>
                  }
                />
                {details.notes?.length > 0 && (
                  <Fact
                    label="Notes"
                    wide
                    value={
                      <ul className="diag-alts" style={{ marginTop: 0 }}>
                        {details.notes.map((n: string, i: number) => (
                          <li key={i}>{n}</li>
                        ))}
                      </ul>
                    }
                  />
                )}
              </div>
            </>
          )}

          {plan.serving && (
            <>
              <Subhead>Serving parameters (automatic · does not change the artifact)</Subhead>
              <div className="facts">
                <Fact label="KV precision" value={plan.serving.kv_dtype} mono />
                <Fact
                  label="Speculative decoding"
                  value={
                    plan.serving.spec
                      ? `${plan.serving.spec} · draft ${plan.serving.draft_tokens}`
                      : 'None (no component detected)'
                  }
                  mono
                />
                <Fact
                  label="Serving context"
                  value={
                    plan.serving.context_ceiling
                      ? `${plan.serving.max_context} (ceiling ${plan.serving.context_ceiling})`
                      : plan.serving.max_context
                  }
                  mono
                />
                <Fact
                  label="Concurrency / port"
                  value={`${plan.serving.max_concurrency} / ${plan.serving.port}`}
                  mono
                />
                <Fact
                  label="Embedding Q4"
                  wide
                  value={
                    plan.serving.embedding_q4
                      ? 'Enabled (native row-split Q8_G32 storage)'
                      : 'Omitted (the embedding of this recipe is not a confirmed native Q8 row format)'
                  }
                />
              </div>
              <Collapse
                ghost
                size="small"
                style={{ marginBottom: 8 }}
                items={[
                  {
                    key: 'flags',
                    label: (
                      <span className="muted">Automatic serving flags ({plan.serving.launcher_flags.length})</span>
                    ),
                    children: (
                      <div className="console" style={{ maxHeight: 200 }}>
                        {plan.serving.launcher_flags.join('\n')}
                      </div>
                    ),
                  },
                ]}
              />
              <p className="muted" style={{ fontSize: 12, marginBottom: 8, lineHeight: 1.6 }}>
                Shares geometry, cache precision and speculative components with the launcher; the benchmark tool supports different flags — see the report for the full measured command. Context/concurrency/port can be adjusted in step 3; 
                cache precision follows the geometry; speculative decoding can be turned off in step 3 to reduce extra overhead.
              </p>
            </>
          )}

          {r && (
            <div data-ai-id="plan.resources" data-ai-title="Resource estimates (not measured)">
              <Subhead>Resource estimates (not measured)</Subhead>
              <ResourceMeter label="Temporary disk" est={r.temporary_disk_bytes} avail={r.available_disk_bytes} />
              <ResourceMeter label="Serving VRAM" est={r.peak_vram_bytes} avail={r.available_vram_bytes} />
              <ResourceMeter label="Peak RAM" est={r.peak_ram_bytes} avail={r.available_ram_bytes} />
              <div className="muted" style={{ fontSize: 12, marginBottom: 8 }}>
                Input {humanBytes(r.input_bytes)}
                {r.estimated_output_bytes !== null && ` · estimated artifact ${humanBytes(r.estimated_output_bytes)}`}
              </div>
              <Collapse
                ghost
                size="small"
                items={[
                  {
                    key: 'assume',
                    label: <span className="muted">Estimate assumptions ({r.assumptions.length})</span>,
                    children: (
                      <ul className="diag-alts" style={{ marginTop: 0 }}>
                        {r.assumptions.map((a, i) => (
                          <li key={i}>{a}</li>
                        ))}
                      </ul>
                    ),
                  },
                ]}
              />
            </div>
          )}

          {diags.length > 0 && (
            <>
              <Subhead>Diagnostics and risks</Subhead>
              {diags.map((d, i) => (
                <Diag key={d.code + '-' + i} d={d} />
              ))}
            </>
          )}

          {capabilities.length > 0 && (
            <Collapse
              ghost
              size="small"
              style={{ marginTop: 8 }}
              items={[
                {
                  key: 'caps',
                  label: <span className="muted">All verified targets ({capabilities.length})</span>,
                  children: (
                    <Table
                      size="small"
                      pagination={false}
                      rowKey="name"
                      dataSource={capabilities}
                      columns={[
                        {
                          title: 'Recipe',
                          dataIndex: 'name',
                          render: (v: string) => <span className="mono">{v}</span>,
                        },
                        {
                          title: 'bpw',
                          dataIndex: 'typical_bpw',
                          width: 70,
                          render: (v: number | null) => v ?? 'From source',
                        },
                        {
                          title: 'Available',
                          dataIndex: 'available',
                          width: 80,
                          render: (v: boolean) =>
                            v ? <Chip tone="ok">Available</Chip> : <Chip tone="bad">Unavailable</Chip>,
                        },
                        {
                          title: 'Reason / notes',
                          dataIndex: 'unavailable_reason',
                          render: (v: string | null, row: BackendCapability) => (
                            <span className="muted">{v ?? row.notes[0] ?? '—'}</span>
                          ),
                        },
                      ]}
                    />
                  ),
                },
              ]}
            />
          )}
        </div>
      )}
    </Panel>
  );
}
