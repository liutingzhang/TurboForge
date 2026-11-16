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
  jailbreak: '越狱处理',
  'quantize+compile': '量化 + 编译',
  speculative: 'MTP/DFlash2 随附',
  validate: '验证',
};

const STEP_STATUS: Record<string, { label: string; tone: 'ember' | 'bad' | 'default'; cls: string }> = {
  planned: { label: '已规划', tone: 'ember', cls: 'is-planned' },
  blocked: { label: '被阻止', tone: 'bad', cls: 'is-blocked' },
  skipped: { label: '跳过', tone: 'default', cls: 'is-skipped' },
};

function ResourceMeter({ label, est, avail }: { label: string; est: number | null; avail: number | null }) {
  if (est === null) {
    return <Meter label={label} hint="未建模 · 不臆造数值" ratio={0} />;
  }
  const ratio = avail && avail > 0 ? est / avail : 0;
  const over = avail !== null && est > avail;
  return (
    <Meter
      label={label}
      hint={`估算 ${humanBytes(est)} / 可用 ${humanBytes(avail)}`}
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
      title="构建计划"
      icon={<NodeIndexOutlined />}
      extra={
        plan && <Chip tone={plan.executable ? 'ok' : 'bad'}>{plan.executable ? '可执行' : '被阻止'}</Chip>
      }
    >
      {error && <Notice tone="bad" title="计划生成失败">{error}</Notice>}
      {!error && loading && <Skeleton active paragraph={{ rows: 4 }} />}
      {!error && !loading && !plan && (
        <EmptyHint icon={<NodeIndexOutlined />}>
          配置变更后自动生成计划：路线、兼容性、资源与风险一目了然
        </EmptyHint>
      )}
      {!error && !loading && plan && (
        <div className="fade">
          <div className="facts">
            <Fact label="后端配方" value={plan.backend ?? '（无）'} mono />
            <Fact label="越狱处理器" value={plan.processor ?? 'none'} mono />
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
              <Subhead>转换目标</Subhead>
              <div className="facts">
                <Fact label="目标" value={details.label} wide />
                <Fact label="标称位宽" value={(details.nominal_bits ?? '由源决定') + ' bit'} mono />
                <Fact label="典型 bpw" value={details.typical_bpw ?? '由源决定'} mono />
                <Fact label="设备" value={details.device} mono />
                <Fact label="组件" value={details.components ?? 'text'} mono />
                <Fact
                  label="存储格式"
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
                    label="说明"
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
              <Subhead>服务参数（自动 · 不改变产物）</Subhead>
              <div className="facts">
                <Fact label="KV 精度" value={plan.serving.kv_dtype} mono />
                <Fact
                  label="投机解码"
                  value={
                    plan.serving.spec
                      ? `${plan.serving.spec} · draft ${plan.serving.draft_tokens}`
                      : '无（未检测到组件）'
                  }
                  mono
                />
                <Fact
                  label="服务上下文"
                  value={
                    plan.serving.context_ceiling
                      ? `${plan.serving.max_context}（上限 ${plan.serving.context_ceiling}）`
                      : plan.serving.max_context
                  }
                  mono
                />
                <Fact
                  label="并发 / 端口"
                  value={`${plan.serving.max_concurrency} / ${plan.serving.port}`}
                  mono
                />
                <Fact
                  label="嵌入 Q4"
                  wide
                  value={
                    plan.serving.embedding_q4
                      ? '启用（原生 row-split Q8_G32 存储）'
                      : '省略（此配方的嵌入不属于已确认的原生 Q8 行格式）'
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
                      <span className="muted">自动服务旗标（{plan.serving.launcher_flags.length}）</span>
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
                与启动器共享几何、缓存精度和推测组件；基准工具支持的旗标不同，完整实测命令见报告。上下文/并发/端口可在第三步调整；
                缓存精度由几何决定；可在第三步关闭推测解码以减少额外开销。
              </p>
            </>
          )}

          {r && (
            <div data-ai-id="plan.resources" data-ai-title="资源预估（非实测）">
              <Subhead>资源预估（非实测）</Subhead>
              <ResourceMeter label="临时磁盘" est={r.temporary_disk_bytes} avail={r.available_disk_bytes} />
              <ResourceMeter label="服务显存" est={r.peak_vram_bytes} avail={r.available_vram_bytes} />
              <ResourceMeter label="峰值内存" est={r.peak_ram_bytes} avail={r.available_ram_bytes} />
              <div className="muted" style={{ fontSize: 12, marginBottom: 8 }}>
                输入 {humanBytes(r.input_bytes)}
                {r.estimated_output_bytes !== null && ` · 预计产物 ${humanBytes(r.estimated_output_bytes)}`}
              </div>
              <Collapse
                ghost
                size="small"
                items={[
                  {
                    key: 'assume',
                    label: <span className="muted">估算假设（{r.assumptions.length}）</span>,
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
              <Subhead>诊断与风险</Subhead>
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
                  label: <span className="muted">全部已验证目标（{capabilities.length}）</span>,
                  children: (
                    <Table
                      size="small"
                      pagination={false}
                      rowKey="name"
                      dataSource={capabilities}
                      columns={[
                        {
                          title: '配方',
                          dataIndex: 'name',
                          render: (v: string) => <span className="mono">{v}</span>,
                        },
                        {
                          title: 'bpw',
                          dataIndex: 'typical_bpw',
                          width: 70,
                          render: (v: number | null) => v ?? '由源',
                        },
                        {
                          title: '可用',
                          dataIndex: 'available',
                          width: 80,
                          render: (v: boolean) =>
                            v ? <Chip tone="ok">可用</Chip> : <Chip tone="bad">不可用</Chip>,
                        },
                        {
                          title: '原因 / 备注',
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
