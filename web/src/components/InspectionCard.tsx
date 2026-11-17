import { Skeleton } from 'antd';
import { ScanOutlined } from '@ant-design/icons';
import type { Meta, ModelInspection } from '../api/types';
import { humanBytes } from '../utils/format';
import { Chip, Diag, EmptyHint, Notice, Panel, Stat } from './ui';

interface Props {
  inspection: ModelInspection | null;
  loading: boolean;
  error: string | null;
  meta: Meta | null;
}

export default function InspectionCard({ inspection, loading, error, meta }: Props) {
  const kindLabel =
    meta?.choices.input_kind.find((k) => k.value === inspection?.input_kind)?.label ??
    inspection?.input_kind;

  const spec = inspection?.speculative_components ?? {};
  const mtp = Boolean(spec.mtp?.detected);
  const dflash2 = Boolean(spec.dflash2?.detected);

  return (
    <Panel title="Model identification" icon={<ScanOutlined />}>
      {error && <Notice tone="bad" title="Cannot identify the input">{error}</Notice>}
      {!error && loading && <Skeleton active paragraph={{ rows: 3 }} />}
      {!error && !loading && !inspection && (
        <EmptyHint icon={<ScanOutlined />}>
          After entering the input path, architecture, precision and MTP / DFlash2 components are identified automatically
        </EmptyHint>
      )}
      {!error && !loading && inspection && (
        <div className="fade">
          <div className="stat-grid">
            <Stat label="Architecture" value={inspection.architecture} mono wide />
            <Stat label="Input kind" value={kindLabel} />
            <Stat label="Weight precision" value={inspection.weight_precision} mono />
            <Stat label="Tensors" value={inspection.tensor_count} mono />
            <Stat label="Shards" value={inspection.shard_count} mono />
            <Stat label="Weight size" value={humanBytes(inspection.tensor_bytes)} mono />
            {inspection.element_count > 0 && (
              <Stat label="Elements" value={inspection.element_count.toLocaleString()} mono />
            )}
            {inspection.license && <Stat label="License" value={inspection.license} wide />}
          </div>

          <div className="chip-row">
            {Object.entries(inspection.dtype_counts).map(([dtype, count]) => (
              <Chip key={dtype} mono>
                {dtype} × {count}
              </Chip>
            ))}
            <Chip tone={mtp ? 'ok' : 'default'}>MTP {mtp ? 'detected · carried automatically' : 'not detected'}</Chip>
            <Chip tone={dflash2 ? 'ok' : 'default'}>
              DFlash2 {dflash2 ? 'detected · carried automatically' : 'not detected'}
            </Chip>
          </div>

          {inspection.diagnostics.length > 0 && (
            <div style={{ marginTop: 14 }}>
              {inspection.diagnostics.map((d, i) => (
                <Diag key={i} d={d} />
              ))}
            </div>
          )}
        </div>
      )}
    </Panel>
  );
}
