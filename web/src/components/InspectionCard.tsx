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
    <Panel title="模型识别" icon={<ScanOutlined />}>
      {error && <Notice tone="bad" title="无法识别输入">{error}</Notice>}
      {!error && loading && <Skeleton active paragraph={{ rows: 3 }} />}
      {!error && !loading && !inspection && (
        <EmptyHint icon={<ScanOutlined />}>
          填写输入路径后自动识别架构、精度与 MTP / DFlash2 组件
        </EmptyHint>
      )}
      {!error && !loading && inspection && (
        <div className="fade">
          <div className="stat-grid">
            <Stat label="架构" value={inspection.architecture} mono wide />
            <Stat label="输入类型" value={kindLabel} />
            <Stat label="权重精度" value={inspection.weight_precision} mono />
            <Stat label="张量数" value={inspection.tensor_count} mono />
            <Stat label="分片数" value={inspection.shard_count} mono />
            <Stat label="权重体积" value={humanBytes(inspection.tensor_bytes)} mono />
            {inspection.element_count > 0 && (
              <Stat label="元素数" value={inspection.element_count.toLocaleString()} mono />
            )}
            {inspection.license && <Stat label="许可证" value={inspection.license} wide />}
          </div>

          <div className="chip-row">
            {Object.entries(inspection.dtype_counts).map(([dtype, count]) => (
              <Chip key={dtype} mono>
                {dtype} × {count}
              </Chip>
            ))}
            <Chip tone={mtp ? 'ok' : 'default'}>MTP {mtp ? '已检测 · 自动随附' : '未检测'}</Chip>
            <Chip tone={dflash2 ? 'ok' : 'default'}>
              DFlash2 {dflash2 ? '已检测 · 自动随附' : '未检测'}
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
