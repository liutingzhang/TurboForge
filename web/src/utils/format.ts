const MIB = 1024 * 1024;
const GIB = 1024 * 1024 * 1024;

export function humanBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined) return '—';
  if (bytes >= GIB) return `${(bytes / GIB).toFixed(2)} GiB`;
  if (bytes >= MIB) return `${(bytes / MIB).toFixed(0)} MiB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(0)} KiB`;
  return `${bytes} B`;
}

export function bitsLabel(bits: number | null | undefined): string {
  return bits === null || bits === undefined ? '—' : `${bits}`;
}

export function bpwLabel(bpw: number | null | undefined): string {
  return bpw === null || bpw === undefined ? '由源决定' : `~${bpw} bpw`;
}

export function formatTime(iso: string | undefined): string {
  if (!iso) return '';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleTimeString('zh-CN', { hour12: false });
}

export function elapsedLabel(startMs: number | null, nowMs: number): string {
  if (startMs === null) return '00:00';
  const total = Math.max(0, Math.floor((nowMs - startMs) / 1000));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

// Render a metric value that may be a number, a {label: number} map, or null.
export function metricValueText(value: unknown): string {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'number') {
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }
  if (typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>);
    if (entries.length === 0) return '—';
    return entries
      .map(([k, v]) => `${k}: ${typeof v === 'number' ? v.toFixed(2) : String(v)}`)
      .join('  ·  ');
  }
  return String(value);
}