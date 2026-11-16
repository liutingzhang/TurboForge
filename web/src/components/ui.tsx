import type { ReactNode } from 'react';
import { Button } from 'antd';
import type { Diagnostic } from '../api/types';

export function cx(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(' ');
}

type Tone = 'default' | 'ok' | 'warn' | 'bad' | 'info' | 'ember';

export function Chip({ tone = 'default', mono, children }: { tone?: Tone; mono?: boolean; children: ReactNode }) {
  return <span className={cx('chip', mono && 'mono', tone !== 'default' && 'chip-' + tone)}>{children}</span>;
}

export function Notice({ tone = 'info', title, children }: { tone?: 'info' | 'warn' | 'bad' | 'ok'; title?: ReactNode; children?: ReactNode }) {
  return (
    <div className={cx('notice', 'notice-' + tone)}>
      {title && <div className="notice-title">{title}</div>}
      {children && <div className="notice-body">{children}</div>}
    </div>
  );
}

export function Diag({ d }: { d: Diagnostic }) {
  return (
    <div className={cx('diag', 'diag-' + d.severity)} data-ai-id={`diagnostic.${d.code}`} data-ai-code={d.code} data-ai-title="诊断">
      <div className="diag-code">{d.code}<Button size="small" type="link" data-ai-ignore onClick={() => window.dispatchEvent(new CustomEvent('turboforge:ask', {detail:{id:`diagnostic.${d.code}`,title:'诊断',code:d.code,text:[d.message,...d.alternatives].join('\n')}}))}>解释此问题</Button></div>
      <div className="diag-msg">{d.message}</div>
      {d.alternatives.length > 0 && (
        <ul className="diag-alts">
          {d.alternatives.map((a, i) => <li key={i}>{a}</li>)}
        </ul>
      )}
    </div>
  );
}

export function Panel({ title, icon, extra, children, className }: { title?: ReactNode; icon?: ReactNode; extra?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={cx('panel', className)} data-ai-id={typeof title === 'string' ? `panel.${title}` : undefined} data-ai-title={typeof title === 'string' ? title : undefined}>
      {(title || extra) && (
        <div className="panel-head">
          <span className="panel-title">{icon}{title}</span>
          <span className="panel-head-spacer" />
          {extra}
        </div>
      )}
      <div className="panel-body">{children}</div>
    </section>
  );
}

export function StepHeader({ index, title, note }: { index: ReactNode; title: ReactNode; note?: ReactNode }) {
  return (
    <div className="step-head">
      <span className="step-index">{index}</span>
      <span className="step-title">{title}</span>
      {note && <span className="step-note">{note}</span>}
    </div>
  );
}

export function Subhead({ children }: { children: ReactNode }) {
  return <div className="subhead">{children}</div>;
}

export function Stat({ label, value, mono, wide }: { label: ReactNode; value: ReactNode; mono?: boolean; wide?: boolean }) {
  return (
    <div className={cx('stat', wide && 'stat-wide')}>
      <div className="stat-label">{label}</div>
      <div className={cx('stat-value', mono && 'mono')}>{value}</div>
    </div>
  );
}

export function Fact({ label, value, mono, wide }: { label: ReactNode; value: ReactNode; mono?: boolean; wide?: boolean }) {
  return (
    <div className={cx('fact', wide && 'fact-wide')}>
      <div className="fact-label">{label}</div>
      <div className={cx('fact-value', mono && 'mono')}>{value}</div>
    </div>
  );
}

export function Meter({ label, hint, ratio, over }: { label: ReactNode; hint?: ReactNode; ratio: number; over?: boolean }) {
  const pct = Math.max(0, Math.min(1, ratio)) * 100;
  return (
    <div className="meter">
      <div className="meter-head">
        <span>{label}</span>
        {hint && <span className="muted mono">{hint}</span>}
      </div>
      <div className="meter-track">
        <div className={cx('meter-fill', over && 'over')} style={{ width: pct + '%' }} />
      </div>
    </div>
  );
}

export function EmptyHint({ icon, children }: { icon?: ReactNode; children: ReactNode }) {
  return (
    <div className="empty-hint">
      {icon}
      <div>{children}</div>
    </div>
  );
}
