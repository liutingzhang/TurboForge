import { Tooltip } from 'antd';
import type { Meta } from '../api/types';
import { humanBytes } from '../utils/format';

function EnvItem({ ok, label, hint, auto = false }: { ok: boolean; label: string; hint?: string; auto?: boolean }) {
  const node = (
    <span className="env-item">
      <span className={ok ? 'env-dot ok' : auto ? 'env-dot' : 'env-dot bad'} />
      {label}{!ok && auto ? ' · auto-prepared' : ''}
    </span>
  );
  return hint ? <Tooltip title={hint}>{node}</Tooltip> : node;
}

export default function AppHeader({ meta }: { meta: Meta | null }) {
  return (
    <header className="topbar">
      <div className="brand">
        <div className="brand-glyph">TF</div>
        <div>
          <div className="brand-name">TurboForge</div>
          <div className="brand-tag">Identify & quantize · Dense via NInfer · MoE via Basalt / Strata</div>
        </div>
      </div>
      <div className="topbar-spacer" />
      <div className="env">
        {meta ? (
          <>
            <EnvItem
              ok={meta.upstream.found}
              auto
              label="Upstream converter"
              hint={
                meta.upstream.found
                  ? `tools.convert located (revision ${(meta.upstream.revision || '').slice(0, 12)})`
                  : 'On the first conversion the script fetches the matching converter automatically; no manual install needed'
              }
            />
            <EnvItem ok={meta.env.torch} auto label="torch" hint="Conversion tasks install the required version automatically in a dedicated environment; no manual install needed" />
            <EnvItem
              ok={meta.env.transformers}
              auto
              label="transformers"
              hint="Installed on demand when automatic abliteration is enabled; default conversion needs no pre-install"
            />
            {meta.gpu ? (
              <Tooltip
                title={`Total VRAM ${humanBytes(meta.gpu.total_bytes)} · free ${humanBytes(meta.gpu.free_bytes)}`}
              >
                <span className="env-gpu">
                  {meta.gpu.name.replace('NVIDIA GeForce ', '')} · {humanBytes(meta.gpu.free_bytes)} free
                </span>
              </Tooltip>
            ) : (
              <EnvItem ok={false} label="GPU not detected" hint="NVIDIA VRAM could not be queried" />
            )}
            <Tooltip title={`NInfer recipes verified against revision ${meta.pinned_revision}; AWQ/GPTQ uses a separate HF engine, with only limited on-machine acceptance recorded.`}>
              <span className="env-item">
                <span className="env-dot ok" />
                {meta.recipes.length} wired-in plans
              </span>
            </Tooltip>
          </>
        ) : (
          <span className="env-item">
            <span className="env-dot" />
            Loading environment…
          </span>
        )}
      </div>
    </header>
  );
}
