import { Tooltip } from 'antd';
import type { Meta } from '../api/types';
import { humanBytes } from '../utils/format';

function EnvItem({ ok, label, hint, auto = false }: { ok: boolean; label: string; hint?: string; auto?: boolean }) {
  const node = (
    <span className="env-item">
      <span className={ok ? 'env-dot ok' : auto ? 'env-dot' : 'env-dot bad'} />
      {label}{!ok && auto ? ' · 自动准备' : ''}
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
          <div className="brand-tag">识别量化 · Dense 走 NInfer · MoE 交给 Basalt / Strata</div>
        </div>
      </div>
      <div className="topbar-spacer" />
      <div className="env">
        {meta ? (
          <>
            <EnvItem
              ok={meta.upstream.found}
              auto
              label="上游转换器"
              hint={
                meta.upstream.found
                  ? `已定位 tools.convert（revision ${(meta.upstream.revision || '').slice(0, 12)}）`
                  : '首次转换时脚本自动获取匹配的转换器，无需手动安装'
              }
            />
            <EnvItem ok={meta.env.torch} auto label="torch" hint="转换任务会在专用环境自动安装所需版本，无需手动安装" />
            <EnvItem
              ok={meta.env.transformers}
              auto
              label="transformers"
              hint="启用自动越狱时按需安装；默认转换无需提前安装"
            />
            {meta.gpu ? (
              <Tooltip
                title={`总显存 ${humanBytes(meta.gpu.total_bytes)} · 空闲 ${humanBytes(meta.gpu.free_bytes)}`}
              >
                <span className="env-gpu">
                  {meta.gpu.name.replace('NVIDIA GeForce ', '')} · {humanBytes(meta.gpu.free_bytes)} 空闲
                </span>
              </Tooltip>
            ) : (
              <EnvItem ok={false} label="GPU 未检测" hint="无法查询 NVIDIA 显存" />
            )}
            <Tooltip title={`NInfer 配方核对 revision：${meta.pinned_revision}；AWQ/GPTQ 使用独立 HF 引擎，尚未实机验收。`}>
              <span className="env-item">
                <span className="env-dot ok" />
                {meta.recipes.length} 个已接入方案
              </span>
            </Tooltip>
          </>
        ) : (
          <span className="env-item">
            <span className="env-dot" />
            加载环境…
          </span>
        )}
      </div>
    </header>
  );
}
