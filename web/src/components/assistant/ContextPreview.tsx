import { Alert, Button, Checkbox, Collapse, Input, Space } from 'antd';
import type { SelectionBlock } from '../../api/assistant';

export default function ContextPreview({blocks, setBlocks, summary, snapshot, includeSummary, setIncludeSummary,
  includeLogs, setIncludeLogs, onRefresh, capturedAt, disabled}: {
  blocks: SelectionBlock[]; setBlocks: (blocks: SelectionBlock[]) => void; summary: Record<string, unknown>;
  snapshot: Record<string, unknown> | null; includeSummary: boolean; setIncludeSummary: (value: boolean) => void;
  includeLogs: boolean; setIncludeLogs: (value: boolean) => void; onRefresh: () => void; capturedAt: string; disabled: boolean;
}) {
  const exportSnapshot = () => {
    if (!snapshot) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(snapshot, null, 2)], {type:'application/json;charset=utf-8'}));
    const anchor = document.createElement('a'); anchor.href = url;
    anchor.download = `TurboForge-diagnostic-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
    anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return <div className="ai-context">
    <Collapse size="small" items={[{key:'context', label:`Outgoing content preview · ${blocks.length} selection(s)`, children:<>
      {blocks.map((block,index) => <div key={`${block.id}-${index}`} className="ai-context-block">
        <Space><strong>{block.title}</strong><Button size="small" type="text" disabled={disabled} onClick={() => setBlocks(blocks.filter((_,i) => i !== index))}>Remove</Button></Space>
        <Input.TextArea aria-label={`Selection content ${index + 1}`} value={block.text} autoSize={{minRows:2,maxRows:6}} disabled={disabled}
          onChange={e => setBlocks(blocks.map((b,i) => i === index ? {...b,text:e.target.value} : b))} />
      </div>)}
      <Checkbox checked={includeSummary} disabled={disabled} onChange={e => setIncludeSummary(e.target.checked)}>Include the current model and workflow summary</Checkbox>
      <Checkbox checked={includeLogs} disabled={disabled} onChange={e => setIncludeLogs(e.target.checked)}>Include the current task log tail (up to 120 lines / 12 KiB)</Checkbox>
      {includeSummary && <details><summary>Page summary (supporting material)</summary><pre>{JSON.stringify(summary,null,2)}</pre></details>}
      <Button size="small" disabled={disabled} onClick={onRefresh}>Refresh and view the backend send snapshot</Button>
      <Button size="small" disabled={disabled || !snapshot} onClick={exportSnapshot}>Export the current redacted diagnostics</Button>
      {snapshot && <pre className="ai-snapshot">{JSON.stringify(snapshot,null,2)}</pre>}
      {capturedAt && <small>Captured at {new Date(capturedAt).toLocaleTimeString()}; reselect after the page changes.</small>}
      <Alert type="info" showIcon message="Only the selections and checked materials you confirm are sent. Logs may contain private information; review the preview. You can uncheck attachments or remove selections." />
    </>}]} />
  </div>;
}
