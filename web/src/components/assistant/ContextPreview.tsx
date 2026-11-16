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
    <Collapse size="small" items={[{key:'context', label:`发送内容预览 · ${blocks.length} 个选区`, children:<>
      {blocks.map((block,index) => <div key={`${block.id}-${index}`} className="ai-context-block">
        <Space><strong>{block.title}</strong><Button size="small" type="text" disabled={disabled} onClick={() => setBlocks(blocks.filter((_,i) => i !== index))}>移除</Button></Space>
        <Input.TextArea aria-label={`选区内容 ${index + 1}`} value={block.text} autoSize={{minRows:2,maxRows:6}} disabled={disabled}
          onChange={e => setBlocks(blocks.map((b,i) => i === index ? {...b,text:e.target.value} : b))} />
      </div>)}
      <Checkbox checked={includeSummary} disabled={disabled} onChange={e => setIncludeSummary(e.target.checked)}>附带当前模型与业务摘要</Checkbox>
      <Checkbox checked={includeLogs} disabled={disabled} onChange={e => setIncludeLogs(e.target.checked)}>附带当前任务日志尾部（最多 120 行/12KiB）</Checkbox>
      {includeSummary && <details><summary>页面摘要（辅助材料）</summary><pre>{JSON.stringify(summary,null,2)}</pre></details>}
      <Button size="small" disabled={disabled} onClick={onRefresh}>刷新并查看后端发送快照</Button>
      <Button size="small" disabled={disabled || !snapshot} onClick={exportSnapshot}>导出当前脱敏诊断</Button>
      {snapshot && <pre className="ai-snapshot">{JSON.stringify(snapshot,null,2)}</pre>}
      {capturedAt && <small>选取于 {new Date(capturedAt).toLocaleTimeString()}；页面变化后请重新选取。</small>}
      <Alert type="info" showIcon message="只发送你确认的选区与勾选资料。日志可能包含私密信息，请检查预览；可取消附带或移除选区。" />
    </>}]} />
  </div>;
}
