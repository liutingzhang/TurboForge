import { useState } from 'react';
import { Alert, Button, Checkbox, Collapse, Modal, Space } from 'antd';
import { assistantApi, type StorageInventory } from '../api/assistant';
import { humanBytes } from '../utils/format';

export default function StoragePanel({disabled}: {disabled: boolean}) {
  const [inventory, setInventory] = useState<StorageInventory | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const scan = async () => {
    setBusy(true); setError('');
    try {setInventory(await assistantApi.storage()); setSelected([]);}
    catch (e) {setError((e as Error).message);}
    finally {setBusy(false);}
  };
  const clean = async () => {
    setBusy(true); setError('');
    try {setInventory(await assistantApi.cleanStorage(selected)); setSelected([]);}
    catch (e) {setError((e as Error).message);}
    finally {setBusy(false);}
  };
  const bytes = inventory?.items.filter(item => selected.includes(item.id)).reduce((total, item) => total + item.bytes, 0) ?? 0;
  return <Collapse ghost items={[{key:'storage', label:'磁盘空间与失败产物清理', children:<div data-ai-ignore>
    <p>先扫描并选择失败任务，再确认删除。成功模型、原始模型和依赖环境会保留。</p>
    <Space><Button loading={busy} disabled={disabled || busy} onClick={scan}>扫描构建占用</Button>
      <Button danger disabled={disabled || busy || !selected.length} onClick={() => Modal.confirm({title:'删除选中的失败构建？', content:`将永久删除 ${selected.length} 个失败构建及其日志，约 ${humanBytes(bytes)}。`, okText:'确认删除', cancelText:'保留', onOk:clean})}>清理所选 {humanBytes(bytes)}</Button></Space>
    {error && <Alert type="error" message={error} style={{marginTop:12}} />}
    {inventory && <><p>缓存所在盘可用：{humanBytes(inventory.free_bytes)}</p>
      {inventory.items.length ? inventory.items.map(item => <p key={item.id}><Checkbox disabled={!item.removable || busy || disabled} checked={selected.includes(item.id)} onChange={event => setSelected(values => event.target.checked ? [...values, item.id] : values.filter(id => id !== item.id))}>{item.id} · {humanBytes(item.bytes)} · {item.removable ? '失败产物，可清理' : '保留'}</Checkbox></p>) : <p>没有可列出的受管理构建。</p>}
      <p className="muted">{inventory.note}</p></>}
  </div>}]} />;
}
