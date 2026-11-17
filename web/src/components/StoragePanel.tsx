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
  return <Collapse ghost items={[{key:'storage', label:'Disk space and failed-artifact cleanup', children:<div data-ai-ignore>
    <p>Scan and select failed builds first, then confirm deletion. Successful models, original models and dependency environments are kept.</p>
    <Space><Button loading={busy} disabled={disabled || busy} onClick={scan}>Scan build disk usage</Button>
      <Button danger disabled={disabled || busy || !selected.length} onClick={() => Modal.confirm({title:'Delete the selected failed builds?', content:`This permanently deletes ${selected.length} failed build(s) and their logs, about ${humanBytes(bytes)}.`, okText:'Delete', cancelText:'Keep', onOk:clean})}>Clean selected {humanBytes(bytes)}</Button></Space>
    {error && <Alert type="error" message={error} style={{marginTop:12}} />}
    {inventory && <><p>Free on the cache drive: {humanBytes(inventory.free_bytes)}</p>
      {inventory.items.length ? inventory.items.map(item => <p key={item.id}><Checkbox disabled={!item.removable || busy || disabled} checked={selected.includes(item.id)} onChange={event => setSelected(values => event.target.checked ? [...values, item.id] : values.filter(id => id !== item.id))}>{item.id} · {humanBytes(item.bytes)} · {item.removable ? 'failed artifact, removable' : 'kept'}</Checkbox></p>) : <p>No managed builds to list.</p>}
      <p className="muted">{inventory.note}</p></>}
  </div>}]} />;
}
