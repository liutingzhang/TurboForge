import { useEffect, useState } from 'react';
import { Alert, Button, Input, List, Select, Spin, Tag } from 'antd';
import { assistantApi } from '../../api/assistant';
import type { HelpEntry } from '../../api/assistant';
import { locateTopic } from '../../utils/assistantContext';

export const topicLabels: Record<string, string> = {start:'First use',input:'Model input',quantization:'Choosing quantization',resources:'Resources',
  calibration:'Weight processing & calibration',speculation:'Speculative decoding',errors:'Troubleshooting',quality:'Quality report',performance:'Speed report',
  serving:'Starting the model',connection:'Connecting AI',maintenance:'Maintenance & migration'};

export default function HelpCenter({initialId = '', onAsk}: {initialId?: string; onAsk: (entry: HelpEntry) => void}) {
  const [query, setQuery] = useState(''); const [topic, setTopic] = useState('');
  const [entries, setEntries] = useState<HelpEntry[]>([]); const [entry, setEntry] = useState<HelpEntry | null>(null);
  const [loading, setLoading] = useState(false); const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  useEffect(() => {
    let cancelled = false;
    if (initialId) assistantApi.entry(initialId).then(value => {if (!cancelled) setEntry(value);}).catch(err => {if (!cancelled) setError((err as Error).message);});
    return () => {cancelled = true;};
  }, [initialId]);
  useEffect(() => {
    let cancelled = false;
    const timer = setTimeout(() => {
      setLoading(true); setError('');
      assistantApi.help(query, topic).then(result => {if (!cancelled) setEntries(result.entries);})
        .catch(err => {if (!cancelled) setError((err as Error).message);}).finally(() => {if (!cancelled) setLoading(false);});
    }, 180);
    return () => {cancelled = true; clearTimeout(timer);};
  }, [query, topic]);
  return <div className="ai-help">
    <p className="muted">Offline help needs no AI configuration and sends no content to the model service.</p>
    <Input.Search aria-label="Search in-app help" placeholder="Search questions, parameters or error codes" value={query} onChange={e => {setQuery(e.target.value); setEntry(null);}} allowClear />
    <Select aria-label="Help topic" value={topic} onChange={value => {setTopic(value); setEntry(null);}} options={[{value:'',label:'All topics'}, ...Object.entries(topicLabels).map(([value,label]) => ({value,label}))]} />
    {error && <Alert type="error" message={error} />}
    {notice && <Alert type="info" message={notice} closable onClose={() => setNotice('')} />}
    {entry ? <article className="ai-help-entry">
      <Button type="link" onClick={() => setEntry(null)}>Back to contents</Button>
      <h3>{entry.title}</h3><Tag>{entry.id}</Tag><Tag>{topicLabels[entry.topic]}</Tag>
      <p className="ai-prose">{entry.body}</p><p className="muted">{entry.evidence_level === 'runtime_rules' ? 'Generated from the current backend capability rules.' : 'Generated from project code and docs, with source summaries recorded.'} Actual runtime results follow this run's evidence.</p>
      <details><summary>Knowledge sources</summary>{entry.source_refs.map(ref => <div className="mono" key={ref}>{ref}</div>)}</details>
      <Button onClick={() => onAsk(entry)}>Ask AI about my situation</Button>
      <Button onClick={() => {if (!locateTopic(entry.topic)) setNotice('This area is not visible in the current step; switch to the related step before locating.');}}>Locate related area</Button>
    </article> : <Spin spinning={loading}><List dataSource={entries} locale={{emptyText:'No matching material found; try an error code or pick a topic'}} renderItem={item =>
      <List.Item><Button type="text" className="ai-help-link" onClick={() => setEntry(item)}>{item.title}<small>{topicLabels[item.topic]} · {item.id}</small></Button></List.Item>} /></Spin>}
  </div>;
}
