import { useEffect, useState } from 'react';
import { Alert, Button, Input, List, Select, Spin, Tag } from 'antd';
import { assistantApi } from '../../api/assistant';
import type { HelpEntry } from '../../api/assistant';
import { locateTopic } from '../../utils/assistantContext';

export const topicLabels: Record<string, string> = {start:'第一次使用',input:'模型输入',quantization:'选择量化',resources:'资源',
  calibration:'权重处理与校准',speculation:'推测解码',errors:'错误排查',quality:'质量报告',performance:'速度报告',
  serving:'启动模型',connection:'连接 AI',maintenance:'维护与迁移'};

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
    <p className="muted">离线帮助无需配置 AI，也不会向模型服务发送内容。</p>
    <Input.Search aria-label="搜索站内帮助" placeholder="搜索问题、参数或错误 code" value={query} onChange={e => {setQuery(e.target.value); setEntry(null);}} allowClear />
    <Select aria-label="帮助主题" value={topic} onChange={value => {setTopic(value); setEntry(null);}} options={[{value:'',label:'全部主题'}, ...Object.entries(topicLabels).map(([value,label]) => ({value,label}))]} />
    {error && <Alert type="error" message={error} />}
    {notice && <Alert type="info" message={notice} closable onClose={() => setNotice('')} />}
    {entry ? <article className="ai-help-entry">
      <Button type="link" onClick={() => setEntry(null)}>返回目录</Button>
      <h3>{entry.title}</h3><Tag>{entry.id}</Tag><Tag>{topicLabels[entry.topic]}</Tag>
      <p className="ai-prose">{entry.body}</p><p className="muted">{entry.evidence_level === 'runtime_rules' ? '依据当前后端能力规则生成。' : '依据项目代码与文档生成，并记录来源摘要。'}具体运行结果以本次任务证据为准。</p>
      <details><summary>知识来源</summary>{entry.source_refs.map(ref => <div className="mono" key={ref}>{ref}</div>)}</details>
      <Button onClick={() => onAsk(entry)}>结合我的情况问 AI</Button>
      <Button onClick={() => {if (!locateTopic(entry.topic)) setNotice('当前步骤没有显示此区域，请切换到相关步骤再定位。');}}>定位相关区域</Button>
    </article> : <Spin spinning={loading}><List dataSource={entries} locale={{emptyText:'未找到匹配资料；请尝试错误 code 或选择一个主题'}} renderItem={item =>
      <List.Item><Button type="text" className="ai-help-link" onClick={() => setEntry(item)}>{item.title}<small>{topicLabels[item.topic]} · {item.id}</small></Button></List.Item>} /></Spin>}
  </div>;
}
