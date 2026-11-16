import { useState } from 'react';
import { Alert, Button, Checkbox, Input, InputNumber, Select, Space, Switch } from 'antd';
import { assistantApi } from '../../api/assistant';
import type { AssistantConfig, ConnectionInput } from '../../api/assistant';

export default function AssistantSettings({config, onSaved}: {config: AssistantConfig; onSaved: () => Promise<void>}) {
  const [base, setBase] = useState(config.base_url);
  const [model, setModel] = useState(config.model);
  const [key, setKey] = useState('');
  const [action, setAction] = useState<ConnectionInput['secret_action']>(config.has_api_key ? 'keep' : 'replace');
  const [local, setLocal] = useState(config.auth_mode === 'none');
  const [stream, setStream] = useState(config.stream);
  const [context, setContext] = useState(config.context_tokens);
  const [output, setOutput] = useState(config.max_output_tokens);
  const [timeout, setTimeoutValue] = useState(config.timeout_seconds);
  const [scope, setScope] = useState<'session' | 'dotenv'>('session');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [tested, setTested] = useState('');
  const changed = () => { setTested(''); setNotice(''); };
  const input = (): ConnectionInput => ({base_url: base, model, auth_mode: local ? 'none' : 'bearer',
    secret_action: local ? 'clear' : action, ...(action === 'replace' && !local ? {api_key: key} : {}),
    stream, context_tokens: context, max_output_tokens: output, timeout_seconds: timeout, scope});
  const test = async () => {
    setBusy(true); setError(''); setNotice('');
    const value = input(); setKey('');
    try {
      const result = await assistantApi.test(value); setTested(result.tested_config_id);
      setNotice(`文本调用测试成功，${result.latency_ms} ms；配置保留在后端 5 分钟，可直接应用。`);
    } catch (err) { setError((err as Error).message); }
    finally { setBusy(false); }
  };
  const save = async () => {
    setBusy(true); setError('');
    const value = tested ? {tested_config_id: tested, scope} : input(); setKey('');
    try {
      const result = await assistantApi.save(value); setNotice(result.notice ?? '配置已应用');
      await onSaved();
    } catch (err) { setError((err as Error).message); }
    finally { setBusy(false); }
  };
  const reset = async () => {
    setBusy(true); setKey(''); setError('');
    try { await assistantApi.reset(); await onSaved(); }
    catch (err) { setError((err as Error).message); }
    finally { setBusy(false); }
  };
  return <div className="ai-settings" data-ai-id="assistant.settings" data-ai-ignore>
    <p className="muted">生效来源：{({default:'未配置',environment:'进程环境变量',dotenv:'本机 .env',session:'本次会话'} as Record<string,string>)[config.source] ?? config.source}。已有密钥只显示配置状态。</p>
    {config.config_error && <Alert type="error" message={config.config_error} showIcon />}
    <label htmlFor="ai-base">Base URL</label>
    <Input id="ai-base" value={base} placeholder="https://llm.example.com/v1" disabled={busy} onChange={e => {
      changed(); setBase(e.target.value);
      try { if (new URL(e.target.value).origin !== new URL(config.base_url).origin) {setAction('replace'); setKey('');} } catch { setAction('replace'); }
    }} />
    <small>填写接口根地址，不要填写 /chat/completions。远端使用 HTTPS。</small>
    <label htmlFor="ai-model">模型名称（实际 API 模型 ID）</label>
    <Input id="ai-model" value={model} disabled={busy} onChange={e => {changed(); setModel(e.target.value);}} />
    <Checkbox checked={local} disabled={busy} onChange={e => {changed(); setLocal(e.target.checked); setKey(''); setAction(e.target.checked ? 'clear' : 'replace');}}>本地服务无需 API Key（需管理员允许该 origin）</Checkbox>
    {!local && <>
      <label>API Key · {config.has_api_key ? '后端已配置' : '尚未配置'}</label>
      <Select aria-label="密钥操作" value={action} disabled={busy} options={[
        {value:'keep',label:'保留当前 Key',disabled:!config.has_api_key}, {value:'replace',label:'输入/替换 Key'}, {value:'clear',label:'清除 Key'},
      ]} onChange={value => {changed(); setAction(value); setKey('');}} />
      {action === 'replace' && <Input.Password aria-label="API Key" value={key} autoComplete="off" disabled={busy}
        placeholder={tested ? '测试后的 Key 已清空，可直接应用' : '仅发送到本机后端，不进入聊天内容'} onChange={e => {changed(); setKey(e.target.value);}} />}
    </>}
    <div className="ai-setting-row"><span>流式回答</span><Switch checked={stream} disabled={busy} onChange={value => {changed(); setStream(value);}} /></div>
    <label>服务上下文上限（token）</label><InputNumber aria-label="服务上下文上限" min={256} max={1048576} value={context} disabled={busy} onChange={v => {if (v !== null) {changed(); setContext(v);}}} />
    <label>输出上限（token）</label><InputNumber aria-label="输出上限" min={16} max={32768} value={output} disabled={busy} onChange={v => {if (v !== null) {changed(); setOutput(v);}}} />
    <label>总超时（秒）</label><InputNumber aria-label="总超时" min={5} max={300} value={timeout} disabled={busy} onChange={v => {if (v !== null) {changed(); setTimeoutValue(v);}}} />
    <label>保存方式</label><Select value={scope} disabled={busy} onChange={setScope} options={[
      {value:'session',label:'仅本次会话（推荐）'}, {value:'dotenv',label:'本机 .env 全局默认',disabled:!config.allow_persist},
    ]} />
    <small>测试会产生一次短文本调用费用。Key 提交后立即清空输入框。会话配置过期/重启后丢失；.env 为明文文件，默认禁止网页持久化。</small>
    {error && <Alert type="error" message={error} showIcon />}
    {notice && <Alert type="success" message={notice} showIcon />}
    <Space wrap><Button onClick={test} loading={busy}>测试连接</Button><Button type="primary" onClick={save} disabled={busy || Boolean(config.config_error)}>应用配置</Button><Button onClick={reset} disabled={busy}>恢复服务端默认</Button></Space>
  </div>;
}
