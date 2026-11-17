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
      setNotice(`Text call test succeeded, ${result.latency_ms} ms; the config is kept on the backend for 5 minutes and can be applied directly.`);
    } catch (err) { setError((err as Error).message); }
    finally { setBusy(false); }
  };
  const save = async () => {
    setBusy(true); setError('');
    const value = tested ? {tested_config_id: tested, scope} : input(); setKey('');
    try {
      const result = await assistantApi.save(value); setNotice(result.notice ?? 'Configuration applied');
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
    <p className="muted">Effective source: {({default:'Not configured',environment:'Process environment variables',dotenv:'Local .env',session:'This session'} as Record<string,string>)[config.source] ?? config.source}. Existing keys only show their configuration status.</p>
    {config.config_error && <Alert type="error" message={config.config_error} showIcon />}
    <label htmlFor="ai-base">Base URL</label>
    <Input id="ai-base" value={base} placeholder="https://llm.example.com/v1" disabled={busy} onChange={e => {
      changed(); setBase(e.target.value);
      try { if (new URL(e.target.value).origin !== new URL(config.base_url).origin) {setAction('replace'); setKey('');} } catch { setAction('replace'); }
    }} />
    <small>Enter the API base URL, not /chat/completions. Use HTTPS for remote services.</small>
    <label htmlFor="ai-model">Model name (the actual API model ID)</label>
    <Input id="ai-model" value={model} disabled={busy} onChange={e => {changed(); setModel(e.target.value);}} />
    <Checkbox checked={local} disabled={busy} onChange={e => {changed(); setLocal(e.target.checked); setKey(''); setAction(e.target.checked ? 'clear' : 'replace');}}>Local services need no API key (the admin must allow this origin)</Checkbox>
    {!local && <>
      <label>API Key · {config.has_api_key ? 'configured on the backend' : 'not configured yet'}</label>
      <Select aria-label="Key action" value={action} disabled={busy} options={[
        {value:'keep',label:'Keep current key',disabled:!config.has_api_key}, {value:'replace',label:'Enter/replace key'}, {value:'clear',label:'Clear key'},
      ]} onChange={value => {changed(); setAction(value); setKey('');}} />
      {action === 'replace' && <Input.Password aria-label="API Key" value={key} autoComplete="off" disabled={busy}
        placeholder={tested ? 'The tested key was cleared; you can apply directly' : 'Sent only to the local backend, never into chat content'} onChange={e => {changed(); setKey(e.target.value);}} />}
    </>}
    <div className="ai-setting-row"><span>Streaming answers</span><Switch checked={stream} disabled={busy} onChange={value => {changed(); setStream(value);}} /></div>
    <label>Service context limit (tokens)</label><InputNumber aria-label="Service context limit" min={256} max={1048576} value={context} disabled={busy} onChange={v => {if (v !== null) {changed(); setContext(v);}}} />
    <label>Output limit (tokens)</label><InputNumber aria-label="Output limit" min={16} max={32768} value={output} disabled={busy} onChange={v => {if (v !== null) {changed(); setOutput(v);}}} />
    <label>Total timeout (seconds)</label><InputNumber aria-label="Total timeout" min={5} max={300} value={timeout} disabled={busy} onChange={v => {if (v !== null) {changed(); setTimeoutValue(v);}}} />
    <label>Save method</label><Select value={scope} disabled={busy} onChange={setScope} options={[
      {value:'session',label:'This session only (recommended)'}, {value:'dotenv',label:'Local .env global default',disabled:!config.allow_persist},
    ]} />
    <small>Testing incurs one short text-call charge. The key input is cleared immediately after submission. Session config is lost on expiry/restart; .env is a plaintext file and web persistence is disabled by default.</small>
    {error && <Alert type="error" message={error} showIcon />}
    {notice && <Alert type="success" message={notice} showIcon />}
    <Space wrap><Button onClick={test} loading={busy}>Test connection</Button><Button type="primary" onClick={save} disabled={busy || Boolean(config.config_error)}>Apply configuration</Button><Button onClick={reset} disabled={busy}>Restore server defaults</Button></Space>
  </div>;
}
