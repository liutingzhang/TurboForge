import { useCallback, useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { Alert, App, Button, Drawer, Input, Space, Tabs, Tag } from 'antd';
import type { TextAreaRef } from 'antd/es/input/TextArea';
import { CloseOutlined, MessageOutlined, SelectOutlined, SettingOutlined } from '@ant-design/icons';
import { assistantApi, chat } from '../../api/assistant';
import type { AssistantConfig, ChatInput, HelpEntry, SelectionBlock } from '../../api/assistant';
import { locateTopic, readable, redact, textSelection, validateSelections } from '../../utils/assistantContext';
import { usePageSelection } from '../../hooks/usePageSelection';
import AssistantSettings from './AssistantSettings';
import HelpCenter from './HelpCenter';
import ContextPreview from './ContextPreview';
import Conversation from './Conversation';
import type { Turn } from './Conversation';

export interface AssistantPageContext { buildId: string | null; modelIdentity: string; summary: Record<string, unknown> }

export default function AssistantDrawer({context, children}: {context: AssistantPageContext; children: ReactNode}) {
  const { message } = App.useApp();
  const [open, setOpen] = useState(false); const [settings, setSettings] = useState(false);
  const [tab, setTab] = useState('chat'); const [helpId, setHelpId] = useState('');
  const [config, setConfig] = useState<AssistantConfig | null>(null); const [configVersion, setConfigVersion] = useState(0);
  const [error, setError] = useState(''); const [notice, setNotice] = useState('');
  const [blocks, setBlocks] = useState<SelectionBlock[]>([]); const [question, setQuestion] = useState('');
  const [capturedAt, setCapturedAt] = useState(''); const [selecting, setSelecting] = useState(false);
  const [includeSummary, setIncludeSummary] = useState(true); const [includeLogs, setIncludeLogs] = useState(false);
  const [snapshot, setSnapshot] = useState<Record<string, unknown> | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]); const [busy, setBusy] = useState(false);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const requestId = useRef<string | null>(null); const controller = useRef<AbortController | null>(null);
  const launch = useRef<HTMLButtonElement>(null); const questionInput = useRef<TextAreaRef>(null);
  const mounted = useRef(true); const savedBoundary = useRef('');
  const boundaryRef = useRef('');
  const currentBoundary = JSON.stringify([context.buildId, context.modelIdentity]);
  boundaryRef.current = currentBoundary;
  const summaryText = JSON.stringify(context.summary); const previousSummary = useRef(summaryText);

  const refreshConfig = useCallback(async () => {
    const value = await assistantApi.config();
    if (mounted.current) {setConfig(value); setConfigVersion(v => v + 1);}
  }, []);
  useEffect(() => {
    mounted.current = true;
    const restoredBoundary = boundaryRef.current;
    refreshConfig().catch(err => setError((err as Error).message));
    const saved = sessionStorage.getItem('turboforge.assistant.conversation');
    if (saved) assistantApi.history(saved).then(history => {
      if (!mounted.current || restoredBoundary !== boundaryRef.current) return;
      setConversationId(history.conversation_id);
      setTurns(history.turns.map((turn,index) => {
        let prompt = turn.messages.find(m => m.role === 'user')?.content ?? '';
        try {prompt = (JSON.parse(prompt) as {question: string}).question;} catch {prompt = 'Past question';}
        return {id:`restored-${index}`,question:prompt,answer:turn.messages.find(m => m.role === 'assistant')?.content ?? '',
          references:turn.references,capturedAt:turn.captured_at,buildId:turn.build_id};
      }));
      if (history.build_id !== context.buildId) {setNotice('Restored a conversation from an older task; new questions will build a new context.'); setConversationId(null);}
    }).catch(() => {sessionStorage.removeItem('turboforge.assistant.conversation');});
    return () => {mounted.current = false; controller.current?.abort();};
  }, [refreshConfig]); // Bootstrap once; task changes are handled separately below.

  const stop = useCallback(() => {
    if (requestId.current) assistantApi.cancel(requestId.current).catch(() => undefined);
    controller.current?.abort();
    setNotice('Stopped reading the answer; the upstream may still bill.');
  }, []);
  useEffect(() => {
    if (savedBoundary.current && savedBoundary.current !== currentBoundary) {
      if (controller.current) stop();
      setConversationId(null); setBlocks([]); setSnapshot(null); setIncludeLogs(false); setCapturedAt('');
      sessionStorage.removeItem('turboforge.assistant.conversation');
      setNotice('The model or task changed. Past answers belong to the old snapshot; new questions use the new context.');
    }
    savedBoundary.current = currentBoundary;
  }, [currentBoundary, stop]);
  useEffect(() => {
    if (previousSummary.current !== summaryText) {setSnapshot(null); if (turns.length) setNotice('The page state changed; past answers are based on the earlier snapshot.');}
    previousSummary.current = summaryText;
  }, [summaryText, turns.length]);

  const addBlocks = useCallback((selected: SelectionBlock[]) => {
    const next = [...blocks, ...selected.filter(item => !blocks.some(old => old.id === item.id && old.text === item.text))];
    validateSelections(next);
    setBlocks(next); setCapturedAt(new Date().toISOString()); setSnapshot(null); setSelecting(false); setOpen(true); setSettings(false); setTab('chat');
    setQuestion(value => value || 'Explain this in simple terms and tell me the next step.');
  }, [blocks]);
  const selectionError = useCallback((value: string) => {message.info(value);}, [message]);
  const cancelSelection = useCallback(() => {setSelecting(false); setOpen(true);}, []);
  const rectangle = usePageSelection(selecting, addBlocks, selectionError, cancelSelection);
  useEffect(() => {
    // rc-trigger closes Portal dropdowns on outside pointerdown. Capture before
    // its listener so an already opened list can be selected without changing it.
    const preserveDropdown = (event: PointerEvent) => {
      if (!(event.target instanceof Element) || !event.target.closest('.ai-launcher')) return;
      if (!Array.from(document.querySelectorAll('.ant-select-dropdown,.ant-dropdown')).some(readable)) return;
      event.preventDefault(); event.stopImmediatePropagation();
      setSelecting(true); setOpen(false);
    };
    window.addEventListener('pointerdown', preserveDropdown, true);
    return () => window.removeEventListener('pointerdown', preserveDropdown, true);
  }, []);
  useEffect(() => {
    const ask = (event: Event) => {
      const value = (event as CustomEvent<SelectionBlock>).detail;
      if (value) {try {
        addBlocks([value]);
        if (value.code) {setHelpId(value.code); setTab('help');}
      } catch (err) {message.warning((err as Error).message);}}
    };
    window.addEventListener('turboforge:ask', ask);
    return () => window.removeEventListener('turboforge:ask', ask);
  }, [addBlocks, message]);

  const selectPage = () => {
    try {const selected = textSelection(); if (selected) {addBlocks([selected]); return;}}
    catch (err) {message.warning((err as Error).message); return;}
    setOpen(false); setSelecting(true);
  };
  const input = (): ChatInput => ({question: redact(question), selection: blocks.map(b => ({...b,text:redact(b.text)})),
    conversation_id:conversationId,build_id:context.buildId,captured_at:capturedAt || new Date().toISOString(),
    include:{business_summary:includeSummary,recent_logs:includeLogs}, page_summary:includeSummary ? context.summary : {}});
  const updateTurn = (id: string, patch: Partial<Turn> | ((turn: Turn) => Partial<Turn>)) => setTurns(all => all.map(t => t.id === id ? {...t,...(typeof patch === 'function' ? patch(t) : patch)} : t));
  const send = async () => {
    if (busy || !question.trim()) return;
    if (!config?.configured) {setSettings(true); return;}
    try {validateSelections(blocks);} catch (err) {setError((err as Error).message); return;}
    const value = input(); const id = crypto.randomUUID(); const pendingRequestId = crypto.randomUUID();
    const sentBoundary = boundaryRef.current;
    setTurns(all => [...all,{id,question:value.question,answer:'',references:[],buildId:context.buildId}]);
    setBusy(true); setError(''); setNotice('');
    const abort = new AbortController(); controller.current = abort; requestId.current = pendingRequestId;
    try {
      await chat({...value, request_id:pendingRequestId}, abort.signal, {
        meta: metadata => {
          if (abort.signal.aborted || sentBoundary !== boundaryRef.current) {
            void assistantApi.cancel(metadata.request_id).catch(() => undefined);
            return;
          }
          requestId.current = metadata.request_id; setConversationId(metadata.conversation_id);
          sessionStorage.setItem('turboforge.assistant.conversation', metadata.conversation_id);
          updateTurn(id,{references:metadata.references,capturedAt:metadata.captured_at});
          if (metadata.history_trimmed) setNotice('To fit the model window, not all past conversation was included this time.');
        },
        delta: text => updateTurn(id,turn => ({answer:turn.answer + text})),
        done: data => updateTurn(id,{done:data}),
      });
      if (sentBoundary === boundaryRef.current) setQuestion('');
    } catch (err) {
      updateTurn(id,{error:abort.signal.aborted ? 'Stopped; the answer is incomplete' : (err as Error).message});
    } finally {
      controller.current = null; requestId.current = null; setBusy(false);
    }
  };
  const preview = async () => {
    setError('');
    try {setSnapshot(await assistantApi.preview({...input(),question:question.trim() || 'Review the current materials'}));}
    catch (err) {setError((err as Error).message);}
  };
  const newConversation = async () => {
    if (busy) return;
    try {if (conversationId) await assistantApi.remove(conversationId);}
    catch (err) {setError((err as Error).message); return;}
    setConversationId(null); sessionStorage.removeItem('turboforge.assistant.conversation');
    setTurns([]); setNotice(''); setError(''); setQuestion(''); setSnapshot(null);
    setBlocks([]); setCapturedAt(''); setIncludeLogs(false); setSettings(false); setTab('chat');
    questionInput.current?.focus();
  };
  const locate = (topic: string) => {
    if (topic === 'connection') {setSettings(true); return;}
    if (!locateTopic(topic)) message.info('That area is not visible in the current step; switch to the related step first.');
  };
  const askHelp = (entry: HelpEntry) => {setTab('chat'); setSettings(false); setQuestion(`Explain this in my current situation: ${entry.title}`);};
  let destination = 'Not configured';
  try {if (config?.base_url) destination = new URL(config.base_url).host;} catch {destination = 'Invalid address';}
  return <>{children}
    {!open && !selecting && <Button ref={launch} type="primary" className="ai-launcher" icon={<MessageOutlined />} aria-label="Open the TurboForge AI assistant" data-ai-ignore onMouseDown={event => {
      const selected = textSelection(); if (selected) {event.preventDefault(); try {addBlocks([selected]);} catch (err) {message.info((err as Error).message);}}
    }} onClick={() => setOpen(true)}>Ask AI</Button>}
    {selecting && <><div className="ai-selection-toolbar" data-ai-ignore data-ai-selection-control>
      <span>Click an area, drag to box-select, Esc to cancel; you can scroll and reselect.</span><Button onClick={cancelSelection}>Cancel selection</Button></div>
      {rectangle && <div className="ai-selection-rectangle" data-ai-ignore style={{left:rectangle.x,top:rectangle.y,width:rectangle.width,height:rectangle.height}} />}</>}
    <Drawer width={480} open={open} mask={false} rootClassName="ai-drawer-root" title={<Space>TurboForge assistant<Tag color={config?.configured ? 'green' : 'default'}>{config?.configured ? 'Configured' : 'Not connected'}</Tag></Space>}
      onClose={() => settings ? setSettings(false) : setOpen(false)} afterOpenChange={visible => {
        // rc-drawer restores its old active element after invoking this callback.
        if (!visible && !selecting) queueMicrotask(() => launch.current?.focus({preventScroll:true}));
      }}
      extra={<Space size={0}><Button type="text" aria-label="Assistant settings" icon={<SettingOutlined />} onClick={() => setSettings(v => !v)} />
        <Button type="text" disabled={busy} onClick={newConversation}>New conversation</Button></Space>}
      styles={{body:{padding:16,display:'flex',flexDirection:'column',gap:12}}}>
      <div data-ai-ignore className="ai-drawer-content">
        {settings ? <><Button type="link" onClick={() => setSettings(false)}>Back to chat</Button>{config ? <AssistantSettings key={configVersion} config={config} onSaved={refreshConfig} /> : <Button onClick={() => refreshConfig().catch(err => setError((err as Error).message))}>Reload configuration</Button>}</> : <>
          <Tabs activeKey={tab} onChange={setTab} items={[{key:'chat',label:'Ask AI'},{key:'help',label:'Help center'}]} />
          {tab === 'help' ? <HelpCenter initialId={helpId} onAsk={askHelp} /> : <>
            {(!config?.configured || config.config_error) && <Alert type="info" showIcon message="After connecting AI you can ask about the page, or browse the help center directly" action={<Button size="small" onClick={() => setSettings(true)}>Configure connection</Button>} />}
            {notice && <Alert type="info" message={notice} closable onClose={() => setNotice('')} />}
            <ContextPreview blocks={blocks} setBlocks={value => {setBlocks(value); setSnapshot(null); if (!value.length) {setCapturedAt(''); setQuestion(current => current === 'Explain this in simple terms and tell me the next step.' ? '' : current);}}} summary={context.summary}
              snapshot={snapshot} includeSummary={includeSummary} setIncludeSummary={value => {setIncludeSummary(value); setSnapshot(null);}}
              includeLogs={includeLogs} setIncludeLogs={value => {setIncludeLogs(value); setSnapshot(null);}} capturedAt={capturedAt} disabled={busy} onRefresh={preview} />
            <Conversation turns={turns} onReference={id => {setHelpId(id); setTab('help');}} onLocate={locate} />
            <div className="ai-composer"><Space wrap><Button icon={<SelectOutlined />} onClick={selectPage} disabled={busy}>Select page</Button>
              <Button size="small" disabled={busy} onClick={() => setQuestion('Why did the current problem happen, and how should I handle it?')}>Why it failed</Button><Button size="small" disabled={busy} onClick={() => setQuestion('Based on the current state, what should I do next?')}>Next step</Button></Space>
              <Input.TextArea ref={questionInput} aria-label="Ask the AI" value={question} maxLength={2000} showCount disabled={busy}
                placeholder="Add your question…" autoSize={{minRows:3,maxRows:7}} onChange={e => setQuestion(e.target.value)}
                onKeyDown={e => {if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {e.preventDefault(); void send();}}} />
              <div className="ai-composer-actions"><small>Will be sent to: {destination}</small>{busy ? <Button danger icon={<CloseOutlined />} onClick={stop}>Stop answer</Button> : <Button type="primary" disabled={!question.trim()} onClick={send}>Send</Button>}</div>
            </div>
          </>}
        </>}
        {error && <Alert type="error" showIcon message={error} closable onClose={() => setError('')} />}
      </div>
    </Drawer>
  </>;
}
