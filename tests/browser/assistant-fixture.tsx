// UI integration fixture only: real assistant, Ant Table, fields and report/log DOM.
// No model execution is simulated and no LLM requests are made by this fixture.
import React from 'react';
import { createRoot } from 'react-dom/client';
import { App, ConfigProvider, Input, InputNumber, Table } from 'antd';
import AssistantProvider from '../../web/src/components/assistant/AssistantProvider';
import Field from '../../web/src/components/Field';
import { Diag, Notice, Panel, Fact } from '../../web/src/components/ui';
import { blockFrom, areaSelection, extractText } from '../../web/src/utils/assistantContext';
import { forgeTheme } from '../../web/src/theme';
import '../../web/src/styles.css';

function Fixture() {
  const [value,setValue] = React.useState(1024);
  const [selectionChecks,setSelectionChecks] = React.useState('Not run');
  const checkSelections = () => {
    const checks: string[] = [];
    const check = (ok: boolean, name: string) => {if (!ok) throw new Error(name);checks.push(name);};
    try {
      const notice = document.querySelector('[data-testid="selection-notice"] .notice');
      const body = notice?.querySelector('.notice-body');
      const paragraph = document.querySelector('[data-testid="selection-paragraph"]');
      if (!notice || !body || !paragraph) throw new Error('Fixture element missing');
      notice.scrollIntoView({block:'center'});
      const block = blockFrom(body);
      check(!!block?.text.includes('Quantization quality is not validated this run') && !block.text.includes('Adjacent note that must not be included'), 'The notice captures only its own title and body');
      check(block?.title === 'Quantization quality is not validated this run', 'The notice uses its own title');
      const title = notice.querySelector('.notice-title');
      check(!!title && blockFrom(title)?.id === block?.id, 'Clicking the notice title or body selects the same card');
      const area = areaSelection(notice.getBoundingClientRect());
      check(area.length === 1 && area[0].text === block?.text, 'Box selection and click selection cover the same notice');
      paragraph.scrollIntoView({block:'center'});
      const explanation = blockFrom(paragraph);
      check(explanation?.text === 'Adjacent note that must not be included.', 'A standalone note can be selected on its own');
      check(explanation?.id !== block?.id, 'Different content in the same panel has an independent identity');
      const panel = paragraph.closest('.panel');
      const range = document.createRange(); range.selectNodeContents(paragraph);
      check(!!panel && extractText(panel,undefined,range) === explanation?.text, 'An exact text range does not include other input values');
      const fact = document.querySelector('[data-testid="selection-fact"] .fact-value');
      fact?.scrollIntoView({block:'center'});
      check(!!fact && blockFrom(fact)?.text === 'Free VRAM 10.18 GiB', 'A fact value carries its own label and unit');
      setSelectionChecks(`Passed ${checks.length} checks: ${checks.join('; ')}`);
    } catch (error) {setSelectionChecks(`Failed: ${(error as Error).message}`);}
  };
  return <ConfigProvider theme={forgeTheme}><App><AssistantProvider context={{buildId:null,modelIdentity:'browser-fixture',
    summary:{target:'auto',stage:'fixture_only'}}}>
    <main style={{padding:32,maxWidth:900}}>
      <h1>Assistant DOM boundary acceptance fixture</h1><p>Used only to verify page capture; it does not represent a real build state.</p>
      <button data-ai-ignore onClick={() => {
        const textarea = document.querySelector<HTMLTextAreaElement>('textarea[aria-label="Ask the AI"]');
        textarea?.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',code:'Enter',bubbles:true,isComposing:true}));
      }}>Trigger IME-composed Enter (fixture)</button>
      <Panel title="Parameters and privacy">
        <div data-testid="selection-notice"><Notice tone="warn" title="Quantization quality is not validated this run">The GPU runnability check runs, and speed is measured if selected; quality stays unvalidated. To compare quantization loss, enable the quality gate.</Notice></div>
        <p data-testid="selection-paragraph">Adjacent note that must not be included.</p>
        <div data-testid="selection-fact"><Fact label="Free VRAM" value="10.18 GiB" /></div>
        <button data-ai-ignore onClick={checkSelections}>Run selection regression</button>
        <output aria-label="Selection regression result">{selectionChecks}</output>
        <Field label="Serving context" tooltip="Changes only affect this fixture value">
          <InputNumber aria-label="Fixture serving context" min={1} value={value} onChange={v => {if(v!==null)setValue(v);}} />
        </Field>
        <div data-ai-ignore><label>Excluded area<Input aria-label="Excluded field" defaultValue="EXCLUDED-FIXTURE-CONTENT" /></label></div>
        <label>Password field<Input.Password aria-label="Fixture password" defaultValue="PASSWORD-FIXTURE-CONTENT" /></label>
        <div hidden>HIDDEN-FIXTURE-CONTENT</div>
      </Panel>
      <section data-ai-id="report.performance" data-ai-title="Performance report (fixture)" style={{marginTop:24}}>
        <h2>Performance report</h2>
        <Table pagination={false} rowKey="name" dataSource={[{name:'Decode speed',value:'52.9 tok/s',status:'measured'},
          {name:'Prefill speed',value:'not measured',status:'not measured'}]} columns={[
          {title:'Metric',dataIndex:'name'}, {title:'Value and unit',dataIndex:'value'}, {title:'Evidence status',dataIndex:'status'}]} />
      </section>
      <section data-ai-id="build.logs" data-ai-title="Open logs (fixture)" style={{marginTop:24}}>
        <h2>Open logs</h2><pre className="console">{'phase=validation\nmodel=D:/private-fixture/model\napi_key=FIXTURE-SECRET\nresult=out of memory\nnext=check GPU memory'}</pre>
      </section>
      <Diag d={{code:'vram-insufficient',message:'Fixture: estimated VRAM is insufficient',severity:'warning',alternatives:['Reduce context']}} />
    </main>
  </AssistantProvider></App></ConfigProvider>;
}
const root = document.getElementById('root');
if (root) createRoot(root).render(<Fixture />);
