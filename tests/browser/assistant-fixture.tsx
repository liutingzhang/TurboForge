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
  const [selectionChecks,setSelectionChecks] = React.useState('未运行');
  const checkSelections = () => {
    const checks: string[] = [];
    const check = (ok: boolean, name: string) => {if (!ok) throw new Error(name);checks.push(name);};
    try {
      const notice = document.querySelector('[data-testid="selection-notice"] .notice');
      const body = notice?.querySelector('.notice-body');
      const paragraph = document.querySelector('[data-testid="selection-paragraph"]');
      if (!notice || !body || !paragraph) throw new Error('夹具缺失');
      notice.scrollIntoView({block:'center'});
      const block = blockFrom(body);
      check(!!block?.text.includes('此次不验证量化质量') && !block.text.includes('不能混入的相邻说明'), '提示只采集自身标题和正文');
      check(block?.title === '此次不验证量化质量', '提示使用自身标题');
      const title = notice.querySelector('.notice-title');
      check(!!title && blockFrom(title)?.id === block?.id, '点击提示标题或正文选择同一卡片');
      const area = areaSelection(notice.getBoundingClientRect());
      check(area.length === 1 && area[0].text === block?.text, '框选与点选的提示范围一致');
      paragraph.scrollIntoView({block:'center'});
      const explanation = blockFrom(paragraph);
      check(explanation?.text === '不能混入的相邻说明。', '独立说明段落可单独选择');
      check(explanation?.id !== block?.id, '同一面板中的不同内容拥有独立标识');
      const panel = paragraph.closest('.panel');
      const range = document.createRange(); range.selectNodeContents(paragraph);
      check(!!panel && extractText(panel,undefined,range) === explanation?.text, '精确文字范围不混入其他输入值');
      const fact = document.querySelector('[data-testid="selection-fact"] .fact-value');
      fact?.scrollIntoView({block:'center'});
      check(!!fact && blockFrom(fact)?.text === '空闲显存 10.18 GiB', '信息数值带上自身标签和单位');
      setSelectionChecks(`通过 ${checks.length} 项：${checks.join('；')}`);
    } catch (error) {setSelectionChecks(`失败：${(error as Error).message}`);}
  };
  return <ConfigProvider theme={forgeTheme}><App><AssistantProvider context={{buildId:null,modelIdentity:'browser-fixture',
    summary:{target:'auto',stage:'fixture_only'}}}>
    <main style={{padding:32,maxWidth:900}}>
      <h1>助手 DOM 边界验收夹具</h1><p>仅用于验证页面采集，不代表真实构建状态。</p>
      <button data-ai-ignore onClick={() => {
        const textarea = document.querySelector<HTMLTextAreaElement>('textarea[aria-label="向 AI 提问"]');
        textarea?.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',code:'Enter',bubbles:true,isComposing:true}));
      }}>触发组合输入 Enter（夹具）</button>
      <Panel title="参数与隐私">
        <div data-testid="selection-notice"><Notice tone="warn" title="此次不验证量化质量">将验证 GPU 可运行，并按选择测速；质量保持未验证状态。若要比较量化损失，请开启质量门禁。</Notice></div>
        <p data-testid="selection-paragraph">不能混入的相邻说明。</p>
        <div data-testid="selection-fact"><Fact label="空闲显存" value="10.18 GiB" /></div>
        <button data-ai-ignore onClick={checkSelections}>运行选区回归</button>
        <output aria-label="选区回归结果">{selectionChecks}</output>
        <Field label="服务上下文" tooltip="修改只影响此夹具数值">
          <InputNumber aria-label="夹具服务上下文" min={1} value={value} onChange={v => {if(v!==null)setValue(v);}} />
        </Field>
        <div data-ai-ignore><label>排除区域<Input aria-label="排除字段" defaultValue="EXCLUDED-FIXTURE-CONTENT" /></label></div>
        <label>密码字段<Input.Password aria-label="夹具密码" defaultValue="PASSWORD-FIXTURE-CONTENT" /></label>
        <div hidden>HIDDEN-FIXTURE-CONTENT</div>
      </Panel>
      <section data-ai-id="report.performance" data-ai-title="性能报告（夹具）" style={{marginTop:24}}>
        <h2>性能报告</h2>
        <Table pagination={false} rowKey="name" dataSource={[{name:'解码速度',value:'52.9 tok/s',status:'实测'},
          {name:'预填充速度',value:'未实测',status:'未实测'}]} columns={[
          {title:'指标',dataIndex:'name'}, {title:'数值与单位',dataIndex:'value'}, {title:'证据状态',dataIndex:'status'}]} />
      </section>
      <section data-ai-id="build.logs" data-ai-title="已打开日志（夹具）" style={{marginTop:24}}>
        <h2>已打开日志</h2><pre className="console">{'phase=validation\nmodel=D:/private-fixture/model\napi_key=FIXTURE-SECRET\nresult=out of memory\nnext=check GPU memory'}</pre>
      </section>
      <Diag d={{code:'vram-insufficient',message:'夹具：预计显存不足',severity:'warning',alternatives:['减少上下文']}} />
    </main>
  </AssistantProvider></App></ConfigProvider>;
}
const root = document.getElementById('root');
if (root) createRoot(root).render(<Fixture />);
