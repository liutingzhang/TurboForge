import { Button, Tag } from 'antd';
import type { ChatDone, Reference } from '../../api/assistant';

export interface Turn {
  id: string; question: string; answer: string; error?: string; references: Reference[];
  capturedAt?: string; buildId: string | null; done?: ChatDone;
}

export default function Conversation({turns, onReference, onLocate}: {
  turns: Turn[]; onReference: (id: string) => void; onLocate: (topic: string) => void;
}) {
  return <div className="ai-conversation" aria-live="polite" aria-relevant="additions text">
    {!turns.length && <div className="ai-empty"><strong>有什么可以帮你？</strong><p>直接提问，或选取页面上的内容，我会结合项目资料解释。</p></div>}
    {turns.map(turn => <article key={turn.id} className="ai-turn">
      <div className="ai-question"><strong>你</strong><p>{turn.question}</p></div>
      <div className="ai-answer"><strong>TurboForge 助手</strong>
        <p className="ai-prose">{turn.answer || (turn.error ? '回答未完成' : '正在检索资料并连接模型…')}</p>
        {turn.error && <p className="ai-error" role="alert">{turn.error}</p>}
        {turn.done?.finish_reason === 'length' && <p className="ai-error">回答达到输出长度上限，可能未完整；可缩小问题或调整输出预算。</p>}
        {turn.references.length > 0 && <div className="ai-references"><small>本次检索依据</small>{turn.references.map(ref =>
          <span key={ref.id}><Button size="small" onClick={() => onReference(ref.id)}>{ref.title}</Button><Button size="small" type="link" aria-label={`定位 ${ref.title}`} onClick={() => onLocate(ref.topic)}>定位</Button></span>)}</div>}
        {turn.capturedAt && <small className="muted">快照：{new Date(turn.capturedAt).toLocaleTimeString()} · {turn.buildId ? '本次构建' : '尚无构建运行证据'}</small>}
        {turn.done && <div><Tag>已完成</Tag><small className="muted">{turn.done.usage ? `服务返回 token：${turn.done.usage.total_tokens ?? '未提供总数'}` : '服务未返回 token 用量'}</small></div>}
      </div>
    </article>)}
  </div>;
}
