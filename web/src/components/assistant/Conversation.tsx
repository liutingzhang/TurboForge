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
    {!turns.length && <div className="ai-empty"><strong>How can I help?</strong><p>Ask directly, or select content on the page; I will explain it using the project materials.</p></div>}
    {turns.map(turn => <article key={turn.id} className="ai-turn">
      <div className="ai-question"><strong>You</strong><p>{turn.question}</p></div>
      <div className="ai-answer"><strong>TurboForge assistant</strong>
        <p className="ai-prose">{turn.answer || (turn.error ? 'Answer incomplete' : 'Retrieving materials and connecting to the model…')}</p>
        {turn.error && <p className="ai-error" role="alert">{turn.error}</p>}
        {turn.done?.finish_reason === 'length' && <p className="ai-error">The answer hit the output length limit and may be incomplete; narrow the question or adjust the output budget.</p>}
        {turn.references.length > 0 && <div className="ai-references"><small>Sources for this answer</small>{turn.references.map(ref =>
          <span key={ref.id}><Button size="small" onClick={() => onReference(ref.id)}>{ref.title}</Button><Button size="small" type="link" aria-label={`Locate ${ref.title}`} onClick={() => onLocate(ref.topic)}>Locate</Button></span>)}</div>}
        {turn.capturedAt && <small className="muted">Snapshot: {new Date(turn.capturedAt).toLocaleTimeString()} · {turn.buildId ? 'this build' : 'no build-run evidence yet'}</small>}
        {turn.done && <div><Tag>Done</Tag><small className="muted">{turn.done.usage ? `Tokens reported by the service: ${turn.done.usage.total_tokens ?? 'total not provided'}` : 'The service did not report token usage'}</small></div>}
      </div>
    </article>)}
  </div>;
}
