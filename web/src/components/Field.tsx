import { Button, Form } from 'antd';
import { InfoCircleOutlined } from '@ant-design/icons';
import { useEffect } from 'react';
import type { ReactNode } from 'react';

interface FieldProps {
  label: string;
  required?: boolean;
  tooltip?: string;
  extra?: ReactNode;
  children: ReactNode;
}

// Every field states plainly whether it is required or optional, with a help
// tooltip, so the user never has to guess what a control expects.
export default function Field({ label, required = false, tooltip, extra, children }: FieldProps) {
  useEffect(() => {
    const root = Array.from(document.querySelectorAll<HTMLElement>('[data-ai-field]')).find(element => element.dataset.aiField === label);
    root?.querySelectorAll('input,textarea').forEach(input => {
      if (!input.closest('[data-ai-ignore]') && !input.matches('[type="password"],[type="hidden"]')) input.setAttribute('data-ai-value', 'true');
    });
  });
  const semanticIds: Record<string, string> = {'Model':'workflow.model','Input model':'workflow.model','Quantization plan':'workflow.quantization',
    'Quantization target':'workflow.quantization','Speculative decoding':'workflow.speculation'};
  return (
    <Form.Item
      className="field"
      data-ai-id={semanticIds[label] ?? `field.${label}`}
      data-ai-field={label}
      data-ai-title={label}
      required={required}
      tooltip={tooltip ? { title: tooltip, icon: <InfoCircleOutlined /> } : undefined}
      extra={extra ? <span className="field-extra">{extra}</span> : undefined}
      label={
        <span className="field-label">
          {label}
          <Button type="text" size="small" aria-label={`Explain ${label}`} className="ai-field-help" data-ai-ignore
            onClick={event => {event.preventDefault(); event.stopPropagation(); window.dispatchEvent(new CustomEvent('turboforge:ask', {detail:{id:`field.${label}`,title:label,text:`${label}: ${tooltip ?? 'Explain the meaning, impact and scope of this field.'}`}}));}}>?</Button>
          <span className={required ? 'req is-required' : 'req is-optional'}>
            {required ? 'Required' : 'Optional'}
          </span>
        </span>
      }
    >
      {children}
    </Form.Item>
  );
}
