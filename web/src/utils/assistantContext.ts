import type { SelectionBlock } from '../api/assistant';

export function redact(text: string): string {
  return text.replace(/(bearer\s+)[^\s"'<>]+/gi, '$1<已隐藏>')
    .replace(/((?:api[_-]?key|access[_-]?token|password|secret)\s*[=:]\s*["']?)[^\s,;"']+/gi, '$1<已隐藏>')
    .replace(/\b(?:sk-|hf_)[A-Za-z0-9_-]{12,}/g, '<已隐藏密钥>')
    .replace(/([?&](?:key|token|api_key|access_token)=)[^&\s]+/gi, '$1<已隐藏>')
    .replace(/[A-Za-z]:[\\/][^\n\r"'<>|]*/g, '<本机路径>')
    .replace(/(^|\s)\/(?:home|Users|mnt|tmp|var|root)\/[^\n\r"'<>|]*/g, '$1<本机路径>');
}

const ignored = '[data-ai-ignore],.ai-drawer-root,.ai-launcher,.ai-selection-toolbar,.ai-selection-rectangle,script,style,input[type="password"],input[type="hidden"],iframe';
export function readable(element: Element): boolean {
  if (element.closest(ignored)) return false;
  const style = getComputedStyle(element);
  const rect = element.getBoundingClientRect();
  return style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0' &&
    rect.width > 0 && rect.height > 0 && rect.bottom > 0 && rect.right > 0 && rect.top < innerHeight && rect.left < innerWidth;
}

function intersects(a: DOMRect, b: DOMRect): boolean {
  return a.right >= b.left && a.left <= b.right && a.bottom >= b.top && a.top <= b.bottom;
}

// Prefer the nearest meaningful component. A panel is a layout container, not
// the owner of every notice, paragraph and value displayed inside it.
export function selectionRoot(element: Element): Element {
  const row = element.closest('tr');
  if (row) return row;
  const component = element.closest('[data-ai-id],.ant-form-item,.notice,.diag,.ant-select-dropdown,.fact,.stat,.meter,.chip,.ant-tag');
  if (component && !component.matches('.panel,section')) return component;
  return element.closest('p,h1,h2,h3,h4,h5,h6,li,pre,blockquote,button,dt,dd') ?? element;
}

const selectionIds = new WeakMap<Element, string>();
let nextSelectionId = 0;
function selectionId(root: Element): string {
  const semanticId = root.getAttribute('data-ai-id');
  if (semanticId) return semanticId;
  let id = selectionIds.get(root);
  if (!id) {id = `page.selection.${++nextSelectionId}`; selectionIds.set(root,id);}
  return id;
}

export function extractText(root: Element, area?: DOMRect, selection?: Range): string {
  const parts: string[] = [];
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  let node: Node | null;
  let length = 0;
  while ((node = walker.nextNode())) {
    const parent = node.parentElement;
    if (!parent || !readable(parent) || (selection && !selection.intersectsNode(node))) continue;
    const range = document.createRange(); range.selectNodeContents(node);
    if (area && !Array.from(range.getClientRects()).some(rect => intersects(rect, area))) continue;
    let text = node.textContent ?? '';
    if (selection) {
      const start = selection.startContainer === node ? selection.startOffset : 0;
      const end = selection.endContainer === node ? selection.endOffset : text.length;
      text = text.slice(start, end);
    }
    if (text.trim()) { parts.push(text.trim()); length += text.length; }
    if (length > 24000) { parts.push('（范围过大，请缩小选取）'); break; }
  }
  // Read only explicitly marked ordinary values, never arbitrary input.value.
  for (const input of root.querySelectorAll<HTMLInputElement | HTMLTextAreaElement>('input[data-ai-value],textarea[data-ai-value]')) {
    if (selection && !selection.intersectsNode(input)) continue;
    if (readable(input) && (!area || intersects(input.getBoundingClientRect(), area))) parts.push(input.value);
  }
  const headers = root.closest('tr')?.closest('table')?.querySelector('thead');
  if (headers && !root.closest('thead') && !root.contains(headers))
    parts.unshift(Array.from(headers.querySelectorAll('th')).map(cell => cell.textContent ?? '').join(' | '));
  // Keep text-node boundaries until redaction so a path cannot hide unrelated
  // following labels after whitespace has been flattened.
  return parts.map(part => redact(part)).join(' ').replace(/\s+/g, ' ').trim();
}

export function blockFrom(element: Element, area?: DOMRect, range?: Range): SelectionBlock | null {
  if (!readable(element)) return null;
  const root = selectionRoot(element);
  if (!readable(root)) return null;
  const text = extractText(root, area, range);
  if (!text) return null;
  const code = root.getAttribute('data-ai-code');
  return {id: selectionId(root), title: root.getAttribute('data-ai-title') ?? root.querySelector('.notice-title,label,h1,h2,h3,.panel-title,.fact-label,.stat-label,.meter-head')?.textContent?.trim() ?? (root.matches('p,li') ? '说明文字' : '页面选区'),
    text, ...(code ? {code} : {})};
}

export function textSelection(): SelectionBlock | null {
  const selected = window.getSelection();
  if (!selected?.rangeCount || selected.isCollapsed) return null;
  const range = selected.getRangeAt(0).cloneRange();
  const node = range.commonAncestorContainer;
  const root = node instanceof Element ? node : node.parentElement;
  if (!root || root.closest(ignored)) return null;
  return blockFrom(root, undefined, range);
}

export function areaSelection(area: DOMRect): SelectionBlock[] {
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  const roots = new Map<Element, SelectionBlock>();
  let node: Node | null;
  while ((node = walker.nextNode())) {
    const parent = node.parentElement;
    if (!parent || !readable(parent) || !node.textContent?.trim()) continue;
    const range = document.createRange(); range.selectNodeContents(node);
    if (!Array.from(range.getClientRects()).some(rect => intersects(rect, area))) continue;
    const root = selectionRoot(parent);
    if (roots.has(root)) continue;
    const block = blockFrom(root, area);
    if (block) roots.set(root, block);
    if (roots.size > 5) throw new Error('框选超过 5 个区域，请缩小范围');
  }
  return Array.from(roots.values());
}

export function validateSelections(blocks: SelectionBlock[]): void {
  if (blocks.length > 5 || blocks.reduce((sum, b) => sum + b.text.length, 0) > 6000)
    throw new Error('最多选取 5 个区域、合计 6000 字符，请缩小范围');
}

export const navigationTargets: Record<string, string> = {
  start: 'workflow.model', input: 'workflow.model', quantization: 'workflow.quantization',
  resources: 'plan.resources', calibration: 'workflow.quantization', speculation: 'workflow.speculation',
  errors: 'build.status', quality: 'quality.gate', performance: 'report.performance',
  serving: 'build.server', connection: 'assistant.settings', maintenance: 'build.status',
};

export function locateTopic(topic: string): boolean {
  const target = navigationTargets[topic];
  const element = Array.from(document.querySelectorAll<HTMLElement>('[data-ai-id]')).find(item => item.dataset.aiId === target);
  if (!element) return false;
  element.scrollIntoView({behavior: 'smooth', block: 'center'});
  element.classList.add('ai-location-highlight');
  setTimeout(() => element.classList.remove('ai-location-highlight'), 2400);
  return true;
}
