export interface AssistantConfig {
  base_url: string; model: string; has_api_key: boolean; auth_mode: 'bearer' | 'none';
  stream: boolean; enabled: boolean; context_tokens: number; max_output_tokens: number;
  timeout_seconds: number; configured: boolean; source: string; csrf_token: string;
  allow_persist: boolean; config_error: string | null;
}
export interface ConnectionInput {
  base_url: string; model: string; auth_mode: 'bearer' | 'none'; secret_action: 'keep' | 'replace' | 'clear';
  api_key?: string; stream: boolean; context_tokens: number; max_output_tokens: number;
  timeout_seconds: number; scope: 'session' | 'dotenv';
}
export interface SelectionBlock { id: string; title: string; text: string; code?: string }
export interface HelpEntry {
  id: string; title: string; body: string; topic: string; aliases: string[];
  evidence_level: string; source_refs: string[]; source_sha256?: Record<string, string>;
}
export interface Reference { id: string; title: string; topic: string; source_refs: string[]; evidence_level: string }
export interface ChatMeta {
  request_id: string; conversation_id: string; captured_at: string; build_id: string | null;
  references: Reference[]; evidence: Record<string, unknown>; history_trimmed: boolean;
}
export interface ChatDone { finish_reason: string; usage: Record<string, number> | null; request_id: string }
export interface ChatInput {
  question: string; selection: SelectionBlock[]; build_id: string | null; conversation_id: string | null;
  request_id?: string;
  captured_at: string; include: { business_summary: boolean; recent_logs: boolean };
  page_summary: Record<string, unknown>;
}
export interface SavedTurn {
  messages: Array<{role: 'user' | 'assistant'; content: string}>; references: Reference[];
  captured_at: string; build_id: string | null;
}

let csrf = '';
let handshake: Promise<AssistantConfig> | null = null;

async function parseResponse<T>(response: Response): Promise<T> {
  const data = await response.json();
  if (!response.ok) {
    const message = typeof data.error === 'string' ? data.error : `Assistant request failed (${response.status})`;
    throw new Error(message === 'The assistant request contains unknown fields' ? 'The page and local service versions differ; restart TurboForge and refresh the page.' : message);
  }
  return data as T;
}

async function config(): Promise<AssistantConfig> {
  if (!handshake) {
    handshake = fetch('/api/assistant/config', {cache: 'no-store'}).then(parseResponse<AssistantConfig>)
      .then(value => { csrf = value.csrf_token; return value; }).finally(() => { handshake = null; });
  }
  return handshake;
}

async function request<T>(path: string, method = 'GET', body?: unknown): Promise<T> {
  if (method !== 'GET' && !csrf) await config();
  return parseResponse<T>(await fetch('/api/assistant' + path, {
    method, cache: 'no-store', headers: {'Content-Type': 'application/json', 'X-Assistant-CSRF': csrf},
    ...(body === undefined ? {} : {body: JSON.stringify(body)}),
  }));
}

export const assistantApi = {
  config,
  storage: () => request<StorageInventory>('/storage'),
  cleanStorage: (ids: string[]) => request<StorageInventory>('/storage', 'DELETE', {ids}),
  preview: (input: ChatInput) => request<Record<string, unknown>>('/context/preview', 'POST', input),
  test: (input: ConnectionInput) => request<{ok: boolean; latency_ms: number; tested_config_id: string}>('/config/test', 'POST', input),
  save: (input: ConnectionInput | {tested_config_id: string; scope: 'session' | 'dotenv'}) => request<AssistantConfig & {notice?: string}>('/config', 'PUT', input),
  reset: () => request<AssistantConfig>('/config/session', 'DELETE'),
  help: (q = '', topic = '') => request<{entries: HelpEntry[]; total: number}>(`/help?q=${encodeURIComponent(q)}&topic=${encodeURIComponent(topic)}`),
  entry: (id: string) => request<HelpEntry>('/help/' + encodeURIComponent(id)),
  cancel: (id: string) => request('/chat/' + encodeURIComponent(id) + '/cancel', 'POST', {}),
  history: (id: string) => request<{conversation_id: string; build_id: string | null; model: string; turns: SavedTurn[]}>('/conversations/' + encodeURIComponent(id)),
  remove: (id: string) => request('/conversations/' + encodeURIComponent(id), 'DELETE'),
};

export interface StorageInventory {
  items: Array<{id: string; status: string; bytes: number; removable: boolean}>;
  free_bytes: number; note: string;
}

export async function chat(input: ChatInput, signal: AbortSignal, callbacks: {
  meta: (value: ChatMeta) => void; delta: (text: string) => void; done: (value: ChatDone) => void;
}): Promise<void> {
  if (!csrf) await config();
  const response = await fetch('/api/assistant/chat', {method: 'POST', signal,
    headers: {'Content-Type': 'application/json', 'X-Assistant-CSRF': csrf}, body: JSON.stringify(input)});
  if (!response.ok) { await parseResponse(response); return; }
  if (!response.headers.get('Content-Type')?.includes('text/event-stream')) {
    const result = await parseResponse<ChatMeta & ChatDone & {text: string}>(response);
    callbacks.meta(result); callbacks.delta(result.text); callbacks.done(result); return;
  }
  if (!response.body) throw new Error('The browser did not provide a response stream');
  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8', {fatal: true});
  let buffer = ''; let finished = false;
  const frame = (value: string) => {
    const lines = value.split(/\r?\n/);
    const event = lines.find(line => line.startsWith('event:'))?.slice(6).trim();
    const raw = lines.filter(line => line.startsWith('data:')).map(line => line.slice(5).replace(/^ /, '')).join('\n');
    if (!raw) return;
    const data = JSON.parse(raw);
    if (event === 'meta') callbacks.meta(data as ChatMeta);
    else if (event === 'delta' && typeof data.text === 'string') callbacks.delta(data.text);
    else if (event === 'done') { callbacks.done(data as ChatDone); finished = true; }
    else if (event === 'error') throw new Error(typeof data.error === 'string' ? data.error : 'The answer did not complete');
  };
  try {
    while (true) {
      const part = await reader.read();
      buffer += decoder.decode(part.value, {stream: !part.done});
      if (buffer.length > 2 * 1024 * 1024) throw new Error('Assistant events exceeded the size limit');
      let match: RegExpExecArray | null;
      while ((match = /\r?\n\r?\n/.exec(buffer))) {
        frame(buffer.slice(0, match.index));
        buffer = buffer.slice(match.index + match[0].length);
      }
      if (part.done) break;
    }
    if (!finished || buffer.trim()) throw new Error('The answer stream ended unexpectedly; retry manually');
  } finally {
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}
