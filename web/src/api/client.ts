import type {
  BuildStatus,
  PrecheckResult,
  ForgeConfig,
  InspectResponse,
  Meta,
  PlanResponse,
  Report,
  ModelServer,
} from './types';

// Same origin: Flask serves both this frontend and /api, so there is no base URL
// to configure and no CORS. In dev, Vite proxies /api to the Flask server.
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const message = body && body.error ? body.error : `HTTP ${res.status}`;
    throw new Error(message);
  }
  return body as T;
}

export const api = {
  precheck: (model: string, original_model?: string) => request<PrecheckResult>('/api/precheck', {method:'POST', body:JSON.stringify({model, ...(original_model?.trim() ? {original_model:original_model.trim()} : {})})}),
  servers: (method:'GET'|'DELETE' = 'GET') => request<{servers:Array<{build_id:string;status:string;base_url:string}>}>('/api/servers', {method}),
  meta: () => request<Meta>('/api/meta'),
  autoBuild: (model: string, options: Record<string, unknown>) => request<{build_id:string}>('/api/auto-build', {
    method:'POST', body:JSON.stringify({model, options}),
  }),
  pickPath: (kind: 'file' | 'directory') => request<{ path: string | null }>('/api/pick-path', {
    method: 'POST', body: JSON.stringify({ kind }),
  }),
  server: (id: string, method = 'GET') => request<ModelServer>(`/api/build/${id}/server`, { method }),
  inspect: (input_path: string, metadata_only = false) =>
    request<InspectResponse>('/api/inspect', {
      method: 'POST',
      body: JSON.stringify({ input_path, metadata_only }),
    }),
  plan: (config: ForgeConfig) =>
    request<PlanResponse>('/api/plan', {
      method: 'POST',
      body: JSON.stringify(config),
    }),
  build: (config: ForgeConfig) =>
    request<{ build_id: string }>('/api/build', {
      method: 'POST',
      body: JSON.stringify(config),
    }),
  status: (id: string) => request<BuildStatus>(`/api/build/${id}`),
  log: (id: string) => request<{ log: string }>(`/api/build/${id}/log`),
  report: async (id: string, name: 'performance' | 'quality'): Promise<Report | null> => {
    const body = await request<{ report: Report | null }>(
      `/api/build/${id}/report?name=${name}`,
    );
    return body.report;
  },
};
