/**
 * Thin client for Annotate → ipred proxy (`/api/ipred/*`).
 * Frontend never talks to the ipred port directly.
 */
import { API_BASE } from '@/config';

export interface IpredSession {
  session_id: string;
  project_id: string;
  current_feature_id?: string | null;
  current_model_id?: string | null;
  current_run_id?: string | null;
}

export interface FeatureSetup {
  id: string;
  name: string;
  kind: 'procedure' | 'weights' | string;
  builtin?: boolean;
  procedure_id?: string;
  params?: Record<string, unknown>;
  encoder_setup_id?: string;
  weights_path?: string;
  weights_format?: string;
  inference?: Record<string, unknown>;
  content_hash?: string;
}

export interface IpredPreprocessResult {
  feature_id: string;
  project_id: string;
  setup_id: string;
  slice_index: number;
  n_channels: number;
  height: number;
  width: number;
  labels: string[];
  cache_hit: boolean;
  blob_dir?: string;
}

async function parseError(res: Response): Promise<Error> {
  const text = await res.text();
  try {
    const j = JSON.parse(text) as { detail?: unknown };
    if (typeof j.detail === 'string') return new Error(j.detail);
    return new Error(text || res.statusText);
  } catch {
    return new Error(text || res.statusText);
  }
}

export async function ipredHealth(): Promise<{ status: string; service?: string }> {
  const res = await fetch(`${API_BASE}/api/ipred/health`);
  if (!res.ok) throw await parseError(res);
  return res.json() as Promise<{ status: string; service?: string }>;
}

export async function openIpredSession(payload: {
  kind: string;
  source: string;
  server_uri?: string | null;
  root?: string | null;
}): Promise<IpredSession> {
  const res = await fetch(`${API_BASE}/api/ipred/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw await parseError(res);
  return res.json() as Promise<IpredSession>;
}

export async function listIpredSetups(): Promise<FeatureSetup[]> {
  const res = await fetch(`${API_BASE}/api/ipred/setups`);
  if (!res.ok) throw await parseError(res);
  const body = (await res.json()) as { setups: FeatureSetup[] };
  return body.setups ?? [];
}

export async function getIpredSetup(setupId: string): Promise<FeatureSetup> {
  const res = await fetch(`${API_BASE}/api/ipred/setups/${encodeURIComponent(setupId)}`);
  if (!res.ok) throw await parseError(res);
  return res.json() as Promise<FeatureSetup>;
}

export async function upsertIpredSetup(payload: {
  name: string;
  kind: string;
  procedure_id?: string;
  params?: Record<string, unknown>;
  encoder_setup_id?: string | null;
  weights_path?: string;
  weights_format?: string;
  inference?: Record<string, unknown>;
  setup_id?: string;
}): Promise<FeatureSetup> {
  const res = await fetch(`${API_BASE}/api/ipred/setups`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw await parseError(res);
  return res.json() as Promise<FeatureSetup>;
}

export async function listIpredTrainers(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/api/ipred/trainers`);
  if (!res.ok) throw await parseError(res);
  const body = (await res.json()) as { trainers: string[] };
  return body.trainers ?? [];
}

export interface FeatureModuleInfo {
  id: string;
  name: string;
  description: string;
  runtime: string;
  ready: boolean;
  accepts_input_from: boolean;
  produces_channels: boolean;
  produces_embedding: boolean;
  params_schema: Record<string, unknown>;
}

export interface CompositionNode {
  id: string;
  module: string;
  params?: Record<string, unknown>;
  input_from?: string;
}

export interface CompositionDoc {
  id: string;
  name: string;
  kind?: string;
  builtin?: boolean;
  nodes: CompositionNode[];
  outputs: string[];
  content_hash?: string;
  preview_labels?: string[];
}

export async function listIpredModules(): Promise<FeatureModuleInfo[]> {
  const res = await fetch(`${API_BASE}/api/ipred/modules`);
  if (!res.ok) throw await parseError(res);
  const body = (await res.json()) as { modules: FeatureModuleInfo[] };
  return body.modules ?? [];
}

export async function listIpredCompositions(): Promise<CompositionDoc[]> {
  const res = await fetch(`${API_BASE}/api/ipred/compositions`);
  if (!res.ok) throw await parseError(res);
  const body = (await res.json()) as { compositions: CompositionDoc[] };
  return body.compositions ?? [];
}

export async function getIpredComposition(id: string): Promise<CompositionDoc> {
  const res = await fetch(
    `${API_BASE}/api/ipred/compositions/${encodeURIComponent(id)}`,
  );
  if (!res.ok) throw await parseError(res);
  return res.json() as Promise<CompositionDoc>;
}

export async function upsertIpredComposition(payload: {
  name: string;
  nodes: CompositionNode[];
  outputs: string[];
  composition_id?: string;
  builtin?: boolean;
}): Promise<CompositionDoc> {
  const res = await fetch(`${API_BASE}/api/ipred/compositions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw await parseError(res);
  return res.json() as Promise<CompositionDoc>;
}

export async function previewIpredComposition(payload: {
  name?: string;
  nodes: CompositionNode[];
  outputs: string[];
}): Promise<{ preview_labels: string[] }> {
  const res = await fetch(`${API_BASE}/api/ipred/compositions/preview`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: payload.name ?? 'preview', ...payload }),
  });
  if (!res.ok) throw await parseError(res);
  return res.json() as Promise<{ preview_labels: string[] }>;
}

export async function ipredPreprocess(payload: {
  session_id: string;
  feature_setup_id?: string;
  composition_id?: string;
  slice_index?: number;
  array_ref?: string;
}): Promise<IpredPreprocessResult> {
  const res = await fetch(`${API_BASE}/api/ipred/preprocess`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw await parseError(res);
  return res.json() as Promise<IpredPreprocessResult>;
}

export function ipredChannelUrl(featureId: string, index: number): string {
  return `${API_BASE}/api/ipred/features/${encodeURIComponent(featureId)}/channels/${index}`;
}

export interface IpredFeatureImportance {
  label: string;
  importance: number;
}

export interface IpredTrainResult {
  model_id: string;
  feature_id: string;
  trainer_id: string;
  class_ids: number[];
  train_accuracy: number;
  n_train: number;
  n_cal: number;
  n_samples: number;
  params: Record<string, unknown>;
  /** Trainer-specific (CatBoost); omit or empty when unavailable. */
  feature_importances?: IpredFeatureImportance[];
}

export interface IpredInferResult {
  run_id: string;
  model_id: string;
  feature_id: string;
  alpha: number;
  class_ids: number[];
  counts: { singleton: number; multi: number; abstain: number };
  q_by_class?: Record<string, number>;
}

export async function ipredTrain(payload: {
  session_id: string;
  shapes: unknown[];
  feature_id?: string | null;
  trainer_id?: string;
  config?: Record<string, unknown>;
}): Promise<IpredTrainResult> {
  const res = await fetch(`${API_BASE}/api/ipred/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw await parseError(res);
  return res.json() as Promise<IpredTrainResult>;
}

export async function ipredInfer(payload: {
  session_id: string;
  model_id?: string | null;
  feature_id?: string | null;
  alpha?: number;
}): Promise<IpredInferResult> {
  const res = await fetch(`${API_BASE}/api/ipred/infer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw await parseError(res);
  return res.json() as Promise<IpredInferResult>;
}

export function ipredRunCommitUrl(runId: string): string {
  return `${API_BASE}/api/ipred/runs/${encodeURIComponent(runId)}/commit.png`;
}

export function ipredRunStatusUrl(runId: string): string {
  return `${API_BASE}/api/ipred/runs/${encodeURIComponent(runId)}/status.png`;
}

export function ipredRunProbaUrl(runId: string, classIndex: number): string {
  return `${API_BASE}/api/ipred/runs/${encodeURIComponent(runId)}/proba/${classIndex}.png`;
}

export interface IpredThresholdClassResult {
  run_id: string;
  class_id: number;
  class_index: number;
  threshold: number;
  width: number;
  height: number;
  n_positive: number;
  label_map_b64: string;
}

export async function ipredThresholdClass(
  runId: string,
  payload: { class_id: number; threshold: number },
): Promise<IpredThresholdClassResult> {
  const res = await fetch(
    `${API_BASE}/api/ipred/runs/${encodeURIComponent(runId)}/threshold-class`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    },
  );
  if (!res.ok) throw await parseError(res);
  return res.json() as Promise<IpredThresholdClassResult>;
}
