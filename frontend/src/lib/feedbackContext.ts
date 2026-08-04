/**
 * feedbackContext — builds the application-context blob prefilled into the
 * "Bugs & Feature Requests" Google Form, and the prefilled form URL.
 *
 * Pure (reads store snapshots via getState + globals), so it can be called from a
 * click handler outside React. This is sent to an external service (Google Forms)
 * once the user confirms in FeedbackConsentModal — it is NOT purely anonymous: the
 * `Dataset` line includes the currently open sample's source path, which can carry
 * usernames, study identifiers, or project names depending on how it's organized.
 */
import { APP_VERSION, GIT_COMMIT } from '@/config';
import { useToolStore } from '@/stores/toolStore';
import { useDatasetStore } from '@/stores/datasetStore';
import { useSettingsStore } from '@/stores/settingsStore';
import { samClient, webgpuAvailable } from '@/lib/sam/samClient';

/** GPU acceleration status for the in-browser SAM engine. */
function gpuStatus(): string {
  const backend = samClient.getBackend(); // 'webgpu' | 'wasm' | null (null until loaded)
  if (backend === 'webgpu') return 'WebGPU (active)';
  if (backend === 'wasm') return 'CPU / WASM (active)';
  return webgpuAvailable() ? 'WebGPU available (SAM not yet loaded)' : 'no WebGPU (CPU only)';
}

/** Human-readable app context for a bug report (multi-line). */
export function buildFeedbackContext(): string {
  const ds = useDatasetStore.getState();
  const tool = useToolStore.getState();
  const settings = useSettingsStore.getState();
  const m = ds.meta;

  const dataset = m
    ? `${ds.kind ?? '—'}:${ds.source ?? '—'} · ${m.width}×${m.height}×${m.nSlices} · ${m.dtype}${m.isRgb ? ' RGB' : ''}`
    : 'none loaded';

  const lines = [
    `App version: ${APP_VERSION} (commit ${GIT_COMMIT})`,
    `OS / Browser: ${navigator.userAgent}${navigator.platform ? ` · ${navigator.platform}` : ''}`,
    `Screen: ${window.location.pathname}`,
    `Active tool: ${tool.tool}${tool.tool === 'magic' ? ` (${tool.magicEngine})` : ''}`,
    `Dataset: ${dataset}`,
    `GPU acceleration: ${gpuStatus()}`,
    `Session id: ${settings.sessionId}`,
  ];
  return lines.join('\n');
}

/** Build the prefilled Google Form URL for the single "context" question.
 *  Pass `embedded` when rendering the form inside an `<iframe>` (Google Forms
 *  honours `embedded=true` for its embedded layout). */
export function buildFeedbackUrl(formUrl: string, entryId: string, context: string, embedded = false): string {
  if (!formUrl) return formUrl;
  const sep = formUrl.includes('?') ? '&' : '?';
  const params = new URLSearchParams({ usp: 'pp_url' });
  if (embedded) params.set('embedded', 'true');
  if (entryId) params.set(`entry.${entryId}`, context);
  return `${formUrl}${sep}${params.toString()}`;
}
