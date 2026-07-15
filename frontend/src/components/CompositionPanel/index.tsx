/**
 * CompositionPanel — modular feature composition window for ipred.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ArrowsClockwise,
  CircleNotch,
  FloppyDisk,
  Plus,
  Trash,
} from '@phosphor-icons/react';
import { useConnectionStore } from '@/stores/connectionStore';
import {
  getIpredComposition,
  listIpredCompositions,
  listIpredModules,
  previewIpredComposition,
  upsertIpredComposition,
  type CompositionDoc,
  type CompositionNode,
  type FeatureModuleInfo,
} from '@/lib/ipredApi';
import { cn } from '@/lib/utils';

const btn =
  'px-2 py-1 rounded border text-[11px] transition-colors disabled:opacity-40 ' +
  'bg-white text-gray-700 border-gray-200 hover:bg-sky-50 hover:border-sky-300';

const section =
  'rounded-md border border-gray-200 bg-white p-4 flex flex-col gap-3';

function defaultParams(mod: FeatureModuleInfo): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [key, schema] of Object.entries(mod.params_schema ?? {})) {
    const s = schema as { default?: unknown };
    if (s && 'default' in s) out[key] = s.default;
  }
  return out;
}

function newNodeId(nodes: CompositionNode[]): string {
  let i = nodes.length + 1;
  const ids = new Set(nodes.map((n) => n.id));
  while (ids.has(`n${i}`)) i += 1;
  return `n${i}`;
}

/** Build / edit ordered module composition and show concat preview. */
export default function CompositionPanel() {
  const preferredCompositionId = useConnectionStore((s) => s.preferredCompositionId);
  const setPreferredCompositionId = useConnectionStore((s) => s.setPreferredCompositionId);

  const [modules, setModules] = useState<FeatureModuleInfo[]>([]);
  const [compositions, setCompositions] = useState<CompositionDoc[]>([]);
  const [name, setName] = useState('Custom composition');
  const [nodes, setNodes] = useState<CompositionNode[]>([]);
  const [outputs, setOutputs] = useState<string[]>([]);
  const [previewLabels, setPreviewLabels] = useState<string[]>([]);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const moduleById = useMemo(() => {
    const m = new Map<string, FeatureModuleInfo>();
    for (const x of modules) m.set(x.id, x);
    return m;
  }, [modules]);

  const refresh = useCallback(async () => {
    setError(null);
    try {
      const [mods, comps] = await Promise.all([
        listIpredModules(),
        listIpredCompositions(),
      ]);
      setModules(mods);
      setCompositions(comps.sort((a, b) => a.name.localeCompare(b.name)));
      const preferred =
        preferredCompositionId &&
        comps.find((c) => c.id === preferredCompositionId);
      const pick = preferred ?? comps.find((c) => c.id === 'comp-skimage-slimsam') ?? comps[0];
      if (pick) {
        const detail = await getIpredComposition(pick.id);
        setEditingId(detail.id);
        setName(detail.name);
        setNodes(detail.nodes ?? []);
        setOutputs(detail.outputs ?? []);
        setPreviewLabels(detail.preview_labels ?? []);
        if (pick.id !== preferredCompositionId) {
          setPreferredCompositionId(pick.id);
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [preferredCompositionId, setPreferredCompositionId]);

  useEffect(() => {
    void refresh();
    // Intentionally once on mount + when preference externally cleared
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const selectComposition = async (id: string) => {
    setError(null);
    setPreferredCompositionId(id);
    try {
      const detail = await getIpredComposition(id);
      setEditingId(detail.id);
      setName(detail.name);
      setNodes(detail.nodes ?? []);
      setOutputs(detail.outputs ?? []);
      setPreviewLabels(detail.preview_labels ?? []);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const refreshPreview = async (
    nextNodes: CompositionNode[],
    nextOutputs: string[],
  ) => {
    try {
      const r = await previewIpredComposition({
        name,
        nodes: nextNodes,
        outputs: nextOutputs,
      });
      setPreviewLabels(r.preview_labels ?? []);
    } catch {
      setPreviewLabels([]);
    }
  };

  const addModule = (moduleId: string) => {
    const mod = moduleById.get(moduleId);
    if (!mod) return;
    const id = newNodeId(nodes);
    const node: CompositionNode = {
      id,
      module: moduleId,
      params: defaultParams(mod),
    };
    if (mod.accepts_input_from && nodes.length > 0) {
      // Prefer last node that can feed an image / emb
      node.input_from = nodes[nodes.length - 1].id;
    }
    const nextNodes = [...nodes, node];
    const nextOutputs =
      mod.produces_channels && !outputs.includes(id)
        ? [...outputs, id]
        : outputs;
    setNodes(nextNodes);
    setOutputs(nextOutputs);
    void refreshPreview(nextNodes, nextOutputs);
  };

  const removeNode = (nid: string) => {
    const nextNodes = nodes
      .filter((n) => n.id !== nid)
      .map((n) =>
        n.input_from === nid ? { ...n, input_from: undefined } : n,
      );
    const nextOutputs = outputs.filter((o) => o !== nid);
    setNodes(nextNodes);
    setOutputs(nextOutputs);
    void refreshPreview(nextNodes, nextOutputs);
  };

  const updateNodeParams = (nid: string, patch: Record<string, unknown>) => {
    const nextNodes = nodes.map((n) =>
      n.id === nid ? { ...n, params: { ...(n.params ?? {}), ...patch } } : n,
    );
    setNodes(nextNodes);
    void refreshPreview(nextNodes, outputs);
  };

  const setInputFrom = (nid: string, inputFrom: string) => {
    const nextNodes = nodes.map((n) =>
      n.id === nid
        ? { ...n, input_from: inputFrom || undefined }
        : n,
    );
    setNodes(nextNodes);
    void refreshPreview(nextNodes, outputs);
  };

  const toggleOutput = (nid: string) => {
    const next = outputs.includes(nid)
      ? outputs.filter((o) => o !== nid)
      : [...outputs, nid];
    setOutputs(next);
    void refreshPreview(nodes, next);
  };

  const moveOutput = (nid: string, dir: -1 | 1) => {
    const i = outputs.indexOf(nid);
    if (i < 0) return;
    const j = i + dir;
    if (j < 0 || j >= outputs.length) return;
    const next = [...outputs];
    [next[i], next[j]] = [next[j], next[i]];
    setOutputs(next);
    void refreshPreview(nodes, next);
  };

  const save = async (asClone: boolean) => {
    setBusy(true);
    setError(null);
    try {
      const saved = await upsertIpredComposition({
        name: asClone ? `${name} copy` : name,
        nodes,
        outputs,
        composition_id: asClone || !editingId ? undefined : editingId,
      });
      setPreferredCompositionId(saved.id);
      setEditingId(saved.id);
      setName(saved.name);
      await refresh();
      await selectComposition(saved.id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const chainSummary = nodes
    .map((n) => {
      const label = moduleById.get(n.module)?.name ?? n.module;
      const arrow = n.input_from ? `←${n.input_from}` : '';
      return `${n.id}:${label}${arrow}`;
    })
    .join(' · ');

  return (
    <section className={section} data-testid="ipred-composition">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold text-gray-800">Composition</h2>
        <button type="button" className={btn} onClick={() => void refresh()}>
          <ArrowsClockwise size={12} className="inline" /> Reload
        </button>
      </div>

      <p className="text-[11px] text-gray-500 leading-snug">
        String modules together. <strong>Outputs</strong> order is how channels
        concatenate into the feature bank.
      </p>

      <label className="flex flex-col gap-1 text-xs text-gray-600">
        Active composition
        <select
          className="rounded border border-gray-200 px-2 py-1.5 text-sm text-gray-800"
          value={preferredCompositionId ?? ''}
          onChange={(e) => void selectComposition(e.target.value)}
          data-testid="ipred-composition-select"
        >
          {compositions.map((c) => (
            <option key={c.id} value={c.id}>
              {c.name}
              {c.builtin ? ' (builtin)' : ''}
            </option>
          ))}
        </select>
      </label>

      <div className="grid grid-cols-1 md:grid-cols-[minmax(0,11rem)_1fr] gap-3">
        <div className="flex flex-col gap-1.5">
          <span className="text-[10px] font-semibold uppercase text-gray-500 tracking-wide">
            Modules
          </span>
          {modules.map((m) => (
            <button
              key={m.id}
              type="button"
              disabled={!m.ready || busy}
              title={m.description}
              onClick={() => addModule(m.id)}
              className={cn(
                'flex items-center justify-between gap-1 rounded border px-2 py-1.5 text-left text-[11px]',
                m.ready
                  ? 'bg-white border-gray-200 hover:border-sky-300 hover:bg-sky-50 text-gray-800'
                  : 'bg-gray-50 border-gray-100 text-gray-400 cursor-not-allowed',
              )}
              data-testid={`ipred-module-${m.id}`}
            >
              <span>
                <Plus size={11} className="inline mr-1" />
                {m.name}
              </span>
              <span className="text-[9px] text-gray-400">{m.runtime}</span>
            </button>
          ))}
        </div>

        <div className="flex flex-col gap-2 min-w-0">
          <label className="flex flex-col gap-0.5 text-xs text-gray-600">
            Name
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="rounded border border-gray-200 px-1.5 py-1 text-sm text-gray-800"
            />
          </label>

          <p className="text-[10px] text-gray-500 truncate" title={chainSummary}>
            Graph: {chainSummary || 'empty'}
          </p>

          <div className="flex flex-col gap-2 max-h-72 overflow-y-auto">
            {nodes.map((n) => {
              const mod = moduleById.get(n.module);
              return (
                <div
                  key={n.id}
                  className="rounded border border-gray-100 bg-gray-50 p-2 flex flex-col gap-1.5"
                  data-testid={`ipred-node-${n.id}`}
                >
                  <div className="flex items-center justify-between gap-1">
                    <span className="text-[11px] font-medium text-gray-800">
                      {n.id} · {mod?.name ?? n.module}
                    </span>
                    <button
                      type="button"
                      className={btn}
                      onClick={() => removeNode(n.id)}
                      title="Remove"
                    >
                      <Trash size={11} />
                    </button>
                  </div>
                  {mod?.accepts_input_from && (
                    <label className="flex flex-col gap-0.5 text-[10px] text-gray-600">
                      Input from
                      <select
                        className="rounded border border-gray-200 px-1 py-0.5 text-[11px]"
                        value={n.input_from ?? ''}
                        onChange={(e) => setInputFrom(n.id, e.target.value)}
                      >
                        <option value="">raw / gray</option>
                        {nodes
                          .filter((o) => o.id !== n.id)
                          .map((o) => (
                            <option key={o.id} value={o.id}>
                              {o.id} ({o.module})
                            </option>
                          ))}
                      </select>
                    </label>
                  )}
                  {n.module === 'tomojepa' && (
                    <label className="flex flex-col gap-0.5 text-[10px] text-gray-600">
                      Weights
                      <select
                        className="rounded border border-gray-200 px-1 py-0.5 text-[11px]"
                        value={String(n.params?.weights_id ?? 'mark25')}
                        onChange={(e) =>
                          updateNodeParams(n.id, { weights_id: e.target.value })
                        }
                      >
                        <option value="mark25">Mark25</option>
                        <option value="mark11">Mark11</option>
                      </select>
                    </label>
                  )}
                  {(n.module === 'tomojepa' || n.module === 'pca' || n.module === 'clahe') && (
                    <div className="grid grid-cols-2 gap-1 text-[10px] text-gray-600">
                      {n.module === 'tomojepa' && (
                        <>
                          <label className="flex flex-col gap-0.5">
                            Input size
                            <input
                              type="number"
                              className="rounded border border-gray-200 px-1 py-0.5"
                              value={Number(n.params?.input_size ?? 512)}
                              onChange={(e) =>
                                updateNodeParams(n.id, {
                                  input_size: Number(e.target.value) || 512,
                                })
                              }
                            />
                          </label>
                          <label className="inline-flex items-center gap-1 mt-4">
                            <input
                              type="checkbox"
                              checked={Boolean(n.params?.resize ?? true)}
                              onChange={(e) =>
                                updateNodeParams(n.id, { resize: e.target.checked })
                              }
                            />
                            Resize
                          </label>
                        </>
                      )}
                      {n.module === 'pca' && (
                        <label className="flex flex-col gap-0.5">
                          PCA dims
                          <input
                            type="number"
                            className="rounded border border-gray-200 px-1 py-0.5"
                            value={Number(n.params?.dims ?? 64)}
                            onChange={(e) =>
                              updateNodeParams(n.id, {
                                dims: Number(e.target.value) || 64,
                              })
                            }
                          />
                        </label>
                      )}
                      {n.module === 'clahe' && (
                        <label className="flex flex-col gap-0.5">
                          Clip limit
                          <input
                            type="number"
                            step={0.01}
                            className="rounded border border-gray-200 px-1 py-0.5"
                            value={Number(n.params?.clip_limit ?? 0.01)}
                            onChange={(e) =>
                              updateNodeParams(n.id, {
                                clip_limit: Number(e.target.value) || 0.01,
                              })
                            }
                          />
                        </label>
                      )}
                    </div>
                  )}
                  <div className="flex items-center gap-2 text-[10px]">
                    <label className="inline-flex items-center gap-1 text-gray-700">
                      <input
                        type="checkbox"
                        checked={outputs.includes(n.id)}
                        onChange={() => toggleOutput(n.id)}
                        disabled={!mod?.produces_channels}
                      />
                      In bank concat
                    </label>
                    {outputs.includes(n.id) && (
                      <span className="flex gap-1">
                        <button type="button" className={btn} onClick={() => moveOutput(n.id, -1)}>
                          ↑
                        </button>
                        <button type="button" className={btn} onClick={() => moveOutput(n.id, 1)}>
                          ↓
                        </button>
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <div
        className="rounded border border-sky-100 bg-sky-50/70 p-2"
        data-testid="ipred-concat-preview"
      >
        <span className="text-[10px] font-semibold uppercase text-gray-500 tracking-wide">
          Bank will concatenate
        </span>
        <p className="text-[11px] text-gray-800 mt-1 break-words">
          {previewLabels.length
            ? previewLabels.join(' · ')
            : '— add channel-producing nodes and mark them as outputs —'}
        </p>
        <p className="text-[10px] text-gray-500 mt-1">
          Outputs order: {outputs.join(' → ') || 'none'}
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          className={btn}
          disabled={busy || nodes.length === 0}
          onClick={() => void save(false)}
        >
          {busy ? (
            <CircleNotch size={12} className="inline animate-spin" />
          ) : (
            <FloppyDisk size={12} className="inline" />
          )}{' '}
          Save
        </button>
        <button
          type="button"
          className={btn}
          disabled={busy || nodes.length === 0}
          onClick={() => void save(true)}
        >
          Clone & save
        </button>
      </div>

      {error && (
        <p className="text-xs text-red-600" data-testid="ipred-composition-error">
          {error}
        </p>
      )}
    </section>
  );
}
