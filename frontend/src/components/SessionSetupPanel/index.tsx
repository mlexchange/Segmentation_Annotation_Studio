/**
 * # REMOVE THIS AND USE YOUR OWN STUFF
 *
 * SessionSetupPanel — TEMPORARY scaffold: reusable label name/color defs +
 * optional past CatBoost shelf model. Replace with your own label taxonomy
 * loader and model-selection UX, then delete this panel.
 */
import { useCallback, useEffect, useState } from 'react';
import { BookmarkSimple, FloppyDisk, Trash } from '@phosphor-icons/react';
import { API_BASE } from '@/config';
import { KillSwitchBanner } from '@/components/KillSwitchBanner';
import { useConnectionStore } from '@/stores/connectionStore';
import {
  DEFAULT_COLORS,
  useClassStore,
  type AnnotationClass,
} from '@/stores/classStore';
import { formatFeatureRecipe } from '@/lib/featureRecipe';
import { cn } from '@/lib/utils';

interface LabelSetSummary {
  id: string;
  name: string;
  n_classes: number;
  created_at: string;
  classes?: AnnotationClass[];
}

interface ShelfModelSummary {
  id: string;
  name: string;
  class_ids: number[];
  n_train: number;
  n_cal: number;
  uses_sam?: boolean;
  feature_recipe?: Record<string, unknown>;
  n_features?: number;
}

const btn =
  'px-2 py-1 rounded border text-[11px] transition-colors disabled:opacity-40 ' +
  'bg-white text-gray-700 border-gray-200 hover:bg-sky-50 hover:border-sky-300';

//////////////////////////////////////////////////////////////////////////////
// # REMOVE THIS AND USE YOUR OWN STUFF
// Hardcoded starter classes (sample / void / air) — not production taxonomy.
//////////////////////////////////////////////////////////////////////////////
const STARTER: AnnotationClass[] = [
  { classId: 1, label: 'sample', color: DEFAULT_COLORS[0], isVisible: true },
  { classId: 2, label: 'void', color: DEFAULT_COLORS[1], isVisible: true },
  { classId: 3, label: 'air', color: DEFAULT_COLORS[2], isVisible: true },
];

interface SessionSetupPanelProps {
  compact?: boolean;
}

/**
 * # REMOVE THIS AND USE YOUR OWN STUFF
 * Pick reusable label names/colors and an optional shelf model (+ feature recipe).
 */
export default function SessionSetupPanel({ compact = false }: SessionSetupPanelProps) {
  const preferredLabelSetId = useConnectionStore((s) => s.preferredLabelSetId);
  const preferredClasses = useConnectionStore((s) => s.preferredClasses);
  const preferredShelfModelId = useConnectionStore((s) => s.preferredShelfModelId);
  const preferredShelfModelName = useConnectionStore((s) => s.preferredShelfModelName);
  const preferredFeatureRecipe = useConnectionStore((s) => s.preferredFeatureRecipe);
  const preferredFeatureSetupId = useConnectionStore((s) => s.preferredFeatureSetupId);
  const setPreferredLabelSet = useConnectionStore((s) => s.setPreferredLabelSet);
  const setPreferredShelfModel = useConnectionStore((s) => s.setPreferredShelfModel);
  const sessionClasses = useClassStore((s) => s.classes);
  const setClasses = useClassStore((s) => s.setClasses);

  const [sets, setSets] = useState<LabelSetSummary[]>([]);
  const [models, setModels] = useState<ShelfModelSummary[]>([]);
  const [newName, setNewName] = useState('My labels');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const [lsRes, mRes] = await Promise.all([
        fetch(`${API_BASE}/api/label-sets`),
        fetch(`${API_BASE}/api/clf/models`),
      ]);
      if (lsRes.ok) {
        const body = (await lsRes.json()) as { sets: LabelSetSummary[] };
        setSets(body.sets ?? []);
      }
      if (mRes.ok) {
        const body = (await mRes.json()) as ShelfModelSummary[];
        setModels(Array.isArray(body) ? body : []);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const applyClassesNow = (classes: AnnotationClass[]) => {
    setClasses(classes.map((c) => ({ ...c })));
  };

  const selectSet = async (id: string | null) => {
    setError(null);
    if (!id) {
      setPreferredLabelSet({ id: null, classes: null });
      return;
    }
    setBusy(true);
    try {
      const res = await fetch(`${API_BASE}/api/label-sets/${id}`);
      if (!res.ok) throw new Error(await res.text());
      const body = (await res.json()) as { id: string; classes: AnnotationClass[] };
      const classes = body.classes ?? [];
      setPreferredLabelSet({ id: body.id, classes });
      applyClassesNow(classes);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const saveCurrentAsSet = async () => {
    setError(null);
    const classes = sessionClasses.length > 0 ? sessionClasses : STARTER;
    if (!sessionClasses.length) applyClassesNow(STARTER);
    setBusy(true);
    try {
      const res = await fetch(`${API_BASE}/api/label-sets`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: newName, classes }),
      });
      if (!res.ok) throw new Error(await res.text());
      const body = (await res.json()) as { id: string; classes: AnnotationClass[] };
      setPreferredLabelSet({ id: body.id, classes: body.classes });
      applyClassesNow(body.classes);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const deleteSet = async (id: string) => {
    setBusy(true);
    try {
      await fetch(`${API_BASE}/api/label-sets/${id}`, { method: 'DELETE' });
      if (preferredLabelSetId === id) setPreferredLabelSet({ id: null, classes: null });
      await refresh();
    } finally {
      setBusy(false);
    }
  };

  const selectedModel = models.find((m) => m.id === preferredShelfModelId) ?? null;
  const recipeShown = preferredFeatureRecipe ?? selectedModel?.feature_recipe ?? null;

  return (
    <div
      className={cn(
        'flex flex-col gap-3 rounded-lg border-2 border-amber-500/80 bg-white/95 text-gray-800',
        compact ? 'p-2.5' : 'p-4',
      )}
      data-testid="session-setup-panel-scaffold"
    >
      <KillSwitchBanner detail="Scaffold: label-set loader + past-model picker on Connect/Browse. Replace with your taxonomy and model UX." />

      <div className="flex items-center gap-1.5">
        <BookmarkSimple size={16} className="text-sky-700" />
        <span className="text-xs font-semibold uppercase tracking-wide text-sky-900">
          Labels & model
        </span>
        <span className="ml-auto font-mono text-[9px] font-bold text-amber-700">
          TEMP / REMOVE
        </span>
      </div>
      <p className="text-[11px] text-gray-500 leading-snug -mt-1">
        Save label <span className="font-medium">names and colors</span> so you don’t redefine them
        each sample. Optionally pick a past model — its required features are shown below.
      </p>

      {/* # REMOVE THIS AND USE YOUR OWN STUFF — label set load/save */}
      <div className="flex flex-col gap-1">
        <label className="text-[10px] font-medium text-gray-600 uppercase tracking-wide">
          Label names & colors{' '}
          <span className="font-mono normal-case text-amber-700"># REMOVE</span>
        </label>
        <select
          className="rounded border border-gray-200 px-2 py-1.5 text-xs"
          value={preferredLabelSetId ?? ''}
          disabled={busy}
          onChange={(e) => {
            void selectSet(e.target.value || null);
          }}
        >
          <option value="">None (use draft / empty)</option>
          {sets.map((s) => (
            <option key={s.id} value={s.id}>
              {s.name} ({s.n_classes})
            </option>
          ))}
        </select>
        {preferredClasses && preferredClasses.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mt-1">
            {preferredClasses.map((c) => (
              <span
                key={c.classId}
                className="inline-flex items-center gap-1 rounded-full border border-gray-200 bg-gray-50 px-1.5 py-0.5 text-[10px] text-gray-700"
              >
                <span
                  className="inline-block h-2.5 w-2.5 rounded-sm shrink-0"
                  style={{ background: c.color }}
                  title={c.color}
                />
                {c.label}
              </span>
            ))}
          </div>
        )}
        <div className="flex gap-1 mt-1">
          <input
            className="flex-1 rounded border border-gray-200 px-2 py-1 text-[11px]"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            placeholder="Set name"
          />
          <button
            type="button"
            className={btn}
            disabled={busy}
            title={
              sessionClasses.length
                ? 'Save current label names & colors'
                : 'Save starter sample/void/air names & colors'
            }
            onClick={() => {
              void saveCurrentAsSet();
            }}
          >
            <FloppyDisk size={12} />
            Save
          </button>
        </div>
        {sets.length > 0 && !compact && (
          <ul className="max-h-24 overflow-y-auto flex flex-col gap-0.5 mt-1">
            {sets.map((s) => (
              <li
                key={s.id}
                className="flex items-center gap-1 text-[10px] rounded px-1 py-0.5 hover:bg-gray-50"
              >
                <button
                  type="button"
                  className="flex-1 text-left truncate text-gray-700 hover:text-sky-700"
                  onClick={() => {
                    void selectSet(s.id);
                  }}
                >
                  {s.name}
                </button>
                <button
                  type="button"
                  className={cn(btn, 'px-1')}
                  title="Delete"
                  disabled={busy}
                  onClick={() => {
                    void deleteSet(s.id);
                  }}
                >
                  <Trash size={11} />
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* # REMOVE THIS AND USE YOUR OWN STUFF — shelf model selection */}
      <div className="flex flex-col gap-1">
        <label className="text-[10px] font-medium text-gray-600 uppercase tracking-wide">
          Past model (optional){' '}
          <span className="font-mono normal-case text-amber-700"># REMOVE</span>
        </label>
        <select
          className="rounded border border-gray-200 px-2 py-1.5 text-xs"
          value={preferredShelfModelId ?? ''}
          disabled={busy}
          onChange={(e) => {
            const id = e.target.value || null;
            const m = models.find((x) => x.id === id);
            setPreferredShelfModel({
              id,
              name: m?.name ?? null,
              featureRecipe: m?.feature_recipe ?? null,
            });
          }}
        >
          <option value="">None</option>
          {models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.name}
              {m.n_features != null ? ` · ${m.n_features} feats` : ''}
            </option>
          ))}
        </select>
        {preferredShelfModelId && (
          <div className="rounded-md border border-sky-100 bg-sky-50/80 px-2 py-1.5 text-[10px] text-sky-900 leading-snug">
            <p className="font-medium">{preferredShelfModelName ?? 'Selected model'}</p>
            <p className="mt-0.5 text-sky-800/90">
              Features in: {formatFeatureRecipe(recipeShown)}
              {preferredFeatureSetupId
                ? ` · ipred setup ${preferredFeatureSetupId}`
                : ''}
            </p>
            {selectedModel?.class_ids?.length ? (
              <p className="mt-0.5 text-sky-800/80">
                Class ids: {selectedModel.class_ids.join(', ')}
              </p>
            ) : null}
            <p className="mt-0.5 text-sky-700/70">
              Preprocess will use these feature settings; Apply on Train.
            </p>
          </div>
        )}
        {models.length === 0 && (
          <p className="text-[10px] text-gray-400">No saved models yet — train & save on Train.</p>
        )}
      </div>

      {error && <p className="text-[10px] text-red-600 break-words">{error}</p>}
    </div>
  );
}
