/**
 * ReferencePage — authors the per-dataset annotation guide: for each class a
 * label, color, a written description of what it is / how it looks, and example
 * image crops. The guide is dataset-scoped (persisted by sourceKey via
 * useGuideSync) and its classes surface as one-click suggestions in the Annotate
 * tab, keeping annotators consistent with the lead's intended labels and colors.
 */
import { useRef, useState } from 'react';
import { useNavigate } from 'react-router';
import { Plus, Trash, Images, DownloadSimple, Sparkle, CircleDashed, Export, UploadSimple } from '@phosphor-icons/react';
import { useDatasetStore } from '@/stores/datasetStore';
import { useClassStore } from '@/stores/classStore';
import { useReferenceGuideStore, type GuideClass } from '@/stores/referenceGuideStore';
import { useGuideSync, generateGuide } from '@/hooks/useGuideSync';
import { useSave } from '@/hooks/useSave';
import { buildSourceKey } from '@/lib/sourceKey';
import { getClassPalette } from '@/lib/classColors';

/** Reads a File as a base64 data URL. */
function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

/** One editable guide entry: color, label, description, and example crops. */
function GuideEntryRow({ index, entry }: { index: number; entry: GuideClass }) {
  const { updateEntry, removeEntry } = useReferenceGuideStore();
  const fileRef = useRef<HTMLInputElement>(null);

  const addCrops = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const urls = await Promise.all(Array.from(files).map(fileToDataUrl));
    updateEntry(index, { exampleCrops: [...entry.exampleCrops, ...urls] });
  };

  const removeCrop = (cropIdx: number) =>
    updateEntry(index, { exampleCrops: entry.exampleCrops.filter((_, i) => i !== cropIdx) });

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-gray-200 bg-white p-3">
      <div className="flex items-center gap-2">
        <input
          type="color"
          aria-label="Class color"
          value={entry.color || '#1f77b4'}
          onChange={(e) => updateEntry(index, { color: e.target.value })}
          className="h-7 w-8 cursor-pointer rounded border border-gray-200"
        />
        <input
          placeholder="Class label"
          value={entry.label}
          onChange={(e) => updateEntry(index, { label: e.target.value })}
          className="flex-1 rounded border border-gray-200 px-2 py-1 text-sm font-medium"
        />
        <button
          aria-label="Remove class from guide"
          onClick={() => removeEntry(index)}
          className="p-1 text-gray-400 hover:text-red-500"
        >
          <Trash size={16} />
        </button>
      </div>

      <textarea
        placeholder="What is this class? How does it look? When should it (not) be used?"
        value={entry.description}
        onChange={(e) => updateEntry(index, { description: e.target.value })}
        rows={3}
        className="w-full resize-y rounded border border-gray-200 px-2 py-1 text-sm"
      />

      <div className="flex flex-wrap items-center gap-2">
        {entry.exampleCrops.map((src, cropIdx) => (
          <div key={cropIdx} className="group relative">
            <img
              src={src}
              alt={`${entry.label} example ${cropIdx + 1}`}
              className="h-16 w-16 rounded border border-gray-200 object-cover"
            />
            <button
              aria-label="Remove example"
              onClick={() => removeCrop(cropIdx)}
              className="absolute -right-1.5 -top-1.5 hidden rounded-full bg-red-500 p-0.5 text-white group-hover:block"
            >
              <Trash size={10} />
            </button>
          </div>
        ))}
        <button
          onClick={() => fileRef.current?.click()}
          className="flex h-16 w-16 flex-col items-center justify-center gap-1 rounded border border-dashed border-gray-300 text-gray-400 hover:border-sky-400 hover:text-sky-600"
        >
          <Images size={18} />
          <span className="text-[10px]">Example</span>
        </button>
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          multiple
          className="hidden"
          onChange={(e) => { addCrops(e.target.files); e.target.value = ''; }}
        />
      </div>
    </div>
  );
}

export default function ReferencePage() {
  const navigate = useNavigate();
  const { source, kind, serverUri, meta } = useDatasetStore();
  const { classes } = useClassStore();
  const { entries, notes, addEntry, setNotes, applyGenerated, setGuide } = useReferenceGuideStore();
  const importRef = useRef<HTMLInputElement>(null);

  const sourceKey = source && kind
    ? buildSourceKey(kind as 'tiled' | 'local', source, serverUri)
    : null;
  useGuideSync(sourceKey);

  const { buildSavePayload, versions, fetchVersionPayload } = useSave(sourceKey);
  // '' = current annotation; otherwise the chosen version number.
  const [genSource, setGenSource] = useState<string>('');
  const [generating, setGenerating] = useState(false);

  /** Generate example crops per class from the current annotation or a saved version. */
  const handleGenerate = async () => {
    if (!sourceKey) { alert('Open a sample first — the guide is generated from its annotation.'); return; }
    setGenerating(true);
    try {
      const payload = genSource === ''
        ? buildSavePayload()
        : await fetchVersionPayload(Number(genSource));
      if (!payload) { alert('Nothing to generate from — annotate some regions first.'); return; }

      const shapeCount = Object.values(payload.slices).reduce((n, s) => n + (Array.isArray(s) ? s.length : 0), 0);
      if (shapeCount === 0 || payload.classes.length === 0) {
        alert(genSource === ''
          ? 'The current sample has no annotations yet. Draw some regions in the Annotate tab, or pick a saved version.'
          : 'That version has no annotations to generate examples from.');
        return;
      }

      const result = await generateGuide(sourceKey, payload);
      applyGenerated(result.classes);

      const totalCrops = result.classes.reduce((n, c) => n + c.exampleCrops.length, 0);
      if (totalCrops === 0) {
        alert('Added the classes, but no example crops could be extracted (the image could not be read for cropping). Class labels and colors were still imported.');
      }
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Guide generation failed.');
    } finally {
      setGenerating(false);
    }
  };

  /** Next unused color from the active palette, given colors already in the guide. */
  const nextColor = () => {
    const palette = getClassPalette();
    const used = new Set(entries.map((e) => e.color));
    return palette.find((c) => !used.has(c)) ?? palette[entries.length % palette.length];
  };

  /** Download the guide (classes + descriptions + embedded example crops) as a shareable JSON file. */
  const exportGuide = () => {
    const blob = new Blob([JSON.stringify({ classes: entries, notes }, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'annotation-guide.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  /** Load a shared guide file, attaching it to the currently-open dataset. */
  const importGuide = async (file: File | undefined) => {
    if (!file || !sourceKey) return;
    try {
      const parsed = JSON.parse(await file.text());
      const rawClasses = Array.isArray(parsed?.classes) ? parsed.classes : [];
      const cleaned: GuideClass[] = rawClasses.map((c: Record<string, unknown>) => ({
        label: String(c.label ?? ''),
        color: String(c.color ?? '#1f77b4'),
        description: String(c.description ?? ''),
        exampleCrops: Array.isArray(c.exampleCrops) ? (c.exampleCrops as string[]).map(String) : [],
      }));
      setGuide(cleaned, String(parsed?.notes ?? ''), sourceKey);
    } catch {
      alert('Could not read that file as an annotation guide.');
    }
  };

  /** Seed the guide from the classes currently defined in the Annotate tab. */
  const importFromClasses = () => {
    const have = new Set(entries.map((e) => e.label.toLowerCase()));
    for (const c of classes) {
      if (have.has(c.label.toLowerCase())) continue;
      addEntry({ label: c.label, color: c.color, description: '', exampleCrops: [] });
    }
  };

  if (!meta || !sourceKey) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-4 text-sky-200">
        <p>No sample loaded. Pick a sample to author its annotation guide.</p>
        <button
          type="button"
          onClick={() => navigate('/browse')}
          className="rounded-md bg-sky-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-sky-700"
        >
          Go to Browse
        </button>
      </div>
    );
  }

  return (
    <div className="mx-auto flex h-full w-full max-w-3xl flex-col gap-4 overflow-y-auto p-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-lg font-semibold text-gray-800">Annotation Guide</h1>
          <p className="text-sm text-gray-500">
            Describe each class so annotators label consistently. These classes appear as one-click
            suggestions in the Annotate tab. Saved automatically with this dataset.
          </p>
        </div>
        <div className="flex flex-shrink-0 items-center gap-2">
          <button
            onClick={() => importRef.current?.click()}
            className="flex items-center gap-1 rounded-md bg-gray-100 px-2 py-1 text-xs font-medium text-gray-700 hover:bg-gray-200"
            title="Load a shared guide file into this dataset"
          >
            <UploadSimple size={13} />
            Import
          </button>
          <button
            onClick={exportGuide}
            disabled={entries.length === 0}
            className="flex items-center gap-1 rounded-md bg-gray-100 px-2 py-1 text-xs font-medium text-gray-700 hover:bg-gray-200 disabled:opacity-50"
            title="Download this guide as a shareable file"
          >
            <Export size={13} />
            Export
          </button>
          <input
            ref={importRef}
            type="file"
            accept="application/json,.json"
            className="hidden"
            onChange={(e) => { importGuide(e.target.files?.[0]); e.target.value = ''; }}
          />
        </div>
      </div>

      <div className="flex flex-col gap-1">
        <label className="text-xs font-semibold uppercase tracking-wide text-gray-500">Task notes</label>
        <textarea
          placeholder="Overall guidance for this annotation task…"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          rows={2}
          className="w-full resize-y rounded border border-gray-200 px-2 py-1 text-sm"
        />
      </div>

      <div className="flex flex-wrap items-center gap-2 rounded-lg border border-gray-200 bg-gray-50 p-3">
        <Sparkle size={16} className="text-sky-600" />
        <span className="text-sm font-medium text-gray-700">Generate example crops from</span>
        <select
          value={genSource}
          onChange={(e) => setGenSource(e.target.value)}
          className="rounded border border-gray-200 bg-white px-2 py-1 text-sm"
        >
          <option value="">Current annotation</option>
          {versions.map((v) => (
            <option key={v.version} value={String(v.version)}>
              Version {v.version} ({v.shape_count} shapes)
            </option>
          ))}
        </select>
        <button
          onClick={handleGenerate}
          disabled={generating}
          className="flex items-center gap-1.5 rounded-md bg-sky-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-sky-700 disabled:opacity-60"
        >
          {generating ? <CircleDashed size={14} className="animate-spin" /> : <Sparkle size={14} />}
          {generating ? 'Generating…' : 'Generate'}
        </button>
        <span className="w-full text-xs text-gray-400">
          Adds the largest example of each annotated class. Your descriptions are kept.
        </span>
      </div>

      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold uppercase tracking-wide text-gray-500">
          Classes ({entries.length})
        </span>
        <div className="flex items-center gap-2">
          {classes.length > 0 && (
            <button
              onClick={importFromClasses}
              className="flex items-center gap-1 rounded-md bg-gray-100 px-2 py-1 text-xs font-medium text-gray-700 hover:bg-gray-200"
              title="Add classes currently defined in the Annotate tab"
            >
              <DownloadSimple size={13} />
              Import current classes
            </button>
          )}
          <button
            onClick={() => addEntry({ label: '', color: nextColor(), description: '', exampleCrops: [] })}
            className="flex items-center gap-1 rounded-md bg-sky-600 px-2 py-1 text-xs font-medium text-white hover:bg-sky-700"
          >
            <Plus size={13} />
            Add class
          </button>
        </div>
      </div>

      <div className="flex flex-col gap-2 pb-6">
        {entries.length === 0 && (
          <p className="rounded-lg border border-dashed border-gray-300 py-8 text-center text-sm text-gray-400">
            No classes in the guide yet. Add one, or import the classes you've defined in Annotate.
          </p>
        )}
        {entries.map((entry, i) => (
          <GuideEntryRow key={i} index={i} entry={entry} />
        ))}
      </div>
    </div>
  );
}
