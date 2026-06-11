/**
 * ExportPage — COCO dataset export with split table, dry-run preview, and import.
 */
import { useState } from 'react';
import { useDatasetStore } from '@/stores/datasetStore';
import { useAnnotationStore } from '@/stores/annotationStore';
import { useClassStore } from '@/stores/classStore';
import { API_BASE } from '@/config';

type Split = 'train' | 'valid' | 'test' | 'auto';

export default function ExportPage() {
  const { source, kind, serverUri, meta, renderOpts } = useDatasetStore();
  const { byImage, splitBySlice, negativeSlices, setSplitForSlice } = useAnnotationStore();
  const { classes } = useClassStore();

  const [outDir, setOutDir] = useState('');
  const [mode, setMode] = useState<'fail' | 'overwrite' | 'merge'>('fail');
  const [preview, setPreview] = useState<Record<string, unknown> | null>(null);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [status, setStatus] = useState('');

  if (!source || !meta) {
    return (
      <div className="flex h-full items-center justify-center text-gray-400">
        <p>No annotations to export. Connect to a dataset and annotate first.</p>
      </div>
    );
  }

  const sliceMap = byImage[source] ?? {};
  const annotatedSlices = Object.keys(sliceMap).filter((k) => sliceMap[k]?.length > 0);
  const negSlices = negativeSlices[source] ?? [];
  const allSlices = Array.from(new Set([...annotatedSlices, ...negSlices])).sort((a, b) => Number(a) - Number(b));

  if (allSlices.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-gray-400">
        <p>No Annotations to Export! Annotate an image first.</p>
      </div>
    );
  }

  const buildPayload = (dryRun: boolean) => ({
    out_dir: outDir,
    kind,
    source,
    server_uri: serverUri,
    mode,
    dry_run: dryRun,
    render: {
      norm: renderOpts.norm,
      scale: renderOpts.scale,
      vmin_pct: renderOpts.vminPct,
      vmax_pct: renderOpts.vmaxPct,
      cmap: renderOpts.cmap,
    },
    classes: classes.map((c) => ({ classId: c.classId, label: c.label, color: c.color, isVisible: c.isVisible })),
    slices: sliceMap,
    split_by_slice: splitBySlice[source] ?? {},
    auto_split: { ratios: [0.8, 0.1, 0.1], seed: 1234 },
    negative_slices: negSlices,
  });

  const handleDryRun = async () => {
    if (!outDir) { setStatus('Please enter an output directory.'); return; }
    setStatus('Running dry run…');
    try {
      const res = await fetch(`${API_BASE}/api/export/coco`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildPayload(true)),
      });
      const data = await res.json();
      if (!res.ok) { setStatus(`Error: ${JSON.stringify(data)}`); return; }
      setPreview(data);
      setStatus('');
    } catch (e) { setStatus(`Failed: ${e}`); }
  };

  const handleWrite = async () => {
    setStatus('Writing dataset…');
    try {
      const res = await fetch(`${API_BASE}/api/export/coco`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildPayload(false)),
      });
      const data = await res.json();
      if (!res.ok) { setStatus(`Error: ${JSON.stringify(data)}`); return; }
      setResult(data);
      setStatus('');
    } catch (e) { setStatus(`Failed: ${e}`); }
  };

  return (
    <div className="flex flex-col h-full overflow-y-auto p-6 max-w-3xl mx-auto space-y-6">
      <h2 className="text-2xl font-semibold text-gray-800">Export COCO Dataset</h2>

      {/* Render options summary */}
      <div className="bg-gray-50 rounded-lg p-3 text-sm space-y-1">
        <p className="font-medium text-gray-700">Render options (recorded in export)</p>
        <p className="text-gray-500 font-mono text-xs">
          norm={renderOpts.norm} scale={renderOpts.scale} vmin={renderOpts.vminPct}% vmax={renderOpts.vmaxPct}% cmap={renderOpts.cmap}
        </p>
      </div>

      {/* Split table */}
      <div>
        <h3 className="text-sm font-semibold mb-2 text-gray-700">Split assignment ({allSlices.length} slices)</h3>
        <div className="border rounded-lg overflow-hidden text-sm">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="text-left px-3 py-2 text-xs font-semibold text-gray-500">Slice</th>
                <th className="text-left px-3 py-2 text-xs font-semibold text-gray-500">Shapes</th>
                <th className="text-left px-3 py-2 text-xs font-semibold text-gray-500">Type</th>
                <th className="text-left px-3 py-2 text-xs font-semibold text-gray-500">Split</th>
              </tr>
            </thead>
            <tbody>
              {allSlices.map((k) => {
                const isNeg = negSlices.includes(k);
                const shapeCount = sliceMap[k]?.length ?? 0;
                const split = splitBySlice[source]?.[k] ?? 'auto';
                return (
                  <tr key={k} className="border-t hover:bg-gray-50">
                    <td className="px-3 py-1.5 font-mono">{Number(k) + 1}</td>
                    <td className="px-3 py-1.5">{shapeCount}</td>
                    <td className="px-3 py-1.5">
                      {isNeg ? <span className="text-amber-600 text-xs font-medium">negative</span> : 'positive'}
                    </td>
                    <td className="px-3 py-1.5">
                      <select
                        className="text-xs border rounded px-1 py-0.5"
                        value={split}
                        onChange={(e) => setSplitForSlice(source, Number(k), e.target.value as Split)}
                      >
                        {(['auto', 'train', 'valid', 'test'] as const).map((s) => (
                          <option key={s} value={s}>{s}</option>
                        ))}
                      </select>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Output dir + mode */}
      <div className="space-y-3">
        <div>
          <label className="text-sm font-medium text-gray-700 block mb-1">Output directory</label>
          <input
            className="w-full border rounded-md px-3 py-2 text-sm font-mono"
            placeholder="/abs/path/to/dataset"
            value={outDir}
            onChange={(e) => setOutDir(e.target.value)}
          />
        </div>
        <div>
          <label className="text-sm font-medium text-gray-700 block mb-1">Write mode</label>
          <select
            className="border rounded-md px-3 py-2 text-sm"
            value={mode}
            onChange={(e) => setMode(e.target.value as typeof mode)}
          >
            <option value="fail">fail (abort if exists)</option>
            <option value="overwrite">overwrite</option>
            <option value="merge">merge (replace matched images)</option>
          </select>
        </div>
      </div>

      {status && <p className="text-sm text-gray-600">{status}</p>}

      {/* Dry-run preview */}
      {preview && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-3 text-sm">
          <p className="font-medium text-green-800 mb-1">Dry run preview</p>
          <pre className="text-xs text-green-700 whitespace-pre-wrap">{JSON.stringify(preview, null, 2)}</pre>
        </div>
      )}

      {result && (
        <div className="bg-sky-50 border border-sky-200 rounded-lg p-3 text-sm">
          <p className="font-medium text-sky-800 mb-1">Export complete</p>
          <pre className="text-xs text-sky-700 whitespace-pre-wrap">{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}

      <div className="flex gap-3">
        <button
          onClick={handleDryRun}
          className="px-4 py-2 text-sm rounded-md border border-sky-600 text-sky-600 hover:bg-sky-50"
        >
          Dry run preview
        </button>
        {preview && (
          <button
            onClick={handleWrite}
            className="px-4 py-2 text-sm rounded-md bg-sky-600 text-white hover:bg-sky-700"
          >
            Write dataset
          </button>
        )}
      </div>
    </div>
  );
}
