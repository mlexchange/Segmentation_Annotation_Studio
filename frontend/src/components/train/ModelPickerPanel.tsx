/**
 * ModelPickerPanel — choose a model family (DINOv3 + LoRA fine-tune, or a
 * dlsia TUNet trained from scratch) and, for DINOv3, which checkpoint/size.
 */
import type { DinoCheckpoint, TrainCapability } from '@/hooks/useTrainCapability';

export type ModelFamily = 'dinov3_lora' | 'dlsia_tunet';

interface ModelPickerPanelProps {
  capability: TrainCapability;
  modelFamily: ModelFamily;
  onModelFamilyChange: (family: ModelFamily) => void;
  selectedCheckpoint: DinoCheckpoint | null;
  onSelectCheckpoint: (checkpoint: DinoCheckpoint) => void;
}

const SIZE_LABELS: Record<string, string> = {
  vits16: 'Small (21M params)',
  vits16plus: 'Small+ (29M params)',
  vitb16: 'Base (86M params)',
  vitl16: 'Large (300M params)',
  vith16plus: 'Huge+ (840M params) — needs significant memory',
  vit7b16: '7B (6.7B params) — ~27 GB of weights; expect slow training',
};

function formatSize(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(0)} MB`;
}

export default function ModelPickerPanel({
  capability, modelFamily, onModelFamilyChange, selectedCheckpoint, onSelectCheckpoint,
}: ModelPickerPanelProps) {
  return (
    <div className="space-y-3">
      <p className="text-xs font-medium text-slate-400 uppercase tracking-wide">Model</p>
      <div className="flex flex-col gap-1.5">
        <label
          className={`flex items-start gap-3 px-3 py-2.5 rounded-lg border cursor-pointer transition-colors ${
            modelFamily === 'dinov3_lora' ? 'border-sky-500 bg-sky-900/30' : 'border-slate-600 hover:border-slate-500'
          }`}
        >
          <input
            type="radio" name="model-family" className="mt-0.5 shrink-0 accent-sky-500"
            checked={modelFamily === 'dinov3_lora'} onChange={() => onModelFamilyChange('dinov3_lora')}
          />
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-slate-200">DINOv3 + LoRA</p>
            <p className="text-xs text-slate-400 mt-0.5">
              Fine-tunes a pretrained foundation backbone — works well with few annotated slices.
            </p>
          </div>
        </label>
        <label
          className={`flex items-start gap-3 px-3 py-2.5 rounded-lg border cursor-pointer transition-colors ${
            modelFamily === 'dlsia_tunet' ? 'border-sky-500 bg-sky-900/30' : 'border-slate-600 hover:border-slate-500'
          }`}
        >
          <input
            type="radio" name="model-family" className="mt-0.5 shrink-0 accent-sky-500"
            checked={modelFamily === 'dlsia_tunet'} onChange={() => onModelFamilyChange('dlsia_tunet')}
          />
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-slate-200">dlsia TUNet</p>
            <p className="text-xs text-slate-400 mt-0.5">
              A lightweight U-Net trained from scratch on just your data — no pretrained checkpoint needed.
            </p>
          </div>
        </label>
      </div>

      {modelFamily === 'dinov3_lora' && (
        <div className="space-y-1.5 pt-1">
          {capability.dinov3.checkpoints.length === 0 ? (
            <p className="text-xs text-amber-300">
              No DINOv3 checkpoints found in {capability.models_dir || 'the configured models directory'}.
              Place a <code className="font-mono">dinov3_&lt;arch&gt;_pretrain_*.pth</code> file there and reload.
            </p>
          ) : (
            capability.dinov3.checkpoints.map((ckpt) => (
              <label
                key={ckpt.file_name}
                className={`flex items-center gap-3 px-3 py-2 rounded-lg border cursor-pointer transition-colors ${
                  selectedCheckpoint?.file_name === ckpt.file_name
                    ? 'border-sky-500 bg-sky-900/30' : 'border-slate-600 hover:border-slate-500'
                }`}
              >
                <input
                  type="radio" name="dino-checkpoint" className="shrink-0 accent-sky-500"
                  checked={selectedCheckpoint?.file_name === ckpt.file_name}
                  onChange={() => onSelectCheckpoint(ckpt)}
                />
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-slate-200">{SIZE_LABELS[ckpt.arch] ?? ckpt.arch}</p>
                  <p className="text-xs text-slate-400">{ckpt.file_name} · {formatSize(ckpt.size_bytes)}</p>
                </div>
              </label>
            ))
          )}
        </div>
      )}

      {modelFamily === 'dlsia_tunet' && !capability.dlsia.available && (
        <p className="text-xs text-amber-300 pt-1">
          dlsia isn't installed on this server. Install it (<code className="font-mono">uv pip install dlsia</code>)
          to use the TUNet model family.
        </p>
      )}
    </div>
  );
}
