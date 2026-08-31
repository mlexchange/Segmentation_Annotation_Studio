/**
 * CapabilityBanner — Train-tab readiness status. Shows install guidance when
 * torch/dlsia aren't available (e.g. the Docker image, which intentionally
 * excludes them) instead of letting every downstream action fail opaquely.
 */
import { CheckCircle, WarningCircle } from '@phosphor-icons/react';
import type { TrainCapability } from '@/hooks/useTrainCapability';

interface CapabilityBannerProps {
  capability: TrainCapability;
}

export default function CapabilityBanner({ capability }: CapabilityBannerProps) {
  if (!capability.torch_available) {
    return (
      <div className="flex items-start gap-2 rounded-lg border border-amber-600/50 bg-amber-900/20 px-3 py-2.5 text-sm text-amber-200">
        <WarningCircle size={18} className="mt-0.5 shrink-0" />
        <div>
          <p className="font-medium">Training is unavailable on this server.</p>
          <p className="mt-1 text-xs text-amber-200/80">
            torch isn't installed. Run this app via <code className="font-mono">start_all.sh</code> with{' '}
            <code className="font-mono">INSTALL_ML=1</code> (default on Apple Silicon) on a machine where it
            can install ML dependencies — the Docker image intentionally omits them.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 rounded-lg border border-slate-700 bg-slate-800/60 px-3 py-2 text-xs text-slate-300">
      <span className="flex items-center gap-1.5 text-emerald-300">
        <CheckCircle size={14} />
        torch {capability.torch_version} · {capability.device ?? 'cpu'}
      </span>
      <span>
        dlsia (TUNet):{' '}
        <span className={capability.dlsia.available ? 'text-slate-200' : 'text-amber-300'}>
          {capability.dlsia.available ? 'available' : 'not installed'}
        </span>
      </span>
      {capability.busy && <span className="text-sky-300">A training/inference job is currently running.</span>}
    </div>
  );
}
