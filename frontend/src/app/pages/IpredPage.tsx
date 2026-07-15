/**
 * IpredPage — configure feature compositions and trainer defaults.
 */
import { useNavigate } from 'react-router';
import { Cpu } from '@phosphor-icons/react';
import { useConnectionStore } from '@/stores/connectionStore';
import CompositionPanel from '@/components/CompositionPanel';
import IpredConfigPanel from '@/components/IpredConfigPanel';

export default function IpredPage() {
  const navigate = useNavigate();
  const kind = useConnectionStore((s) => s.kind);

  if (!kind) {
    return (
      <div className="flex flex-col h-full items-center justify-center gap-4">
        <Cpu size={48} className="text-sky-700" />
        <p className="text-sky-200 text-sm">Connect a dataset before configuring ipred.</p>
        <button
          type="button"
          className="px-4 py-2 rounded-md bg-sky-600 text-white text-sm hover:bg-sky-700"
          onClick={() => navigate('/connect')}
        >
          Go to Connect
        </button>
      </div>
    );
  }

  return (
    <div className="h-full overflow-auto p-6">
      <div className="max-w-4xl mx-auto flex flex-col gap-4">
        <header className="flex flex-col gap-1">
          <h1 className="text-lg font-semibold text-gray-900">Ipred</h1>
          <p className="text-sm text-gray-600">
            Compose feature modules (skimage, CLAHE, SlimSAM, TomoJEPA, PCA). Preprocess
            Compute uses the active composition.
          </p>
        </header>
        <CompositionPanel />
        <IpredConfigPanel />
      </div>
    </div>
  );
}
