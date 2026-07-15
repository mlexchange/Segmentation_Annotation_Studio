/**
 * IpredConfigPanel — service health + trainer defaults (compositions are separate).
 */
import { useCallback, useEffect, useState } from 'react';
import { ArrowsClockwise, CircleNotch } from '@phosphor-icons/react';
import { useConnectionStore } from '@/stores/connectionStore';
import { ipredHealth, listIpredTrainers } from '@/lib/ipredApi';
import { cn } from '@/lib/utils';

const btn =
  'px-2 py-1 rounded border text-[11px] transition-colors disabled:opacity-40 ' +
  'bg-white text-gray-700 border-gray-200 hover:bg-sky-50 hover:border-sky-300';

const section =
  'rounded-md border border-gray-200 bg-white p-4 flex flex-col gap-3';

type HealthState = 'unknown' | 'ok' | 'down' | 'checking';

/** Configure ipred service health and trainer defaults. */
export default function IpredConfigPanel() {
  const preferredTrainerId = useConnectionStore((s) => s.preferredTrainerId);
  const preferredTrainerConfig = useConnectionStore((s) => s.preferredTrainerConfig);
  const setPreferredTrainer = useConnectionStore((s) => s.setPreferredTrainer);
  const preferredCompositionId = useConnectionStore((s) => s.preferredCompositionId);

  const [health, setHealth] = useState<HealthState>('unknown');
  const [trainers, setTrainers] = useState<string[]>(['catboost']);
  const [error, setError] = useState<string | null>(null);

  const refreshHealth = useCallback(async () => {
    setHealth('checking');
    try {
      const h = await ipredHealth();
      setHealth(h.status === 'ok' ? 'ok' : 'down');
    } catch {
      setHealth('down');
    }
  }, []);

  const refreshTrainers = useCallback(async () => {
    setError(null);
    try {
      const trainerIds = await listIpredTrainers().catch(() => ['catboost']);
      setTrainers(trainerIds.length ? trainerIds : ['catboost']);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void refreshHealth();
    void refreshTrainers();
  }, [refreshHealth, refreshTrainers]);

  return (
    <div className="flex flex-col gap-4">
      <section className={section} data-testid="ipred-health">
        <div className="flex items-center justify-between gap-2">
          <h2 className="text-sm font-semibold text-gray-800">Service</h2>
          <button
            type="button"
            className={btn}
            onClick={() => void refreshHealth()}
            disabled={health === 'checking'}
          >
            {health === 'checking' ? (
              <CircleNotch size={12} className="inline animate-spin" />
            ) : (
              <ArrowsClockwise size={12} className="inline" />
            )}{' '}
            Refresh
          </button>
        </div>
        <p
          className={cn(
            'text-sm',
            health === 'ok' && 'text-emerald-700',
            health === 'down' && 'text-red-600',
            (health === 'unknown' || health === 'checking') && 'text-gray-500',
          )}
          data-testid="ipred-health-status"
        >
          {health === 'ok' && 'ipred reachable'}
          {health === 'down' && 'ipred unreachable — start via ./start_all.sh'}
          {health === 'checking' && 'Checking…'}
          {health === 'unknown' && 'Status unknown'}
        </p>
        {preferredCompositionId && (
          <p className="text-[11px] text-gray-500">
            Active composition{' '}
            <span className="font-mono text-gray-700">{preferredCompositionId}</span>
          </p>
        )}
      </section>

      <section className={section} data-testid="ipred-trainer">
        <h2 className="text-sm font-semibold text-gray-800">Trainer defaults</h2>
        <label className="flex flex-col gap-1 text-xs text-gray-600">
          Plugin
          <select
            className="rounded border border-gray-200 px-2 py-1.5 text-sm text-gray-800"
            value={preferredTrainerId}
            onChange={(e) => setPreferredTrainer({ id: e.target.value })}
            data-testid="ipred-trainer-select"
          >
            {trainers.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </label>
        {preferredTrainerId === 'catboost' && (
          <div className="grid grid-cols-3 gap-2 text-xs">
            <label className="flex flex-col gap-0.5 text-gray-600">
              Iterations
              <input
                type="number"
                min={10}
                value={preferredTrainerConfig.iterations}
                onChange={(e) =>
                  setPreferredTrainer({
                    config: { iterations: Number(e.target.value) || 200 },
                  })
                }
                className="rounded border border-gray-200 px-1.5 py-1 text-gray-800"
              />
            </label>
            <label className="flex flex-col gap-0.5 text-gray-600">
              Depth
              <input
                type="number"
                min={2}
                max={12}
                value={preferredTrainerConfig.depth}
                onChange={(e) =>
                  setPreferredTrainer({
                    config: { depth: Number(e.target.value) || 6 },
                  })
                }
                className="rounded border border-gray-200 px-1.5 py-1 text-gray-800"
              />
            </label>
            <label className="flex flex-col gap-0.5 text-gray-600">
              Learning rate
              <input
                type="number"
                min={0.01}
                step={0.01}
                value={preferredTrainerConfig.learning_rate}
                onChange={(e) =>
                  setPreferredTrainer({
                    config: { learning_rate: Number(e.target.value) || 0.1 },
                  })
                }
                className="rounded border border-gray-200 px-1.5 py-1 text-gray-800"
              />
            </label>
          </div>
        )}
      </section>

      {error && (
        <p className="text-xs text-red-600" data-testid="ipred-config-error">
          {error}
        </p>
      )}
    </div>
  );
}
