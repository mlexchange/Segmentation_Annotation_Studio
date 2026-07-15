import { useEffect, useState } from 'react';
import { Routes, Route, useNavigate, useLocation } from 'react-router';
import { cn } from '@/lib/utils';
import { RouteItem } from '@/types/navigationRouterTypes';
import AnnotateWorkspace from '@/app/pages/AnnotateWorkspace';
import { stageFromPath, type AnnotateStage } from '@/lib/annotateStage';

const WORK_PATHS = new Set(['/preprocess', '/draw', '/train']);

function isWorkPath(pathname: string): boolean {
  return WORK_PATHS.has(pathname);
}

/** Fallback shown for unmatched routes; offers a button back to the first available page. */
function NotFoundFallback({ routes }: { routes: RouteItem[] }) {
  const navigate = useNavigate();
  const location = useLocation();
  const firstPath = routes.length > 0 ? routes[0].path : '/';

  return (
    <div className="flex flex-col items-center justify-center min-h-[50vh] text-white">
      <h1 className="text-2xl font-semibold mb-2">Page not found</h1>
      <p className="text-white/80 mb-4">
        <code className="bg-white/10 px-2 py-0.5 rounded">{location.pathname}</code> doesn&apos;t match any
        page.
      </p>
      <button
        type="button"
        onClick={() => navigate(firstPath)}
        className="px-4 py-2 rounded bg-white/20 hover:bg-white/30 text-sm font-medium"
      >
        Go to {routes.length > 0 ? routes[0].label : 'home'}
      </button>
    </div>
  );
}

export type HubMainContentProps = {
  routes: RouteItem[];
  className?: string;
};

/**
 * HubMainContent — renders the active route inside the main area, applying full-bleed
 * layout for routes flagged isBackgroundTransparent and a 404 fallback otherwise.
 *
 * Annotate Hub stages share one persistent {@link AnnotateWorkspace} so feature /
 * suggest / classifier state survives stage switches.
 */
export default function HubMainContent({ routes, className }: HubMainContentProps) {
  const location = useLocation();
  const onWork = isWorkPath(location.pathname);
  const [workspaceArmed, setWorkspaceArmed] = useState(onWork);

  useEffect(() => {
    if (onWork) setWorkspaceArmed(true);
  }, [onWork]);

  const fullBleed =
    onWork ||
    routes.some((r) => r.isBackgroundTransparent && r.path === location.pathname);

  const otherRoutes = routes.filter((r) => !isWorkPath(r.path));
  const workRoutes = routes.filter((r) => isWorkPath(r.path));

  return (
    <main
      className={cn(
        'bg-sky-900 h-full w-full max-w-full overflow-hidden',
        !fullBleed && 'p-8',
        className,
      )}
    >
      <div
        className={cn(
          'h-full w-full max-w-full',
          fullBleed ? 'overflow-hidden' : 'overflow-y-auto',
        )}
      >
        {workspaceArmed && (
          <div className={cn('h-full w-full', onWork ? 'block' : 'hidden')}>
            <AnnotateWorkspace />
          </div>
        )}
        <Routes>
          {otherRoutes.map((route) => (
            <Route key={route.path} path={route.path} element={route.element} />
          ))}
          {workRoutes.map((route) => (
            <Route
              key={route.path}
              path={route.path}
              element={
                // Workspace is rendered above; route match only drives Hub stage chrome.
                <WorkStageSync expected={stageFromPath(route.path)} />
              }
            />
          ))}
          <Route path="*" element={<NotFoundFallback routes={routes} />} />
        </Routes>
      </div>
    </main>
  );
}

/** Keeps React Router matched for work stages; workspace reads location itself. */
function WorkStageSync({ expected }: { expected: AnnotateStage }) {
  void expected;
  return null;
}
