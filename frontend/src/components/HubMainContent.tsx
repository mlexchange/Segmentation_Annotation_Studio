import { Suspense } from "react";
import { Routes, Route, useNavigate, useLocation } from "react-router";
import { cn } from "@/lib/utils";

import { RouteItem } from "@/types/navigationRouterTypes";

/** Fallback shown for unmatched routes; offers a button back to the first available page. */
function NotFoundFallback({ routes }: { routes: RouteItem[] }) {
  const navigate = useNavigate();
  const location = useLocation();
  const firstPath = routes.length > 0 ? routes[0].path : "/";

  return (
    <div className="flex flex-col items-center justify-center min-h-[50vh] text-white">
      <h1 className="text-2xl font-semibold mb-2">Page not found</h1>
      <p className="text-white/80 mb-4">
        <code className="bg-white/10 px-2 py-0.5 rounded">{location.pathname}</code> doesn’t match any page.
      </p>
      <button
        type="button"
        onClick={() => navigate(firstPath)}
        className="px-4 py-2 rounded bg-white/20 hover:bg-white/30 text-sm font-medium"
      >
        Go to {routes.length > 0 ? routes[0].label : "home"}
      </button>
    </div>
  );
}

export type HubMainContentProps = {
    routes: RouteItem[];
    className?: string;
}

/**
 * HubMainContent — renders the active route inside the main area, applying full-bleed
 * layout for routes flagged isBackgroundTransparent and a 404 fallback otherwise.
 */
export default function HubMainContent({ routes, className }: HubMainContentProps) {
    const location = useLocation();
    const fullBleed = routes.some(
        (r) => r.isBackgroundTransparent && r.path === location.pathname,
    );

    return (
        <main
            className={cn(
                "bg-sky-900 h-full w-full max-w-full overflow-hidden",
                !fullBleed && "p-8",
                className,
            )}
        >
            <div
                className={cn(
                    "h-full w-full max-w-full",
                    fullBleed ? "overflow-hidden" : "overflow-y-auto",
                )}
            >
                <Suspense fallback={<div className="flex h-full w-full items-center justify-center text-white/70 text-sm">Loading…</div>}>
                    <Routes>
                        {routes.map((route) => (
                            <Route
                                key={route.path}
                                path={route.path}
                                element={route.element}
                            />
                        ))}
                        <Route path="*" element={<NotFoundFallback routes={routes} />} />
                    </Routes>
                </Suspense>
            </div>
        </main>
    );
}
