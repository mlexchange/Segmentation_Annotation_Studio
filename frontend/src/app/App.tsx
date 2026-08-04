import { useEffect, useState, lazy } from 'react';
import { useNavigate, useLocation, Navigate } from 'react-router';
import './App.css';
import { RouteItem } from '@/types/navigationRouterTypes';
import HubAppLayout from '@/components/HubAppLayout';
import { useHubSelectedTabs } from '@/hooks/useHubSelectedTabs';
import { DOCS_URL, FEEDBACK_FORM_URL, FEEDBACK_ENTRY_ID } from '@/config';
import { buildFeedbackContext, buildFeedbackUrl } from '@/lib/feedbackContext';
import { PlugsConnected, PencilSimple, MagnifyingGlass, BookOpen, Brain, Cube } from '@phosphor-icons/react';
// Lazy-loaded pages: keeps the heavy Annotate stack (konva, polygon-clipping,
// magicwand, canvas) out of the initial /connect bundle — each page is its own chunk.
const ConnectPage = lazy(() => import('./pages/ConnectPage'));
const AnnotatePage = lazy(() => import('./pages/AnnotatePage'));
const BrowsePage = lazy(() => import('./pages/BrowsePage'));
const ReferencePage = lazy(() => import('./pages/ReferencePage'));
const TrainPage = lazy(() => import('./pages/TrainPage'));
const VolumePage = lazy(() => import('./pages/VolumePage'));
import CustomizePages from '@/components/CustomizePages';
import IframeModal from '@/components/IframeModal';
import FeedbackConsentModal from '@/components/FeedbackConsentModal';

const allRoutes: RouteItem[] = [
  {
    path: '/connect',
    label: 'Connect',
    icon: <PlugsConnected size={32} />,
    element: <ConnectPage />,
  },
  {
    path: '/browse',
    label: 'Browse',
    icon: <MagnifyingGlass size={32} />,
    element: <BrowsePage />,
    isBackgroundTransparent: true,
  },
  {
    path: '/reference',
    label: 'Reference',
    icon: <BookOpen size={32} />,
    element: <ReferencePage />,
  },
  {
    path: '/annotate',
    label: 'Annotate',
    icon: <PencilSimple size={32} />,
    element: <AnnotatePage />,
    isBackgroundTransparent: true,
  },
  {
    path: '/volume',
    label: '3D',
    icon: <Cube size={32} />,
    element: <VolumePage />,
    isBackgroundTransparent: true,
  },
  {
    path: '/train',
    label: 'Train',
    icon: <Brain size={32} />,
    element: <TrainPage />,
    isBackgroundTransparent: true,
  },
];

const DEFAULT_PATHS = allRoutes.map((r) => r.path);

/**
 * App — root component: validates persisted tab selection against known routes, redirects on
 * first load, and renders the hub layout plus the tab-customisation modal.
 */
function App() {
  const { selectedPaths, knownPaths, setSelectedPaths } = useHubSelectedTabs();
  const navigate = useNavigate();
  const location = useLocation();
  // Docs / Feedback open in an in-app iframe modal rather than a new tab.
  const [iframeModal, setIframeModal] = useState<{ title: string; url: string } | null>(null);
  // Feedback context is shown for explicit confirmation before it's ever sent —
  // set only when the user clicks "Feedback"; cleared on send or cancel.
  const [feedbackContext, setFeedbackContext] = useState<string | null>(null);

  // Discard routes that no longer exist. Only auto-append routes that are new
  // since the route universe was last recorded (`knownPaths`) — a route absent
  // from `storedValid` because the user explicitly hid it via Customize Layout
  // must stay hidden, which a plain diff against DEFAULT_PATHS can't tell apart
  // from "this route didn't exist yet." `knownPaths` being null (first run, or
  // an install that predates this tracking) treats every route as new once, so
  // it can appear for existing users at most one extra time.
  const storedValid = selectedPaths?.filter((p) => DEFAULT_PATHS.includes(p)) ?? null;
  const newlyAddedPaths = DEFAULT_PATHS.filter((p) => !(knownPaths ?? []).includes(p));
  const validPaths =
    storedValid === null
      ? null
      : [...storedValid, ...newlyAddedPaths.filter((p) => !storedValid.includes(p))];
  const needsInit = validPaths === null || validPaths.length === 0;

  // Navigate on first mount if no valid paths stored; persist merged tab list.
  useEffect(() => {
    if (needsInit) {
      setSelectedPaths(DEFAULT_PATHS, DEFAULT_PATHS);
      navigate('/connect', { replace: true });
    } else if (
      selectedPaths &&
      validPaths &&
      (validPaths.length !== selectedPaths.length ||
        !validPaths.every((p, i) => p === selectedPaths[i]))
    ) {
      setSelectedPaths(validPaths, DEFAULT_PATHS);
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const filteredRoutes = needsInit
    ? allRoutes
    : allRoutes.filter((r) => validPaths!.includes(r.path));

  if (needsInit) {
    return null;
  }

  if (location.pathname === '/' && !filteredRoutes.some((r) => r.path === '/')) {
    return <Navigate to={filteredRoutes[0].path} replace />;
  }

  return (
    <>
      <HubAppLayout
        routes={filteredRoutes}
        headerTitle="Segmentation Annotation Studio"
        docsUrl={DOCS_URL}
        onDocs={DOCS_URL ? () => setIframeModal({ title: 'Documentation', url: DOCS_URL }) : undefined}
        onFeedback={
          FEEDBACK_FORM_URL
            ? () => setFeedbackContext(buildFeedbackContext())
            : undefined
        }
      />
      <CustomizePages
        routes={allRoutes}
        selectedPaths={validPaths!}
        onSelectionChange={(paths) => setSelectedPaths(paths, DEFAULT_PATHS)}
      />
      {iframeModal && (
        <IframeModal title={iframeModal.title} url={iframeModal.url} onClose={() => setIframeModal(null)} />
      )}
      {feedbackContext !== null && (
        <FeedbackConsentModal
          context={feedbackContext}
          onCancel={() => setFeedbackContext(null)}
          onSend={() => {
            setIframeModal({
              title: 'Bugs & Feature Requests',
              url: buildFeedbackUrl(FEEDBACK_FORM_URL, FEEDBACK_ENTRY_ID, feedbackContext, true),
            });
            setFeedbackContext(null);
          }}
        />
      )}
    </>
  );
}

export default App;
