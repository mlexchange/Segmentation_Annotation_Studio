import { useState, useEffect } from 'react';
import { useNavigate, useLocation, Navigate } from 'react-router';
import './App.css';
import { RouteItem } from '@/types/navigationRouterTypes';
import HubAppLayout from '@/components/HubAppLayout';
import { useHubSelectedTabs } from '@/hooks/useHubSelectedTabs';
import { DOCS_URL } from '@/config';
import { PlugsConnected, PencilSimple, MagnifyingGlass, BookOpen } from '@phosphor-icons/react';
import ConnectPage from './pages/ConnectPage';
import AnnotatePage from './pages/AnnotatePage';
import BrowsePage from './pages/BrowsePage';
import ReferencePage from './pages/ReferencePage';
import CustomizePages from '@/components/CustomizePages';

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
];

const DEFAULT_PATHS = allRoutes.map((r) => r.path);

/**
 * App — root component: validates persisted tab selection against known routes, redirects on
 * first load, and renders the hub layout plus the tab-customisation modal.
 */
function App() {
  const { selectedPaths, setSelectedPaths } = useHubSelectedTabs();
  const navigate = useNavigate();
  const location = useLocation();
  const [showTabSelector, setShowTabSelector] = useState(false);

  // Validate stored paths — discard unknown paths and merge in any newly added tabs.
  const storedValid = selectedPaths?.filter((p) => DEFAULT_PATHS.includes(p)) ?? null;
  const validPaths =
    storedValid === null
      ? null
      : [...storedValid, ...DEFAULT_PATHS.filter((p) => !storedValid.includes(p))];
  const needsInit = validPaths === null || validPaths.length === 0;

  // Navigate on first mount if no valid paths stored; persist merged tab list.
  useEffect(() => {
    if (needsInit) {
      setSelectedPaths(DEFAULT_PATHS);
      navigate('/connect', { replace: true });
    } else if (
      selectedPaths &&
      validPaths &&
      (validPaths.length !== selectedPaths.length ||
        !validPaths.every((p, i) => p === selectedPaths[i]))
    ) {
      setSelectedPaths(validPaths);
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
        headerTitle="Segmentation Annotation Tool"
        docsUrl={DOCS_URL}
      />
      <CustomizePages
        routes={allRoutes}
        selectedPaths={validPaths!}
        onSelectionChange={setSelectedPaths}
      />
    </>
  );
}

export default App;
