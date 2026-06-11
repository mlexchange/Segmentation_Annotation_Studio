import { useState, useEffect } from 'react';
import { useNavigate, useLocation, Navigate } from 'react-router';
import './App.css';
import { RouteItem } from '@/types/navigationRouterTypes';
import HubAppLayout from '@/components/HubAppLayout';
import { useHubSelectedTabs } from '@/hooks/useHubSelectedTabs';
import { PlugsConnected, PencilSimple, Export } from '@phosphor-icons/react';
import ConnectPage from './pages/ConnectPage';
import AnnotatePage from './pages/AnnotatePage';
import ExportPage from './pages/ExportPage';
import CustomizePages from '@/components/CustomizePages';

const allRoutes: RouteItem[] = [
  {
    path: '/connect',
    label: 'Connect',
    icon: <PlugsConnected size={32} />,
    element: <ConnectPage />,
  },
  {
    path: '/annotate',
    label: 'Annotate',
    icon: <PencilSimple size={32} />,
    element: <AnnotatePage />,
    isBackgroundTransparent: true,
  },
  {
    path: '/export',
    label: 'Export',
    icon: <Export size={32} />,
    element: <ExportPage />,
  },
];

const DEFAULT_PATHS = allRoutes.map((r) => r.path);

function App() {
  const { selectedPaths, setSelectedPaths } = useHubSelectedTabs();
  const navigate = useNavigate();
  const location = useLocation();
  const [showTabSelector, setShowTabSelector] = useState(false);

  // Validate stored paths — discard any that don't belong to this app
  const validPaths = selectedPaths?.filter((p) => DEFAULT_PATHS.includes(p)) ?? null;
  const needsInit = validPaths === null || validPaths.length === 0;

  // Navigate on first mount if no valid paths stored
  useEffect(() => {
    if (needsInit) {
      setSelectedPaths(DEFAULT_PATHS);
      navigate('/connect', { replace: true });
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
        headerTitle="SAM3 Annotation Studio"
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
