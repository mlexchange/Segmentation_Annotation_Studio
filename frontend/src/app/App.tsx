import { useState } from 'react';
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

function App() {
  const { selectedPaths, setSelectedPaths } = useHubSelectedTabs();
  const navigate = useNavigate();
  const location = useLocation();
  const [showTabSelector, setShowTabSelector] = useState(false);

  const handleStartHub = (paths: string[]) => {
    setSelectedPaths(paths);
    setShowTabSelector(false);
    if (paths.length > 0) navigate(paths[0]);
  };

  const filteredRoutes = selectedPaths !== null
    ? allRoutes.filter(r => selectedPaths.includes(r.path))
    : [];

  if (selectedPaths === null || showTabSelector || filteredRoutes.length === 0) {
    // Default to all tabs selected on first load
    handleStartHub(allRoutes.map(r => r.path));
    return null;
  }

  if (location.pathname === '/' && !filteredRoutes.some(r => r.path === '/')) {
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
        selectedPaths={selectedPaths || []}
        onSelectionChange={setSelectedPaths}
      />
    </>
  );
}

export default App;
