/**
 * App — Hub routes: Connect → Browse → Ipred → Preprocess → Draw → Train.
 */
import { useEffect } from 'react';
import { useNavigate, useLocation, Navigate } from 'react-router';
import './App.css';
import { RouteItem } from '@/types/navigationRouterTypes';
import HubAppLayout from '@/components/HubAppLayout';
import { useHubSelectedTabs } from '@/hooks/useHubSelectedTabs';
import {
  PlugsConnected,
  MagnifyingGlass,
  Cpu,
  Faders,
  PencilSimple,
  TreeStructure,
} from '@phosphor-icons/react';
import ConnectPage from './pages/ConnectPage';
import BrowsePage from './pages/BrowsePage';
import IpredPage from './pages/IpredPage';
import PreprocessPage from './pages/PreprocessPage';
import DrawPage from './pages/DrawPage';
import TrainPage from './pages/TrainPage';
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
    path: '/ipred',
    label: 'Ipred',
    icon: <Cpu size={32} />,
    element: <IpredPage />,
  },
  {
    path: '/preprocess',
    label: 'Preprocess',
    icon: <Faders size={32} />,
    element: <PreprocessPage />,
    isBackgroundTransparent: true,
  },
  {
    path: '/draw',
    label: 'Draw',
    icon: <PencilSimple size={32} />,
    element: <DrawPage />,
    isBackgroundTransparent: true,
  },
  {
    path: '/train',
    label: 'Train & Predict',
    icon: <TreeStructure size={32} />,
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
  const { selectedPaths, setSelectedPaths } = useHubSelectedTabs();
  const navigate = useNavigate();
  const location = useLocation();

  const storedValid = selectedPaths?.filter((p) => DEFAULT_PATHS.includes(p)) ?? null;
  const validPaths =
    storedValid === null
      ? null
      : [...storedValid, ...DEFAULT_PATHS.filter((p) => !storedValid.includes(p))];
  const needsInit = validPaths === null || validPaths.length === 0;

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

  if (location.pathname === '/annotate') {
    return <Navigate to="/preprocess" replace />;
  }

  if (location.pathname === '/cleanup') {
    return <Navigate to="/train" replace />;
  }

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
