import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter } from 'react-router';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import App from './app/App';
import { installImageSliceGc } from './hooks/useImageSlice';
import './app/index.css';

const queryClient = new QueryClient();

// Image slices are cached as blob object URLs, which the browser keeps alive until
// explicitly revoked. Release them as their queries leave the cache.
installImageSliceGc(queryClient);

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter basename={import.meta.env.BASE_URL}>
      <QueryClientProvider client={queryClient}>
        <App />
      </QueryClientProvider>
    </BrowserRouter>
  </StrictMode>
);
