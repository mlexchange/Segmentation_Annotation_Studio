/** Legacy Annotate entry — redirects to Preprocess Hub stage. */
import { Navigate } from 'react-router';

export default function AnnotatePage() {
  return <Navigate to="/preprocess" replace />;
}
