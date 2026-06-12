export type RouteItem = {
    path: string;
    label: string;
    element: React.ReactNode;
    icon?: React.ReactNode;
    /** When true, page fills the main area edge-to-edge (no outer padding). */
    isBackgroundTransparent?: boolean;
}