import HubHeader from "@/components/HubHeader";
import HubMainContent from "@/components/HubMainContent";
import HubSidebar from "@/components/HubSidebar";
import { cn } from "@/lib/utils";

import { RouteItem } from "@/types/navigationRouterTypes";

export type HubAppLayoutProps = {
    routes: RouteItem[];
    headerTitle?: string;
    headerTitleClassName?: string;
    headerLogoUrl?: string;
    mainContentClassName?: string;
    headerClassName?: string;
    sidebarClassName?: string;
    sidebarActiveLinkClassName?: string;
    sidebarInactiveLinkClassName?: string;
    onOpenTabSelector?: () => void;
    }
export default function HubAppLayout ( {
    routes,
    headerTitle,
    headerLogoUrl,
    mainContentClassName, 
    headerClassName,
    headerTitleClassName,
    sidebarClassName,
    sidebarActiveLinkClassName,
    sidebarInactiveLinkClassName,
    onOpenTabSelector
  }: HubAppLayoutProps) {

return (
    <div className="flex h-screen w-screen max-w-full overflow-hidden">
        {/* Sidebar: fixed overlay so it always receives clicks above any route content */}
        <div className="fixed left-0 top-0 z-[9999] h-screen w-24 shrink-0" style={{ pointerEvents: 'none' }}>
            <div className="h-full w-full" style={{ pointerEvents: 'auto' }}>
                <HubSidebar 
                    routes={routes}
                    className={sidebarClassName} 
                    activeLinkClassName={sidebarActiveLinkClassName}
                    inactiveLinkClassName={sidebarInactiveLinkClassName}
                    onOpenTabSelector={onOpenTabSelector}
                />
            </div>
        </div>
        {/* Main: offset by sidebar width; z-0 so sidebar (z-[9999]) always receives clicks */}
        <div className="relative z-0 flex min-w-0 flex-1 flex-col overflow-hidden pl-24">
            <HubHeader 
                title={headerTitle} 
                logoUrl={headerLogoUrl}
                className={headerClassName}
                titleClassName={headerTitleClassName}
                onOpenTabSelector={onOpenTabSelector}
            />
            <HubMainContent 
                routes={routes}
                className={cn("h-[calc(100vh-4rem)] min-h-0 max-w-full overflow-hidden", mainContentClassName)} 
            />
        </div>
    </div>
)
  }
