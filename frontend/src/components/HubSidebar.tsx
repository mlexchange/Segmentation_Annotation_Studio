import { NavLink } from "react-router";
import { cn } from "@/lib/utils";
import { SquaresFour } from "@phosphor-icons/react";

import { RouteItem } from "@/types/navigationRouterTypes";

export type HubSidebarProps = {
    routes: RouteItem[];
    className?: string;
    activeLinkClassName?: string;
    inactiveLinkClassName?: string;
    onOpenTabSelector?: () => void;
};

export default function HubSidebar({ routes, className, activeLinkClassName, inactiveLinkClassName, onOpenTabSelector }: HubSidebarProps) {
    const baseNavStyles = cn(
        "flex flex-col items-center justify-start gap-1 min-h-[5rem] w-full py-3 px-2 rounded-lg",
        "text-white hover:bg-sky-800 cursor-pointer transition-colors",
        "border-0 bg-transparent no-underline",
        inactiveLinkClassName
    );
    const activeNavStyles = cn("bg-sky-300 text-black", activeLinkClassName);

    return (
        <aside
            className={cn("h-full w-full bg-sky-950 flex flex-col py-4 overflow-y-auto", className)}
        >
            {onOpenTabSelector && (
                <div className="flex flex-col items-center w-full px-2 mb-2">
                    <button
                        type="button"
                        onClick={onOpenTabSelector}
                        className={baseNavStyles}
                        title="Select Tabs"
                    >
                        <span className="shrink-0 flex items-center justify-center">
                            <SquaresFour size={32} weight="fill" />
                        </span>
                        <span className="font-light text-center text-sm leading-tight break-words">
                            Select Tabs
                        </span>
                    </button>
                    <div className="h-px w-10/12 border-b border-white/50 my-2" />
                </div>
            )}
            {routes.map((item, index) => (
                <div key={item.path} className="flex flex-col items-center w-full px-2">
                    <NavLink
                        to={item.path}
                        end={item.path === "/"}
                        className={({ isActive }) =>
                            isActive ? cn(baseNavStyles, activeNavStyles) : baseNavStyles
                        }
                        title={item.label}
                    >
                        <span className="shrink-0 flex items-center justify-center">{item.icon}</span>
                        <span className="font-light text-center text-sm leading-tight break-words">
                            {item.label}
                        </span>
                    </NavLink>
                    {index < routes.length - 1 && (
                        <div className="h-px w-10/12 border-b border-white/50 my-2" />
                    )}
                </div>
            ))}
        </aside>
    );
}