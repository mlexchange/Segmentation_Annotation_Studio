import { NavLink } from "react-router";
import { cn } from "@/lib/utils";
import { BookOpenText, SquaresFour, Bug } from "@phosphor-icons/react";

import { RouteItem } from "@/types/navigationRouterTypes";

export type HubSidebarProps = {
    routes: RouteItem[];
    className?: string;
    activeLinkClassName?: string;
    inactiveLinkClassName?: string;
    docsUrl?: string;
    /** When set, renders a "Feedback" (bug/feature) button above Docs. */
    onFeedback?: () => void;
    /** When set, Docs opens via this handler (in-app iframe) instead of a new tab. */
    onDocs?: () => void;
    onOpenTabSelector?: () => void;
};

/** HubSidebar — vertical nav with one link per route, an optional docs link, and an optional tab-selector button. */
export default function HubSidebar({ routes, className, activeLinkClassName, inactiveLinkClassName, docsUrl, onFeedback, onDocs, onOpenTabSelector }: HubSidebarProps) {
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
            {(onFeedback || docsUrl) && (
                <div className="mt-auto flex flex-col items-center w-full px-2 pt-2">
                    <div className="h-px w-10/12 border-b border-white/50 mb-2" />
                    {onFeedback && (
                        <button
                            type="button"
                            onClick={onFeedback}
                            className={baseNavStyles}
                            title="Report a bug or request a feature"
                        >
                            <span className="shrink-0 flex items-center justify-center">
                                <Bug size={32} />
                            </span>
                            <span className="font-light text-center text-sm leading-tight break-words">
                                Feedback
                            </span>
                        </button>
                    )}
                    {docsUrl && (
                        onDocs ? (
                            <button
                                type="button"
                                onClick={onDocs}
                                className={baseNavStyles}
                                title="Open documentation"
                            >
                                <span className="shrink-0 flex items-center justify-center">
                                    <BookOpenText size={32} />
                                </span>
                                <span className="font-light text-center text-sm leading-tight break-words">
                                    Docs
                                </span>
                            </button>
                        ) : (
                            <a
                                href={docsUrl}
                                target="_blank"
                                rel="noopener noreferrer"
                                className={baseNavStyles}
                                title="Open documentation"
                            >
                                <span className="shrink-0 flex items-center justify-center">
                                    <BookOpenText size={32} />
                                </span>
                                <span className="font-light text-center text-sm leading-tight break-words">
                                    Docs
                                </span>
                            </a>
                        )
                    )}
                </div>
            )}
        </aside>
    );
}