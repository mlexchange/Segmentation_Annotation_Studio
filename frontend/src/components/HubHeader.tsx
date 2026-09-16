import { useNavigate } from 'react-router';
import alsLogo from '@/assets/alsLogo.png';
import { cn } from '@/lib/utils';
import { Gear, Warning } from '@phosphor-icons/react';
import { useConnectionStore } from '@/stores/connectionStore';

export type HubHeaderProps = {
    title?: string;
    logoUrl?: string;
    className?: string;
    titleClassName?: string;
    onOpenTabSelector?: () => void;
}

/** Small persistent Tiled-connection indicator; hidden for local/no connection. */
function ConnectionStatus() {
    const kind = useConnectionStore((s) => s.kind);
    const status = useConnectionStore((s) => s.status);
    const navigate = useNavigate();
    if (kind !== 'tiled' || status === 'unknown') return null;

    if (status === 'error') {
        return (
            <button
                onClick={() => navigate('/connect')}
                className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium text-red-700 bg-red-100 hover:bg-red-200 transition-colors"
                title="Lost connection to the Tiled server — click to reconnect"
            >
                <Warning size={14} weight="fill" />
                Tiled disconnected
            </button>
        );
    }
    return (
        <span
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium text-emerald-700"
            title="Connected to the Tiled server"
        >
            <span className="h-2 w-2 rounded-full bg-emerald-500" />
            Tiled connected
        </span>
    );
}

/** HubHeader — top app bar with logo, title, and an optional "Change Tabs" button. */
export default function HubHeader({title="ALS COMPUTING HUB", logoUrl=alsLogo, className, titleClassName, onOpenTabSelector}: HubHeaderProps) {
    return (
    <header className={cn("bg-sky-100 h-16 flex justify-between items-center", className)}>
        <div className="flex items-center space-x-6 ml-6">
            <img src={logoUrl} alt="ALS logo" width={40} height={40} className="h-10 w-10 aspect-square"/>
            <h1 className={cn("text-sky-950 text-2xl font-semibold", titleClassName)}>{title}</h1>
        </div>
        <div className="flex items-center gap-4 mr-6">
        <ConnectionStatus />
        {onOpenTabSelector && (
            <button
                onClick={onOpenTabSelector}
                className="flex items-center gap-2 px-4 py-2 rounded-lg text-sky-950 hover:bg-sky-100 transition-colors"
                title="Change tabs"
            >
                <Gear size={24} />
                <span className="text-sm font-medium">Change Tabs</span>
            </button>
        )}
        </div>
    </header>
    )
}