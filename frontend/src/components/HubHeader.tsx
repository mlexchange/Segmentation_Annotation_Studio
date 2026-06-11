import alsLogo from '@/assets/alsLogo.png';
import { cn } from '@/lib/utils';
import { Gear } from '@phosphor-icons/react';

export type HubHeaderProps = {
    title?: string;
    logoUrl?: string;
    className?: string;
    titleClassName?: string;
    onOpenTabSelector?: () => void;
}
export default function HubHeader({title="ALS COMPUTING HUB", logoUrl=alsLogo, className, titleClassName, onOpenTabSelector}: HubHeaderProps) {
    return (
    <header className={cn("bg-white h-16 flex justify-between items-center", className)}>
        <div className="flex items-center space-x-6 ml-6">
            <img src={logoUrl} alt="ALS logo" className="h-10 aspect-square"/>
            <h1 className={cn("text-sky-950 text-2xl font-semibold", titleClassName)}>{title}</h1>
        </div>
        {onOpenTabSelector && (
            <button
                onClick={onOpenTabSelector}
                className="mr-6 flex items-center gap-2 px-4 py-2 rounded-lg text-sky-950 hover:bg-sky-100 transition-colors"
                title="Change tabs"
            >
                <Gear size={24} />
                <span className="text-sm font-medium">Change Tabs</span>
            </button>
        )}
    </header>
    )
}