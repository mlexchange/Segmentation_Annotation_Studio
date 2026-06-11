import { useState } from 'react';
import { SquaresFour, X } from '@phosphor-icons/react';
import { RouteItem } from '@/types/navigationRouterTypes';
import { cn } from '@/lib/utils';

export type CustomizePagesProps = {
  routes: RouteItem[];
  selectedPaths: string[];
  onSelectionChange: (selectedPaths: string[]) => void;
};

export default function CustomizePages({ 
  routes, 
  selectedPaths, 
  onSelectionChange
}: CustomizePagesProps) {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [tempSelected, setTempSelected] = useState<Set<string>>(() => new Set(selectedPaths));

  const openModal = () => {
    setTempSelected(new Set(selectedPaths));
    setIsModalOpen(true);
  };

  const closeModal = () => {
    setIsModalOpen(false);
  };

  const toggleTab = (path: string) => {
    setTempSelected(prev => {
      const next = new Set(prev);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      return next;
    });
  };

  const handleApply = () => {
    if (tempSelected.size === 0) return;
    onSelectionChange(Array.from(tempSelected));
    closeModal();
  };

  const handleCancel = () => {
    setTempSelected(new Set(selectedPaths));
    closeModal();
  };

  return (
    <>
      {/* Floating Trigger Button - positioned absolutely in top right */}
      <div className="fixed top-3 right-3 z-[9998]">
        <button
          onClick={openModal}
          className="flex items-center gap-2 px-3 py-2 bg-white rounded-lg shadow-lg border text-sky-950 hover:bg-sky-50 transition-colors"
          title="Customize Layout"
        >
          <SquaresFour size={18} />
          <span className="text-sm font-medium">Customize Layout</span>
        </button>
      </div>

      {/* Full Screen Modal */}
      {isModalOpen && (
        <div className="fixed inset-0 z-[10000]">
          {/* Backdrop */}
          <div 
            className="absolute inset-0 bg-black/50 backdrop-blur-sm"
            onClick={closeModal}
          />
          
          {/* Modal Content - Full Screen */}
          <div className="relative w-full h-full bg-gradient-to-br from-sky-900/50 via-sky-800/50 to-sky-950/50 flex flex-col">
            {/* Close Button */}
            <button
              onClick={closeModal}
              className="absolute top-6 right-6 text-white/70 hover:text-white transition-colors z-10"
            >
              <X size={32} />
            </button>

            {/* Content Container */}
            <div className="flex-1 flex flex-col items-center justify-center p-8 overflow-auto">
              {/* Header */}
              <div className="text-center mb-12">
                <h2 className="text-5xl font-bold text-white mb-4">
                  Customize Your Layout
                </h2>
                <p className="text-xl text-sky-200">
                  Select which tabs you want to see in your workspace
                </p>
              </div>

              {/* Tab Grid */}
              <div className="w-full max-w-6xl">
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6 mb-12">
                  {routes.map((route) => {
                    const isSelected = tempSelected.has(route.path);
                    return (
                      <button
                        key={route.path}
                        onClick={() => toggleTab(route.path)}
                        className={cn(
                          'relative flex flex-col items-center justify-center p-8 rounded-xl',
                          'transition-all duration-200 transform hover:scale-105',
                          'border-2 min-h-[160px]',
                          isSelected
                            ? 'bg-sky-300/20 border-sky-300 shadow-lg shadow-sky-300/20'
                            : 'bg-white/5 border-white/20 hover:bg-white/10 hover:border-white/40'
                        )}
                      >
                        {isSelected && (
                          <div className="absolute top-3 right-3 w-7 h-7 bg-sky-300 rounded-full flex items-center justify-center">
                            <svg
                              className="w-4 h-4 text-sky-900"
                              fill="none"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth="3"
                              viewBox="0 0 24 24"
                              stroke="currentColor"
                            >
                              <path d="M5 13l4 4L19 7" />
                            </svg>
                          </div>
                        )}
                        <div className={cn(
                          'mb-4 transition-colors',
                          isSelected ? 'text-sky-300' : 'text-white/70'
                        )}>
                          {route.icon}
                        </div>
                        <span className={cn(
                          'text-base font-medium text-center transition-colors',
                          isSelected ? 'text-white' : 'text-white/70'
                        )}>
                          {route.label}
                        </span>
                      </button>
                    );
                  })}
                </div>

                {/* Footer */}
                <div className="flex items-center justify-between">
                  <div className="text-sky-200 text-lg">
                    {tempSelected.size} {tempSelected.size === 1 ? 'tab' : 'tabs'} selected
                  </div>
                  
                  <div className="flex items-center gap-6">
                    <button
                      type="button"
                      onClick={handleCancel}
                      className="px-8 py-3 text-lg text-white/70 hover:text-white border-2 border-white/30 hover:border-white/50 rounded-lg transition-colors"
                    >
                      Cancel
                    </button>
                    <button
                      type="button"
                      onClick={handleApply}
                      disabled={tempSelected.size === 0}
                      className={cn(
                        'px-10 py-3 text-lg font-semibold rounded-lg transition-all',
                        tempSelected.size > 0
                          ? 'bg-sky-300 text-sky-900 hover:bg-sky-200 shadow-lg shadow-sky-300/30'
                          : 'bg-gray-500 text-gray-300 cursor-not-allowed'
                      )}
                    >
                      Apply Changes
                    </button>
                  </div>
                </div>

                {tempSelected.size === 0 && (
                  <p className="text-center text-red-300 mt-6 text-lg">
                    Please select at least one tab to continue
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}