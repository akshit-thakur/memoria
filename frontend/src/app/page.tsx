// ─── app/page (Home) ────────────────────────────────────────────────────────
// In plain English: the entire application as one component. Layout is just
// a flex row: Sidebar on the left, one of eight views in the middle (Library,
// Search, Duplicates, Faces, Blurry, Map, Dashboard, Settings, Database), and the InfoPanel overlaid as a
// modal portal when a photo is selected.
//
// Every section is wrapped in its own ErrorBoundary so a crash in one view
// can't take down the others.
//
"use client";

import { useState, useCallback } from "react";
import Sidebar, { type ViewKey } from "@/components/Sidebar";
import PhotoGrid from "@/components/PhotoGrid";
import InfoPanel from "@/components/InfoPanel";
import DuplicateGroups from "@/components/DuplicateGroups";
import FaceGrid from "@/components/FaceGrid";
import BlurryList from "@/components/BlurryList";
import LocationMap from "@/components/LocationMap";
import SearchView from "@/components/SearchView";
import SettingsView from "@/components/SettingsView";
import DashboardView from "@/components/DashboardView";
import DiagnosticsDrawer from "@/components/DiagnosticsDrawer";
import DatabaseVisualizer from "@/components/DatabaseVisualizer";
import ErrorBoundary from "@/components/ErrorBoundary";
import { useMedia } from "@/hooks/useMedia";
import type { MediaItem } from "@/types";

/**
 * The application root. Owns three pieces of state:
 *   • `selectedId` — which photo (if any) is open in the InfoPanel.
 *   • `activeView` — which of the nine top-level views is showing.
 *   • `diagnosticsOpen` — whether live system telemetry is expanded.
 * Everything else comes from the useMedia hook or query-cache.
 */
export default function Home() {
  const { items, fetchNext, removeItem, newItemsAvailable, refreshLibrary } = useMedia();
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [activeView, setActiveView] = useState<ViewKey>("library");
  const [diagnosticsOpen, setDiagnosticsOpen] = useState(false);

  const handleItemClick = useCallback((item: MediaItem) => {
    setSelectedId(item.id);
  }, []);

  const handleMediaIdClick = useCallback((mediaId: number) => {
    setSelectedId(mediaId);
  }, []);

  const handleClose = useCallback(() => {
    setSelectedId(null);
  }, []);

  const handleDiagnosticsOpen = useCallback(() => {
    setDiagnosticsOpen(true);
  }, []);

  const handleDiagnosticsClose = useCallback(() => {
    setDiagnosticsOpen(false);
  }, []);

  return (
    <div className="flex h-screen bg-black text-white">
      <ErrorBoundary label="Sidebar">
        <Sidebar
          activeView={activeView}
          onNavigate={setActiveView}
          onDiagnosticsOpen={handleDiagnosticsOpen}
        />
      </ErrorBoundary>

      {activeView === "library" && (
        <ErrorBoundary label="Library">
          <div className="relative flex-1 flex flex-col min-w-0">
            {newItemsAvailable && (
              <div className="absolute top-3 left-1/2 -translate-x-1/2 z-20">
                <button
                  onClick={refreshLibrary}
                  className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium rounded-full shadow-lg transition-colors"
                >
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h5M20 20v-5h-5M4 4l4.5 4.5A9 9 0 1020 12" />
                  </svg>
                  New photos added — refresh
                </button>
              </div>
            )}
            <PhotoGrid items={items} onItemClick={handleItemClick} onItemError={removeItem} onLoadMore={fetchNext} />
          </div>
        </ErrorBoundary>
      )}
      {activeView === "search" && (
        <ErrorBoundary label="Search">
          <SearchView onItemClick={handleItemClick} />
        </ErrorBoundary>
      )}
      {activeView === "duplicates" && (
        <ErrorBoundary label="Duplicates">
          <DuplicateGroups onItemClick={handleItemClick} />
        </ErrorBoundary>
      )}
      {activeView === "faces" && (
        <ErrorBoundary label="Faces">
          <FaceGrid onItemClick={handleMediaIdClick} />
        </ErrorBoundary>
      )}
      {activeView === "blurry" && (
        <ErrorBoundary label="Blurry">
          <BlurryList onItemClick={handleMediaIdClick} />
        </ErrorBoundary>
      )}
      {activeView === "map" && (
        <ErrorBoundary label="Map">
          <LocationMap />
        </ErrorBoundary>
      )}
      {activeView === "dashboard" && (
        <ErrorBoundary label="Dashboard">
          <DashboardView />
        </ErrorBoundary>
      )}
      {activeView === "settings" && (
        <ErrorBoundary label="Settings">
          <SettingsView />
        </ErrorBoundary>
      )}
      {activeView === "database" && (
        <ErrorBoundary label="Database map">
          <DatabaseVisualizer />
        </ErrorBoundary>
      )}

      <ErrorBoundary label="Info Panel">
        <InfoPanel selectedId={selectedId} items={items} onClose={handleClose} />
      </ErrorBoundary>

      <ErrorBoundary label="System Diagnostics">
        <DiagnosticsDrawer open={diagnosticsOpen} onClose={handleDiagnosticsClose} />
      </ErrorBoundary>
    </div>
  );
}
