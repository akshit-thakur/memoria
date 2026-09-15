"use client";

import { useState } from "react";
import {
  Activity,
  Database,
  ChevronLeft,
  ChevronRight,
  CircleSlash,
  Copy,
  Image,
  LayoutDashboard,
  MapPin,
  Search,
  Settings,
  Users,
  type LucideIcon,
} from "lucide-react";
import { useStats } from "@/hooks/useStats";

export type ViewKey =
  | "library"
  | "search"
  | "duplicates"
  | "faces"
  | "blurry"
  | "map"
  | "dashboard"
  | "settings"
  | "database";

interface NavItem {
  key: ViewKey;
  label: string;
  Icon: LucideIcon;
}

const NAV_ITEMS: NavItem[] = [
  { key: "library", label: "Library", Icon: Image },
  { key: "search", label: "Search", Icon: Search },
  { key: "duplicates", label: "Image Deduplication", Icon: Copy },
  { key: "faces", label: "Faces", Icon: Users },
  { key: "blurry", label: "Blurry Photos", Icon: CircleSlash },
  { key: "map", label: "Location Map", Icon: MapPin },
  { key: "dashboard", label: "Dashboard", Icon: LayoutDashboard },
  { key: "settings", label: "Settings", Icon: Settings },
  { key: "database", label: "Database map", Icon: Database },
];

interface SidebarProps {
  activeView: ViewKey;
  onNavigate: (view: ViewKey) => void;
  onDiagnosticsOpen: () => void;
}

// A radius of 15.9 in the 36 px view box produces a circumference of ~100,
// so the completed percentage maps directly to the SVG dash length.
const RING_CIRCUMFERENCE = 100;

function IndexingStatusPill({ onOpen }: { onOpen: () => void }) {
  const { data: stats, isError } = useStats();
  const reindexStatus = stats?.scene_reindex?.status;
  const reindexActive = reindexStatus
    ? ["queued", "running", "indexing", "cancel_requested"].includes(reindexStatus)
    : false;
  const active = Boolean(
    stats && (
      stats.queue_size > 0
      || Boolean(stats.current_file)
      || stats.face_scan_active
      || stats.blur_scan_active
      || stats.scene_scan_active
      || reindexActive
    ),
  );

  const reindexTotal = stats?.scene_reindex?.total_items ?? 0;
  const reindexDone = (stats?.scene_reindex?.processed_items ?? 0)
    + (stats?.scene_reindex?.skipped_items ?? 0);
  const queueTotal = (stats?.total_indexed ?? 0) + (stats?.queue_size ?? 0);
  const progress = reindexActive && reindexTotal > 0
    ? Math.max(4, Math.min(100, Math.round((reindexDone / reindexTotal) * 100)))
    : queueTotal > 0
      ? Math.max(4, Math.min(100, Math.round(((stats?.total_indexed ?? 0) / queueTotal) * 100)))
      : 20;

  const remaining = reindexActive
    ? Math.max(0, reindexTotal - reindexDone)
    : stats?.queue_size ?? 0;
  const label = !stats
    ? isError ? "Status unavailable" : "Checking library…"
    : active
      ? remaining > 0
        ? `Syncing ${remaining.toLocaleString()} ${remaining === 1 ? "photo" : "photos"}…`
        : "Processing library…"
      : "Library Synced";

  return (
    <button
      type="button"
      onClick={onOpen}
      className="mt-4 flex w-full items-center gap-2 rounded-full border border-neutral-800 bg-neutral-900/80 px-3 py-2 text-left transition-colors hover:border-indigo-500/60 hover:bg-neutral-800/80"
      title="Open system diagnostics"
      aria-label={`${label} Open system diagnostics`}
    >
      <svg
        viewBox="0 0 36 36"
        className={`h-7 w-7 shrink-0 -rotate-90 ${active ? "animate-spin" : ""}`}
        aria-hidden="true"
      >
        <circle cx="18" cy="18" r="15.9" fill="none" stroke="#374151" strokeWidth="2.5" />
        <circle
          cx="18"
          cy="18"
          r="15.9"
          fill="none"
          stroke={!stats ? "#737373" : active ? "#818cf8" : "#34d399"}
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeDasharray={`${active ? progress : 100} ${RING_CIRCUMFERENCE}`}
        />
      </svg>
      <span className="min-w-0 flex-1 truncate text-xs font-medium text-neutral-200">
        {label}
      </span>
      <Activity size={14} className={isError ? "text-red-400" : active ? "text-indigo-300" : stats ? "text-emerald-400" : "text-neutral-500"} />
    </button>
  );
}

export default function Sidebar({ activeView, onNavigate, onDiagnosticsOpen }: SidebarProps) {
  const [collapsed, setCollapsed] = useState(false);

  if (collapsed) {
    return (
      <aside
        aria-label="Application navigation"
        className="flex w-12 shrink-0 flex-col items-center gap-1 border-r border-neutral-800 bg-neutral-950 pt-3"
      >
        <button
          type="button"
          onClick={() => setCollapsed(false)}
          title="Expand sidebar"
          className="mb-2 text-neutral-400 hover:text-white"
        >
          <ChevronRight size={16} />
        </button>
        {NAV_ITEMS.map(({ key, label: navLabel, Icon }) => (
          <button
            type="button"
            key={key}
            title={navLabel}
            aria-label={navLabel}
            onClick={() => onNavigate(key)}
            className={`rounded-lg p-2 transition-colors ${activeView === key
              ? "bg-neutral-800 text-white"
              : "text-neutral-500 hover:bg-neutral-800/50 hover:text-white"
            }`}
          >
            <Icon size={18} />
          </button>
        ))}
        <button
          type="button"
          title="Open diagnostics"
          aria-label="Open system diagnostics"
          onClick={onDiagnosticsOpen}
          className="mt-2 rounded-lg p-2 text-neutral-500 hover:bg-neutral-800/50 hover:text-white"
        >
          <Activity size={18} />
        </button>
      </aside>
    );
  }

  return (
    <aside
      aria-label="Application navigation"
      className="flex w-64 shrink-0 flex-col overflow-y-auto border-r border-neutral-800 bg-neutral-950 p-3"
    >
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-neutral-500">
          Navigation
        </h2>
        <button
          type="button"
          onClick={() => setCollapsed(true)}
          title="Collapse sidebar"
          className="text-neutral-400 hover:text-white"
        >
          <ChevronLeft size={16} />
        </button>
      </div>

      <nav className="flex-1">
        <ul className="space-y-1">
          {NAV_ITEMS.map(({ key, label: navLabel, Icon }) => (
            <li key={key}>
              <button
                type="button"
                onClick={() => onNavigate(key)}
                className={`flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors ${activeView === key
                  ? "bg-neutral-800 font-medium text-white"
                  : "text-neutral-400 hover:bg-neutral-800/50 hover:text-white"
                }`}
              >
                <Icon size={18} />
                <span>{navLabel}</span>
              </button>
            </li>
          ))}
        </ul>
      </nav>

      <IndexingStatusPill onOpen={onDiagnosticsOpen} />
    </aside>
  );
}
