// ─── DashboardView ──────────────────────────────────────────────────────────
// In plain English: a full-page dashboard of indexing progress.
// Polls /stats (counts), /processing/runs (throughput), and /progress
// (stall detection + ETA). Shows a 4-state badge: Scanning / Paused—memory /
// Stalled / Idle, plus an ingest progress card with percent + ETA.
//
"use client";

import {
    Image as ImageIcon,
    Users,
    HardDrive,
    Gauge,
    CircleSlash,
    Search as SearchIcon,
    Cpu,
    ListChecks,
    AlertTriangle,
    type LucideIcon,
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { apiUrl } from "@/lib/api";
import { ENDPOINTS } from "@/lib/endpoints";
import { useStats } from "@/hooks/useStats";
import { formatFileSize } from "@/lib/formatters";

interface RunRecord {
    files_per_hour: number;
    gb_per_hour: number;
}

interface ProgressData {
    state: "idle" | "scanning" | "paused_memory" | "stalled";
    reasons: string[];
    session: {
        id: string | null;
        started_at: string | null;
        finished_at: string | null;
        total: number;
        done: number;
        percent: number;
    };
    queue: { tasks: number };
    worker: {
        alive: boolean;
        current_file: string;
        current_file_age_s: number | null;
        last_completed_age_s: number | null;
        paused_memory: boolean;
        pressure: string;
    };
    rate: {
        files_per_min: number | null;
        eta_seconds: number | null;
    };
}

function formatEta(seconds: number | null): string {
    if (seconds == null || seconds <= 0) return "";
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
    const h = Math.floor(seconds / 3600);
    const m = Math.round((seconds % 3600) / 60);
    return m > 0 ? `${h}h ${m}m` : `${h}h`;
}

function formatAge(seconds: number | null): string {
    if (seconds == null) return "—";
    if (seconds < 60) return `${Math.round(seconds)}s ago`;
    if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
    return `${Math.round(seconds / 3600)}h ago`;
}

export default function DashboardView() {
    const { data: stats } = useStats();

    const { data: progressData } = useQuery<ProgressData>({
        queryKey: ["progress"],
        queryFn: async ({ signal }) => {
            const res = await fetch(apiUrl(ENDPOINTS.progress), { signal });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            return res.json();
        },
        staleTime: 0,
        refetchInterval: (query) =>
            query.state.data?.state === "idle" ? 30_000 : 5_000,
        refetchIntervalInBackground: false,
    });

    const { data: runsData } = useQuery({
        queryKey: ["processing-runs-dashboard"],
        queryFn: async ({ signal }) => {
            const res = await fetch(apiUrl(ENDPOINTS.processingRuns(1)), { signal });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            return res.json();
        },
        staleTime: 0,
        refetchInterval: progressData?.state === "idle" ? 30_000 : 5_000,
        refetchIntervalInBackground: false,
    });

    const progress: ProgressData | null = progressData ?? null;
    const latest: RunRecord | null = runsData?.latest ?? null;

    if (!stats) {
        return (
            <div className="flex-1 flex items-center justify-center text-neutral-500">
                Loading dashboard…
            </div>
        );
    }

    const totalIndexed = stats.total_indexed ?? 0;
    const facesTotal = stats.faces_total ?? 0;
    const faceClusters = stats.face_clusters ?? 0;
    const facesIndexed = stats.faces_indexed ?? 0;
    const blurScored = stats.blur_scored ?? 0;
    const blurFlagged = stats.blur_flagged ?? 0;
    const sceneIndexed = stats.scene_indexed ?? 0;
    const queueSize = stats.queue_size ?? 0;
    const vram = stats.vram_usage ?? null;

    const filesPerHour =
        latest && Number.isFinite(latest.files_per_hour) && latest.files_per_hour > 0
            ? latest.files_per_hour
            : null;
    const gbPerHour =
        latest && Number.isFinite(latest.gb_per_hour) && latest.gb_per_hour > 0
            ? latest.gb_per_hour
            : null;
    const hasSpeed = filesPerHour != null || gbPerHour != null;

    const currentDir = stats.current_file
        ? stats.current_file.split("/").slice(0, -1).join("/")
        : null;

    const ingestTotal = progress?.session.total ?? 0;
    const ingestDone = progress?.session.done ?? 0;
    const ingestPct = progress?.session.percent ?? 0;
    const filesPerMin = progress?.rate.files_per_min ?? null;
    const etaSeconds = progress?.rate.eta_seconds ?? null;
    const lastCompletedAge = progress?.worker.last_completed_age_s ?? null;
    const stuckFile = progress?.worker.current_file ?? "";
    const stallReasons = progress?.reasons ?? [];
    const progressState = progress?.state ?? "idle";

    return (
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
            <div className="flex items-center justify-between gap-3">
                <div>
                    <h1 className="text-xl font-semibold">Dashboard</h1>
                    <p className="text-sm text-neutral-400 mt-1">
                        Live overview of indexing progress and processing throughput.
                    </p>
                </div>
                <ScanBadge state={progressState} />
            </div>

            {/* Ingest progress (shown when a session has ever run) */}
            {ingestTotal > 0 && (
                <section>
                    <h2 className="text-sm font-medium text-neutral-300 mb-2">Ingest progress</h2>
                    <div className="rounded-lg bg-neutral-900/60 border border-neutral-800 p-4 space-y-3">
                        <div className="flex items-baseline justify-between gap-2">
                            <span className="text-2xl font-bold tabular-nums">
                                {ingestDone.toLocaleString()}
                                <span className="text-neutral-500 text-base font-normal">
                                    {" / "}{ingestTotal.toLocaleString()}
                                </span>
                            </span>
                            <span className="text-sm font-semibold text-neutral-300 tabular-nums">
                                {ingestPct}%
                            </span>
                        </div>
                        <div className="h-2 w-full rounded-full bg-neutral-800 overflow-hidden">
                            <div
                                className="h-full bg-blue-500 transition-all duration-500 ease-out"
                                style={{ width: `${ingestPct}%` }}
                            />
                        </div>
                        <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-neutral-500">
                            {filesPerMin != null && (
                                <span>
                                    <span className="text-neutral-300">{Math.round(filesPerMin * 60).toLocaleString()}</span> files/hr
                                </span>
                            )}
                            {etaSeconds != null && etaSeconds > 0 && (
                                <span>
                                    ETA <span className="text-neutral-300">{formatEta(etaSeconds)}</span>
                                </span>
                            )}
                            {lastCompletedAge != null && (
                                <span>
                                    Last completion <span className="text-neutral-300">{formatAge(lastCompletedAge)}</span>
                                </span>
                            )}
                        </div>
                        {progressState === "stalled" && stallReasons.length > 0 && (
                            <div className="flex items-start gap-2 text-xs text-red-400 bg-red-950/40 border border-red-900/50 rounded px-3 py-2">
                                <AlertTriangle size={13} className="mt-0.5 shrink-0" />
                                <div>
                                    <span className="font-medium">Stalled: </span>
                                    {stallReasons.join(", ")}
                                    {stuckFile && (
                                        <p className="text-red-500/80 mt-0.5 truncate" title={stuckFile}>
                                            {stuckFile.split("/").slice(-2).join("/")}
                                        </p>
                                    )}
                                </div>
                            </div>
                        )}
                    </div>
                </section>
            )}

            {/* Headline metrics */}
            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
                <StatCard
                    Icon={ImageIcon}
                    color="text-sky-400"
                    label="Images scanned"
                    value={totalIndexed.toLocaleString()}
                    sub={queueSize > 0 ? `${queueSize.toLocaleString()} queued` : "up to date"}
                />
                <StatCard
                    Icon={Users}
                    color="text-indigo-400"
                    label="Faces detected"
                    value={facesTotal.toLocaleString()}
                    sub={`${faceClusters.toLocaleString()} clusters`}
                />
                <StatCard
                    Icon={HardDrive}
                    color="text-emerald-400"
                    label="Size scanned"
                    value={formatFileSize(stats.total_bytes_indexed)}
                    sub="on-disk total"
                />
                <StatCard
                    Icon={Gauge}
                    color="text-amber-400"
                    label="Speed"
                    value={filesPerHour != null ? `${Math.round(filesPerHour).toLocaleString()}/hr` : "—"}
                    sub={
                        hasSpeed
                            ? gbPerHour != null
                                ? `${gbPerHour.toFixed(2)} GB/hr`
                                : "files/hr"
                            : "no recent run"
                    }
                />
            </div>

            {/* Per-feature progress */}
            <section className="space-y-2">
                <h2 className="text-sm font-medium text-neutral-300">Enrichment progress</h2>
                <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
                    <ProgressCard
                        Icon={Users}
                        color="text-indigo-400"
                        label="Face detection"
                        active={!!stats.face_scan_active}
                        done={facesIndexed}
                        total={totalIndexed}
                        extra={`${facesTotal.toLocaleString()} faces · ${faceClusters.toLocaleString()} clusters`}
                    />
                    <ProgressCard
                        Icon={CircleSlash}
                        color="text-amber-400"
                        label="Blur detection"
                        active={!!stats.blur_scan_active}
                        done={blurScored}
                        total={totalIndexed}
                        extra={`${blurFlagged.toLocaleString()} flagged blurry`}
                    />
                    <ProgressCard
                        Icon={SearchIcon}
                        color="text-purple-400"
                        label="Scene embeddings"
                        active={!!stats.scene_scan_active}
                        done={sceneIndexed}
                        total={totalIndexed}
                    />
                </div>
            </section>

            {/* System status */}
            <section className="space-y-2">
                <h2 className="text-sm font-medium text-neutral-300">System</h2>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div className="rounded-lg bg-neutral-900/60 border border-neutral-800 p-4">
                        <div className="flex items-center gap-2 mb-2">
                            <ListChecks size={15} className="text-sky-400 shrink-0" />
                            <span className="text-xs font-semibold uppercase tracking-wider text-neutral-500">
                                Queue
                            </span>
                        </div>
                        <p className="text-sm text-neutral-300">
                            <span className="text-neutral-100 font-medium">{queueSize.toLocaleString()}</span> tasks pending
                        </p>
                        {currentDir && (
                            <p className="text-xs text-indigo-400 mt-2 truncate" title={currentDir}>
                                <span className="inline-block w-1.5 h-1.5 rounded-full bg-indigo-400 animate-pulse mr-1 align-middle" />
                                Scanning {currentDir}
                            </p>
                        )}
                    </div>
                    <div className="rounded-lg bg-neutral-900/60 border border-neutral-800 p-4">
                        <div className="flex items-center gap-2 mb-2">
                            <Cpu size={15} className="text-emerald-400 shrink-0" />
                            <span className="text-xs font-semibold uppercase tracking-wider text-neutral-500">
                                GPU Memory
                            </span>
                            {vram && (
                                <span className="ml-auto text-[10px] uppercase tracking-wider text-neutral-500">
                                    {vram.backend}
                                </span>
                            )}
                        </div>
                        {vram ? (
                            <>
                                <div className="h-1.5 w-full rounded-full bg-neutral-800 overflow-hidden">
                                    <div
                                        className="h-full bg-emerald-500"
                                        style={{
                                            width: `${vram.driver_mb > 0 ? Math.min(100, Math.round((vram.allocated_mb / vram.driver_mb) * 100)) : 0}%`,
                                            transition: "width 0.5s ease",
                                        }}
                                    />
                                </div>
                                <p className="text-xs text-neutral-400 mt-2">
                                    <span className="text-neutral-200 font-medium">
                                        {vram.allocated_mb.toLocaleString()}
                                    </span>{" "}
                                    / {vram.driver_mb.toLocaleString()} MB
                                </p>
                            </>
                        ) : (
                            <p className="text-sm text-neutral-500">Not available</p>
                        )}
                    </div>
                </div>
            </section>
        </div>
    );
}

// ── State badge ────────────────────────────────────────────────────────────────

function ScanBadge({ state }: { state: string }) {
    if (state === "scanning") {
        return (
            <span className="flex items-center gap-2 text-xs text-green-400">
                <span className="inline-block w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                Scanning
            </span>
        );
    }
    if (state === "paused_memory") {
        return (
            <span className="flex items-center gap-2 text-xs text-amber-400">
                <span className="inline-block w-2 h-2 rounded-full bg-amber-400" />
                Paused — memory pressure
            </span>
        );
    }
    if (state === "stalled") {
        return (
            <span className="flex items-center gap-2 text-xs text-red-400">
                <span className="inline-block w-2 h-2 rounded-full bg-red-500" />
                Stalled
            </span>
        );
    }
    return (
        <span className="flex items-center gap-2 text-xs text-neutral-500">
            <span className="inline-block w-2 h-2 rounded-full bg-neutral-600" />
            Idle
        </span>
    );
}

// ── Cards ─────────────────────────────────────────────────────────────────────

function StatCard({
    Icon,
    color,
    label,
    value,
    sub,
}: {
    Icon: LucideIcon;
    color: string;
    label: string;
    value: string;
    sub?: string;
}) {
    return (
        <div className="rounded-lg bg-neutral-900/60 border border-neutral-800 p-4">
            <div className="flex items-center gap-2 mb-2">
                <Icon size={15} className={`${color} shrink-0`} />
                <span className="text-xs font-semibold uppercase tracking-wider text-neutral-500">
                    {label}
                </span>
            </div>
            <p className="text-2xl font-bold text-white tabular-nums">{value}</p>
            {sub && <p className="text-xs text-neutral-500 mt-1">{sub}</p>}
        </div>
    );
}

function ProgressCard({
    Icon,
    color,
    label,
    active,
    done,
    total,
    extra,
}: {
    Icon: LucideIcon;
    color: string;
    label: string;
    active: boolean;
    done: number;
    total: number;
    extra?: string;
}) {
    const pct = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0;
    return (
        <div className="rounded-lg bg-neutral-900/60 border border-neutral-800 p-4">
            <div className="flex items-center gap-2 mb-2">
                <Icon size={15} className={`${color} shrink-0`} />
                <span className="text-xs font-semibold uppercase tracking-wider text-neutral-500">
                    {label}
                </span>
                {active && (
                    <span
                        className="inline-block w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse ml-auto"
                        title="Scanning…"
                    />
                )}
            </div>
            <div className="h-1.5 w-full rounded-full bg-neutral-800 overflow-hidden">
                <div
                    className="h-full bg-neutral-400"
                    style={{ width: `${pct}%`, transition: "width 0.5s ease" }}
                />
            </div>
            <p className="text-xs text-neutral-400 mt-2">
                <span className="text-neutral-200 font-medium">{done.toLocaleString()}</span> /{" "}
                {total.toLocaleString()} photos
                <span className="text-neutral-500"> · {pct}%</span>
            </p>
            {extra && <p className="text-xs text-neutral-500 mt-0.5">{extra}</p>}
        </div>
    );
}
