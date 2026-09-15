"use client";

import { useQuery } from "@tanstack/react-query";
import { useEffect } from "react";
import { Activity, AlertTriangle, Gauge, MemoryStick, X } from "lucide-react";
import { apiUrl } from "@/lib/api";
import { ENDPOINTS } from "@/lib/endpoints";
import type {
  DiagnosticsMetrics,
  DiagnosticStageKey,
  DiagnosticStageMetrics,
  IndexingStats,
  MetricSeries,
  MetricsSnapshot,
  ProcessingRunsResponse,
  QueueStats,
  ResourceAllocation,
} from "@/types";

interface DiagnosticsDrawerProps {
  open: boolean;
  onClose: () => void;
}

function findMetric(
  metrics: MetricsSnapshot | undefined,
  name: string,
): MetricSeries | undefined {
  const counters = metrics?.combined?.counters ?? {};
  const histograms = metrics?.combined?.histograms ?? {};
  const matchesName = ([key]: [string, MetricSeries]) => key.split("|")[0] === name;
  const found = Object.entries(counters).find(matchesName);
  if (found) return found[1];
  const histogram = Object.entries(histograms).find(matchesName);
  return histogram?.[1];
}

function metricNumber(series: MetricSeries | undefined, fallback = 0): number {
  const value = series?.value ?? series?.mean ?? series?.max;
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

const PIPELINE_STAGES: DiagnosticStageKey[] = [
  "thumbnail",
  "exif",
  "phash",
  "faces",
  "scene",
];

function stageMetrics(
  metrics: MetricsSnapshot | undefined,
  stage: DiagnosticStageKey,
): DiagnosticStageMetrics {
  const series = findMetric(metrics, `worker.${stage}`);
  return {
    samples: series?.n ?? 0,
    mean_ms: metricNumber(series),
    p95_ms: series?.p95 ?? 0,
  };
}

function buildResourceMetrics(
  stats: IndexingStats | undefined,
  metrics: MetricsSnapshot | undefined,
): DiagnosticsMetrics["system"] {
  let vram: ResourceAllocation | null = null;
  if (stats?.vram_usage?.driver_mb) {
    vram = {
      used_mb: stats.vram_usage.allocated_mb,
      total_mb: stats.vram_usage.driver_mb,
      percent: (stats.vram_usage.allocated_mb / stats.vram_usage.driver_mb) * 100,
      backend: stats.vram_usage.backend,
    };
  } else {
    const usedMb = metricNumber(findMetric(metrics, "worker.system.gpu_used_mb"));
    const totalMb = metricNumber(findMetric(metrics, "worker.system.gpu_total_mb"));
    const percent = totalMb > 0
      ? (usedMb / totalMb) * 100
      : metricNumber(findMetric(metrics, "worker.system.gpu_mem_util_pct"));
    if (usedMb > 0 || totalMb > 0 || percent > 0) {
      vram = { used_mb: usedMb, total_mb: totalMb || null, percent, backend: "gpu" };
    }
  }

  const fallbackRamPercent = metricNumber(findMetric(metrics, "worker.system.ram_pct"));
  const fallbackRamUsedMb = metricNumber(findMetric(metrics, "worker.system.ram_used_gb")) * 1024;
  const ram = stats?.ram_usage
    ? {
        used_mb: stats.ram_usage.used_mb,
        total_mb: stats.ram_usage.total_mb,
        percent: stats.ram_usage.percent,
        backend: "system",
      }
    : fallbackRamPercent > 0 || fallbackRamUsedMb > 0
      ? {
          used_mb: fallbackRamUsedMb,
          total_mb: fallbackRamPercent > 0
            ? fallbackRamUsedMb / (fallbackRamPercent / 100)
            : null,
          percent: fallbackRamPercent,
          backend: "system",
        }
      : null;

  return { vram, ram };
}

function GaugeBar({ value, color = "bg-indigo-500" }: { value: number; color?: string }) {
  const pct = Math.max(0, Math.min(100, value));
  return (
    <div className="h-2 rounded-full bg-neutral-800 overflow-hidden">
      <div className={`h-full ${color} transition-all`} style={{ width: `${pct}%` }} />
    </div>
  );
}

function resourceLabel(resource: ResourceAllocation | null): string {
  if (!resource) return "Unavailable";
  if (resource.total_mb) {
    return `${Math.round(resource.used_mb).toLocaleString()} / ${Math.round(resource.total_mb).toLocaleString()} MB`;
  }
  return `${resource.percent.toFixed(0)}%`;
}

function StatCard({ label, value, detail }: { label: string; value: string; detail?: string }) {
  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900/70 p-3">
      <div className="text-[11px] uppercase tracking-wider text-neutral-500">{label}</div>
      <div className="mt-1 text-lg font-semibold text-neutral-100">{value}</div>
      {detail && <div className="mt-1 text-xs text-neutral-500">{detail}</div>}
    </div>
  );
}

export default function DiagnosticsDrawer({ open, onClose }: DiagnosticsDrawerProps) {
  useEffect(() => {
    if (!open) return;
    const previousOverflow = document.body.style.overflow;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [open, onClose]);

  const { data, isFetching, error } = useQuery<{
    stats: IndexingStats;
    queue: QueueStats;
    metrics: MetricsSnapshot;
    throughput: ProcessingRunsResponse;
  }>({
    queryKey: ["diagnostics", "live"],
    enabled: open,
    queryFn: async ({ signal }) => {
      const requests = [
        fetch(apiUrl(ENDPOINTS.stats), { signal }),
        fetch(apiUrl(ENDPOINTS.queueStats), { signal }),
        fetch(apiUrl(ENDPOINTS.metrics), { signal }),
        fetch(apiUrl(ENDPOINTS.processingRuns(1)), { signal }),
      ];
      const [statsResponse, queueResponse, metricsResponse, throughputResponse] = await Promise.all(requests);
      for (const response of [statsResponse, queueResponse, metricsResponse, throughputResponse]) {
        if (!response.ok) throw new Error(`Diagnostics HTTP ${response.status}`);
      }
      return {
        stats: (await statsResponse.json()) as IndexingStats,
        queue: (await queueResponse.json()) as QueueStats,
        metrics: (await metricsResponse.json()) as MetricsSnapshot,
        throughput: (await throughputResponse.json()) as ProcessingRunsResponse,
      };
    },
    refetchInterval: open ? 5_000 : false,
    refetchIntervalInBackground: false,
  });

  const stats = data?.stats;
  const metrics = data?.metrics;
  // Delayed retries live in Huey's `schedule` table rather than `task`.
  // Present both as pending work so diagnostics never claims the queue is
  // empty while a retry is waiting for its due time.
  const pending = data?.queue
    ? data.queue.pending + (data.queue.scheduled ?? 0)
    : stats?.queue_size ?? 0;
  const running = data?.queue.running ?? (stats?.current_file ? 1 : 0);
  const queueTotal = pending + running;
  const latestRun = data?.throughput.latest;
  const diagnostics: DiagnosticsMetrics = {
    throughput: {
      photos_per_second: latestRun && latestRun.total_sec > 0
        ? latestRun.files / latestRun.total_sec
        : 0,
      processed_photos: metricNumber(findMetric(metrics, "worker.batch.photos_ok")),
    },
    queue: {
      total: queueTotal,
      pending,
      running,
      failed: data?.queue.failed ?? 0,
    },
    stages: Object.fromEntries(
      PIPELINE_STAGES.map((stage) => [stage, stageMetrics(metrics, stage)]),
    ) as Record<DiagnosticStageKey, DiagnosticStageMetrics>,
    system: buildResourceMetrics(stats, metrics),
  };

  return (
    <div className={`fixed inset-0 z-50 ${open ? "pointer-events-auto" : "pointer-events-none"}`} aria-hidden={!open}>
      <button
        type="button"
        aria-label="Close diagnostics"
        onClick={onClose}
        className={`absolute inset-0 bg-black/60 transition-opacity ${open ? "opacity-100" : "opacity-0"}`}
      />
      <aside
        role="dialog"
        aria-modal="true"
        aria-label="System diagnostics"
        className={`absolute right-0 top-0 h-full w-full max-w-md overflow-y-auto border-l border-neutral-800 bg-neutral-950 p-5 shadow-2xl transition-transform duration-200 ${open ? "translate-x-0" : "translate-x-full"}`}
      >
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2 text-sm font-semibold text-neutral-100">
              <Activity size={16} className="text-indigo-400" /> System diagnostics
            </div>
            <p className="mt-1 text-xs text-neutral-500">Live worker, queue, and resource telemetry</p>
          </div>
          <button type="button" onClick={onClose} className="rounded-md p-1.5 text-neutral-500 hover:bg-neutral-800 hover:text-white" aria-label="Close">
            <X size={18} />
          </button>
        </div>

        {error ? (
          <div className="mt-6 flex gap-2 rounded-lg border border-red-900/60 bg-red-950/30 p-3 text-sm text-red-300">
            <AlertTriangle size={16} className="shrink-0" /> Unable to load live diagnostics.
          </div>
        ) : (
          <>
            <div className="mt-5 grid grid-cols-2 gap-2">
              <StatCard label="Throughput" value={diagnostics.throughput.photos_per_second > 0 ? `${diagnostics.throughput.photos_per_second.toFixed(2)} / sec` : "—"} detail={`${Math.round(diagnostics.throughput.processed_photos).toLocaleString()} photos processed`} />
              <StatCard label="Queue depth" value={diagnostics.queue.total.toLocaleString()} detail={isFetching ? "Refreshing…" : "Live snapshot"} />
            </div>

            <section className="mt-6 space-y-3">
              <h2 className="text-xs font-semibold uppercase tracking-wider text-neutral-500">Queue breakdown</h2>
              <div className="grid grid-cols-3 gap-2 text-center text-xs">
                <StatCard label="Pending" value={diagnostics.queue.pending.toLocaleString()} />
                <StatCard label="Running" value={diagnostics.queue.running.toLocaleString()} />
                <StatCard label="Failed" value={diagnostics.queue.failed.toLocaleString()} />
              </div>
            </section>

            <section className="mt-6 space-y-3">
              <h2 className="text-xs font-semibold uppercase tracking-wider text-neutral-500">Pipeline stages</h2>
              <div className="space-y-2">
                {PIPELINE_STAGES.map((stage) => {
                  const stageData = diagnostics.stages[stage];
                  return (
                    <div key={stage} className="flex items-center justify-between rounded-md border border-neutral-800 bg-neutral-900/50 px-3 py-2 text-xs">
                      <span className="capitalize text-neutral-300">{stage === "phash" ? "pHash" : stage === "faces" ? "Face detection" : stage === "scene" ? "Scene embedding" : stage}</span>
                      <span className="text-neutral-500">{stageData.p95_ms > 0 ? `p95 ${stageData.p95_ms.toFixed(1)} ms` : `${stageData.samples} samples`}</span>
                    </div>
                  );
                })}
              </div>
            </section>

            <section className="mt-6 space-y-4">
              <h2 className="text-xs font-semibold uppercase tracking-wider text-neutral-500">System resources</h2>
              <div className="rounded-lg border border-neutral-800 bg-neutral-900/60 p-3 space-y-3">
                <div className="flex items-center gap-2 text-xs text-neutral-300"><Gauge size={14} className="text-emerald-400" /> VRAM <span className="ml-auto text-neutral-500">{resourceLabel(diagnostics.system.vram)}</span></div>
                <GaugeBar value={diagnostics.system.vram?.percent ?? 0} color="bg-emerald-500" />
                <div className="flex items-center gap-2 text-xs text-neutral-300"><MemoryStick size={14} className="text-sky-400" /> RAM <span className="ml-auto text-neutral-500">{resourceLabel(diagnostics.system.ram)}</span></div>
                <GaugeBar value={diagnostics.system.ram?.percent ?? 0} color="bg-sky-500" />
              </div>
            </section>
          </>
        )}
      </aside>
    </div>
  );
}
