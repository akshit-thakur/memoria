// ─── SettingsView ───────────────────────────────────────────────────────────
// In plain English: the "Settings" tab. Lets you pick a Memoria mode preset
// (Dedup-first, Full enrichment, Semantic search only, Face indexing only),
// browse and scan folders one at a time, and manage browser-retained power
// controls. Pipeline stage toggles are fixed by the selected preset.
//
// Two verification aids live here:
//   • A "Latest run" card (from the persistent throughput log) so you can
//     confirm the worker actually processed with the mode you saved.
//   • A Dedup-first warning explaining what that mode skips.
//
"use client";

import { useQuery, useQueryClient, useMutation } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { AlertTriangle, ChevronDown, ChevronUp, Loader2, RefreshCw, SlidersHorizontal, Trash2 } from "lucide-react";
import { apiUrl } from "@/lib/api";
import { LOCAL_STORAGE_KEYS } from "@/lib/constants";
import { ENDPOINTS } from "@/lib/endpoints";
import { BlobPool } from "@/hooks/useBlobPool";
import { useVisibilityInterval } from "@/hooks/useVisibilityInterval";
import LibraryFolderPicker from "@/components/LibraryFolderPicker";
import type {
  ModelBenchmark,
  ProcessingConfig,
  ProcessingRunsResponse,
  ReindexJob,
  UserSettings,
} from "@/types";

const PRESET_ORDER: { key: string; label: string }[] = [
  { key: "dedup_first", label: "Dedup-first" },
  { key: "full", label: "Full enrichment" },
  { key: "semantic_search", label: "Semantic search only" },
  { key: "face_indexing", label: "Face indexing only" },
];

const FEATURE_LABELS: Record<string, string> = {
  thumbnail: "Thumbnail generation",
  exif: "EXIF extraction",
  phash: "pHash",
  blur: "Blur detection",
  faces: "Face detection",
  scene_clip: "Scene / search embedding",
  sscd_dedup: "SSCD dedup embedding",
};

const PRESET_NAMES: Record<string, string> = {
  ...Object.fromEntries(PRESET_ORDER.map((p) => [p.key, p.label])),
  custom: "Custom",
  fallback_full: "Fallback (config load failed)",
};

const DEFAULT_USER_SETTINGS: UserSettings = {
  autoTune: true,
  advancedControls: false,
  scannedFolders: [],
  exclusions: [],
  batchSize: 16,
  vramWarningPct: 85,
  vramCriticalPct: 95,
  hnswEfSearch: 80,
};

type StoredUserSettings = Partial<UserSettings> & {
  advanced?: unknown;
  vramWarning?: unknown;
  vramCritical?: unknown;
  efSearch?: unknown;
  libraryRoot?: unknown;
};

function boundedNumber(value: unknown, fallback: number, min: number, max: number): number {
  return typeof value === "number" && Number.isFinite(value)
    ? Math.max(min, Math.min(max, value))
    : fallback;
}

function parseStoredSettings(raw: string | null): StoredUserSettings {
  if (!raw) return {};
  const parsed: unknown = JSON.parse(raw);
  return parsed !== null && typeof parsed === "object"
    ? parsed as StoredUserSettings
    : {};
}

export default function SettingsView() {
  const queryClient = useQueryClient();
  const [advanced, setAdvanced] = useState(DEFAULT_USER_SETTINGS.advancedControls);
  const [autoTune, setAutoTune] = useState(DEFAULT_USER_SETTINGS.autoTune);
  const [scannedFolders, setScannedFolders] = useState<string[]>(DEFAULT_USER_SETTINGS.scannedFolders);
  const [exclusions, setExclusions] = useState<string[]>(DEFAULT_USER_SETTINGS.exclusions);
  const [exclusionDraft, setExclusionDraft] = useState("");
  const [batchSize, setBatchSize] = useState(DEFAULT_USER_SETTINGS.batchSize);
  const [vramWarning, setVramWarning] = useState(DEFAULT_USER_SETTINGS.vramWarningPct);
  const [vramCritical, setVramCritical] = useState(DEFAULT_USER_SETTINGS.vramCriticalPct);
  const [efSearch, setEfSearch] = useState(DEFAULT_USER_SETTINGS.hnswEfSearch);
  const [selectedProfile, setSelectedProfile] = useState("");
  const [preferencesLoaded, setPreferencesLoaded] = useState(false);

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(LOCAL_STORAGE_KEYS.MEMORIA_USER_SETTINGS);
      const saved = parseStoredSettings(raw);
      const advancedValue = window.localStorage.getItem(
        LOCAL_STORAGE_KEYS.MEMORIA_SETTINGS_ADVANCED_MODE,
      );

      if (advancedValue === "true" || advancedValue === "false") {
        setAdvanced(advancedValue === "true");
      } else if (typeof saved.advancedControls === "boolean") {
        setAdvanced(saved.advancedControls);
      } else if (typeof saved.advanced === "boolean") {
        setAdvanced(saved.advanced);
      }
      if (typeof saved.autoTune === "boolean") setAutoTune(saved.autoTune);
      if (Array.isArray(saved.scannedFolders)) {
        setScannedFolders(saved.scannedFolders.filter((v): v is string => typeof v === "string"));
      }
      if (Array.isArray(saved.exclusions)) setExclusions(saved.exclusions.filter((v): v is string => typeof v === "string"));
      setBatchSize(boundedNumber(saved.batchSize, DEFAULT_USER_SETTINGS.batchSize, 1, 50));
      const warning = boundedNumber(saved.vramWarningPct ?? saved.vramWarning, DEFAULT_USER_SETTINGS.vramWarningPct, 50, 94);
      const critical = boundedNumber(saved.vramCriticalPct ?? saved.vramCritical, DEFAULT_USER_SETTINGS.vramCriticalPct, 60, 99);
      setVramWarning(warning);
      setVramCritical(Math.max(critical, warning + 1));
      setEfSearch(boundedNumber(saved.hnswEfSearch ?? saved.efSearch, DEFAULT_USER_SETTINGS.hnswEfSearch, 20, 400));
    } catch {
      // Preferences are best-effort and never block the settings screen.
    } finally {
      setPreferencesLoaded(true);
    }
  }, []);

  useEffect(() => {
    if (!preferencesLoaded) return;
    const settings: UserSettings = {
      autoTune,
      advancedControls: advanced,
      scannedFolders,
      exclusions,
      batchSize,
      vramWarningPct: vramWarning,
      vramCriticalPct: vramCritical,
      hnswEfSearch: efSearch,
    };
    try {
      window.localStorage.setItem(
        LOCAL_STORAGE_KEYS.MEMORIA_USER_SETTINGS,
        JSON.stringify(settings),
      );
      window.localStorage.setItem(
        LOCAL_STORAGE_KEYS.MEMORIA_SETTINGS_ADVANCED_MODE,
        String(advanced),
      );
    } catch {
      // Storage can be disabled or full; runtime settings remain usable.
    }
  }, [advanced, autoTune, batchSize, efSearch, exclusions, preferencesLoaded, scannedFolders, vramCritical, vramWarning]);

  const { data: config, isLoading } = useQuery<ProcessingConfig>({
    queryKey: ["processing-config"],
    queryFn: async () => {
      const res = await fetch(apiUrl(ENDPOINTS.processingConfig));
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json() as Promise<ProcessingConfig>;
    },
    staleTime: 30_000,
  });

  const mutation = useMutation<
    { preset: string; features?: ProcessingConfig["features"] },
    Error,
    { preset: string }
  >({
    mutationFn: async (body) => {
      const res = await fetch(apiUrl(ENDPOINTS.processingConfig), {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json() as Promise<{ preset: string; features?: ProcessingConfig["features"] }>;
    },
    onSuccess: (data) => {
      queryClient.setQueryData<ProcessingConfig>(["processing-config"], (prev) =>
        prev
          ? {
              ...prev,
              preset: data.preset,
              features: data.features ?? prev.presets[data.preset] ?? prev.features,
            }
          : prev,
      );
    },
  });

  const { data: benchmarks } = useQuery<{ benchmarks: ModelBenchmark[] }>({
    queryKey: ["processing-benchmarks", "scene_reindex"],
    queryFn: async () => {
      const res = await fetch(apiUrl(ENDPOINTS.processingBenchmarks(12, "scene_reindex")));
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json() as Promise<{ benchmarks: ModelBenchmark[] }>;
    },
    staleTime: 10_000,
  });

  const reindexMutation = useMutation<
    { job: ReindexJob; created: boolean },
    Error,
    string | undefined
  >({
    mutationFn: async (profile?: string) => {
      const res = await fetch(apiUrl(ENDPOINTS.processingReindexScene(profile)), {
        method: "POST",
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json() as Promise<{ job: ReindexJob; created: boolean }>;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["processing-config"] });
      queryClient.invalidateQueries({ queryKey: ["stats"] });
      queryClient.invalidateQueries({ queryKey: ["processing-benchmarks"] });
    },
  });

  const cancelReindexMutation = useMutation<{ job: ReindexJob }, Error, string>({
    mutationFn: async (jobId: string) => {
      const res = await fetch(apiUrl(ENDPOINTS.processingReindexCancel(jobId)), {
        method: "POST",
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json() as Promise<{ job: ReindexJob }>;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["processing-config"] });
      queryClient.invalidateQueries({ queryKey: ["stats"] });
    },
  });

  const resetMetricsMutation = useMutation<void, Error>({
    mutationFn: async () => {
      const res = await fetch(apiUrl(ENDPOINTS.metricsReset), { method: "POST" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["diagnostics"] });
    },
  });

  const { data: runs } = useQuery<ProcessingRunsResponse>({
    queryKey: ["processing-runs"],
    queryFn: async () => {
      const res = await fetch(apiUrl(ENDPOINTS.processingRuns(20)));
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json() as Promise<ProcessingRunsResponse>;
    },
    staleTime: 5_000,
  });
  // Refresh the latest-run card while the tab is visible.
  useVisibilityInterval(
    () => {
      queryClient.invalidateQueries({ queryKey: ["processing-runs"] });
      queryClient.invalidateQueries({ queryKey: ["processing-config"] });
      queryClient.invalidateQueries({ queryKey: ["processing-benchmarks"] });
    },
    10_000,
  );

  useEffect(() => {
    const targetKey = config?.models?.scene?.desired?.model_key
      ?? config?.models?.scene?.configured?.model_key;
    if (!selectedProfile && targetKey) setSelectedProfile(targetKey);
  }, [
    selectedProfile,
    config?.models?.scene?.desired?.model_key,
    config?.models?.scene?.configured?.model_key,
  ]);

  if (isLoading || !config) {
    return (
      <div className="flex-1 flex items-center justify-center text-neutral-500">
        <Loader2 className="animate-spin mr-2" size={18} /> Loading settings…
      </div>
    );
  }

  const saving = mutation.isPending;
  const latest = runs?.latest ?? null;
  const sceneModel = config.models?.scene;
  const sceneTarget = sceneModel?.desired ?? sceneModel?.configured;
  const faceModel = config.models?.face;
  const sceneProfiles = sceneModel?.profiles ?? [];
  const reindex = sceneModel?.reindex_job;
  const reindexDone = reindex
    ? reindex.processed_items + (reindex.skipped_items ?? 0)
    : 0;
  const modelActionError = reindexMutation.error ?? cancelReindexMutation.error;

  const applyPreset = (preset: string) => mutation.mutate({ preset });

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Processing Settings</h1>
        <p className="text-sm text-neutral-400 mt-1">
          Choose how Memoria indexes photos and which folders to scan.
        </p>
      </div>

      {/* Memoria mode */}
      <section className="space-y-2">
        <div>
          <h2 className="text-sm font-medium text-neutral-300">Memoria mode</h2>
          <p className="mt-1 text-xs text-neutral-500">
            Each mode applies a fixed pipeline policy. Individual stage toggles are not exposed here.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {PRESET_ORDER.map((p) => (
            <button
              key={p.key}
              type="button"
              disabled={saving}
              onClick={() => applyPreset(p.key)}
              className={`px-3 py-1.5 rounded-md text-sm border transition-colors disabled:opacity-50 ${
                config.preset === p.key
                  ? "bg-emerald-600 border-emerald-500 text-white"
                  : "bg-neutral-900 border-neutral-700 text-neutral-300 hover:bg-neutral-800"
              }`}
            >
              {p.label}
            </button>
          ))}
        </div>
        <p className="text-xs text-neutral-500">
          Active mode: <span className="text-neutral-300">{PRESET_NAMES[config.preset] ?? config.preset}</span>
        </p>
      </section>

      {/* Dedup-first warning */}
      {config.preset === "dedup_first" && (
        <div className="flex items-start gap-2 rounded-md border border-yellow-700/60 bg-yellow-900/20 p-3 text-sm text-yellow-200">
          <AlertTriangle size={16} className="mt-0.5 shrink-0" />
          <span>
            Dedup-first mode skips semantic search, face indexing, blur scoring,
            pHash, and EXIF enrichment. Run Full Enrichment later for those features.
          </span>
        </div>
      )}

      <section className="max-w-2xl space-y-3 rounded-lg border border-neutral-800 bg-neutral-900/40 p-4">
        <div className="flex items-center justify-between gap-3">
          <div>
            <h2 className="text-sm font-medium text-neutral-300">Library folders</h2>
            <p className="mt-1 text-xs text-neutral-500">Pick a folder and queue a server-side scan for it.</p>
          </div>
          <label className="flex items-center gap-2 text-xs text-neutral-300">
            <input type="checkbox" className="accent-emerald-500" checked={autoTune} onChange={(e) => setAutoTune(e.target.checked)} />
            Auto hardware tuning
          </label>
        </div>
        <LibraryFolderPicker
          scannedFolders={scannedFolders}
          onScannedFoldersChange={setScannedFolders}
        />
        <div className="space-y-2 border-t border-neutral-800 pt-3">
          <div className="text-xs text-neutral-500">Excluded subfolders</div>
          <div className="flex flex-wrap gap-2">
            {exclusions.map((value) => (
              <button key={value} type="button" onClick={() => setExclusions((current) => current.filter((item) => item !== value))} className="rounded-full border border-neutral-700 bg-neutral-950 px-2.5 py-1 text-xs text-neutral-300 hover:border-red-500/60">
                {value} ×
              </button>
            ))}
            <form
              className="flex min-w-[220px] flex-1 gap-2"
              onSubmit={(event) => {
                event.preventDefault();
                const value = exclusionDraft.trim();
                if (value && !exclusions.includes(value)) setExclusions((current) => [...current, value]);
                setExclusionDraft("");
              }}
            >
              <input value={exclusionDraft} onChange={(e) => setExclusionDraft(e.target.value)} placeholder="Add excluded folder name" className="min-w-0 flex-1 rounded-md border border-neutral-800 bg-neutral-950 px-2.5 py-1 text-xs text-neutral-300" />
              <button type="submit" className="rounded-md border border-neutral-700 px-2.5 py-1 text-xs text-neutral-300 hover:bg-neutral-800">Add</button>
            </form>
          </div>
        </div>
      </section>

      <section className="max-w-2xl rounded-lg border border-neutral-800 bg-neutral-900/40">
        <button type="button" aria-expanded={advanced} aria-controls="advanced-engineering-controls" onClick={() => setAdvanced((value) => !value)} className="flex w-full items-center justify-between px-4 py-3 text-left">
          <span className="flex items-center gap-2 text-sm font-medium text-neutral-300"><SlidersHorizontal size={15} className="text-indigo-400" /> Show Advanced Engineering Controls</span>
          {advanced ? <ChevronUp size={16} className="text-neutral-500" /> : <ChevronDown size={16} className="text-neutral-500" />}
        </button>
        {advanced && (
          <div id="advanced-engineering-controls" className="space-y-4 border-t border-neutral-800 px-4 py-4">
            <RangeControl label="Manual batch size (MEMORIA_BATCH_SIZE)" value={batchSize} min={1} max={50} onChange={setBatchSize} suffix=" items" />
            <RangeControl label="VRAM warning threshold" value={vramWarning} min={50} max={94} onChange={(value) => setVramWarning(Math.min(value, vramCritical - 1))} suffix="%" />
            <RangeControl label="VRAM critical threshold" value={vramCritical} min={60} max={99} onChange={(value) => setVramCritical(Math.max(value, vramWarning + 1))} suffix="%" />
            <RangeControl label="HNSW recall (ef_search)" value={efSearch} min={20} max={400} step={10} onChange={setEfSearch} suffix="" />
            <div className="flex flex-wrap gap-2 pt-1">
              <button type="button" disabled={resetMetricsMutation.isPending} onClick={() => resetMetricsMutation.mutate()} className="inline-flex items-center gap-2 rounded-md border border-neutral-700 px-3 py-1.5 text-xs text-neutral-300 hover:bg-neutral-800 disabled:opacity-50"><Trash2 size={13} /> Clear telemetry</button>
              <button type="button" onClick={() => BlobPool.clearUnused()} className="rounded-md border border-neutral-700 px-3 py-1.5 text-xs text-neutral-300 hover:bg-neutral-800">Sweep image cache</button>
              <button type="button" onClick={() => setExclusions([])} className="rounded-md border border-neutral-700 px-3 py-1.5 text-xs text-neutral-300 hover:bg-neutral-800">Reset exclusions</button>
            </div>
            {resetMetricsMutation.error && <p className="text-xs text-red-400">{resetMetricsMutation.error.message}</p>}
            <p className="text-[11px] leading-5 text-neutral-500">These engineering preferences are retained in this browser. Runtime environment variables remain authoritative for deployed workers.</p>
          </div>
        )}
      </section>

      {sceneModel && (
        <section className="space-y-2">
          <h2 className="text-sm font-medium text-neutral-300">Scene / search model</h2>
          <div className="rounded-lg border border-neutral-800 bg-neutral-900/60 p-4 max-w-2xl space-y-3">
            {sceneProfiles.length > 0 && (
              <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                <label className="text-xs text-neutral-400" htmlFor="scene-profile">Target model profile</label>
                <select
                  id="scene-profile"
                  value={selectedProfile}
                  onChange={(event) => setSelectedProfile(event.target.value)}
                  className="rounded-md border border-neutral-700 bg-neutral-950 px-2.5 py-1.5 text-xs text-neutral-200"
                >
                  {selectedProfile && !sceneProfiles.some((profile) => profile.model_key === selectedProfile) && (
                    <option value={selectedProfile}>
                      {sceneTarget?.model_name ?? selectedProfile}
                    </option>
                  )}
                  {sceneProfiles.map((profile) => (
                    <option key={profile.model_key} value={profile.model_key}>{profile.model_name ?? profile.model_key}</option>
                  ))}
                </select>
              </div>
            )}
            {sceneTarget ? (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
                <Stat label="Driver" value={sceneTarget.driver} />
                <Stat label="Model" value={sceneTarget.model_name} />
                <Stat
                  label="Vector"
                  value={sceneTarget.embedding_dim == null
                    ? "No embedding"
                    : `${sceneTarget.embedding_dim} dimensions`}
                />
                <Stat
                  label="Runtime"
                  value={[
                    sceneTarget.actual_provider,
                    sceneTarget.actual_precision ?? sceneTarget.precision,
                  ].filter(Boolean).join(" / ") || "resolved on load"}
                />
              </div>
            ) : (
              <div className="text-xs text-red-400">
                The configured scene model could not be resolved. An existing active model remains available for search.
              </div>
            )}
            {sceneTarget && sceneModel.active?.model_key !== sceneTarget.model_key && (
              <div className="text-xs text-amber-300">
                Searches remain on {sceneModel.active?.model_name ?? "the previous model"} until the target model is fully indexed.
              </div>
            )}
            {(sceneModel.last_error || reindex?.error) && (
              <div className="text-xs text-red-400">
                {sceneModel.last_error ?? reindex?.error}
              </div>
            )}
            {modelActionError && (
              <div className="text-xs text-red-400">
                {modelActionError instanceof Error ? modelActionError.message : "Model action failed"}
              </div>
            )}
            {reindex && ["queued", "running", "indexing", "cancel_requested"].includes(reindex.status) ? (
              <div className="space-y-1">
                <div className="flex justify-between text-xs text-neutral-400">
                  <span>Background re-index: {reindex.status}</span>
                  <span>{reindexDone}/{reindex.total_items} ({reindex.failed_items} failed)</span>
                </div>
                <div className="h-1.5 rounded bg-neutral-800 overflow-hidden">
                  <div
                    className="h-full bg-emerald-500 transition-all"
                    style={{ width: `${reindex.total_items ? Math.min(100, reindexDone / reindex.total_items * 100) : 0}%` }}
                  />
                </div>
                <button
                  type="button"
                  disabled={reindex.status === "cancel_requested" || cancelReindexMutation.isPending}
                  onClick={() => cancelReindexMutation.mutate(reindex.id)}
                  className="text-xs text-neutral-400 hover:text-neutral-200 disabled:opacity-50"
                >
                  {reindex.status === "cancel_requested" ? "Cancellation requested" : "Cancel re-index"}
                </button>
              </div>
            ) : (
              <button
                type="button"
                disabled={(!selectedProfile && !sceneTarget) || reindexMutation.isPending}
                onClick={() => reindexMutation.mutate(selectedProfile || undefined)}
                className="inline-flex items-center gap-2 px-3 py-1.5 rounded-md text-sm border border-neutral-700 bg-neutral-900 text-neutral-300 hover:bg-neutral-800 disabled:opacity-50"
              >
                <RefreshCw size={14} className={reindexMutation.isPending ? "animate-spin" : ""} />
                Re-index scene vectors
              </button>
            )}
            {selectedProfile && selectedProfile !== sceneTarget?.model_key && (
              <p className="text-xs text-amber-300">The selected profile will be requested for the next background re-index. Search remains on the active namespace until completion.</p>
            )}
          </div>
        </section>
      )}

      <section className="space-y-2">
        <h2 className="text-sm font-medium text-neutral-300">Model management</h2>
        <div className="grid max-w-2xl grid-cols-1 gap-3 sm:grid-cols-2">
          <div className="rounded-lg border border-neutral-800 bg-neutral-900/60 p-4">
            <div className="text-xs uppercase tracking-wider text-neutral-500">Active scene model</div>
            <div className="mt-2 text-sm text-neutral-100">{sceneModel?.active?.model_name ?? "Not indexed yet"}</div>
            <div className="mt-1 text-xs text-neutral-500">{sceneModel?.active?.model_key ?? "—"}</div>
          </div>
          <div className="rounded-lg border border-neutral-800 bg-neutral-900/60 p-4">
            <div className="text-xs uppercase tracking-wider text-neutral-500">Face model profile</div>
            <div className="mt-2 text-sm text-neutral-100">{faceModel?.active?.model_name ?? faceModel?.configured?.model_name ?? "Not configured"}</div>
            <div className="mt-1 text-xs text-neutral-500">{faceModel?.status ?? "—"}</div>
          </div>
        </div>
        <div className="max-w-2xl rounded-lg border border-neutral-800 bg-neutral-900/60 p-4">
          <div className="mb-3 flex items-center justify-between">
            <div className="text-sm font-medium text-neutral-300">Re-index benchmarks</div>
            <span className="text-xs text-neutral-500">images/sec · duration</span>
          </div>
          {benchmarks?.benchmarks?.length ? (
            <div className="space-y-2">
              {benchmarks.benchmarks.map((benchmark, index) => (
                <div key={`${benchmark.model_key}-${benchmark.created_at ?? index}`} className="flex items-center justify-between rounded-md bg-neutral-950/70 px-3 py-2 text-xs">
                  <div className="min-w-0"><div className="truncate text-neutral-300">{benchmark.model_key}</div><div className="text-neutral-600">{benchmark.sample_count ?? 0} images</div></div>
                  <div className="ml-3 shrink-0 text-right text-neutral-400"><div>{(benchmark.items_per_second ?? 0).toFixed(2)} img/s</div><div>{((benchmark.duration_ms ?? 0) / 1000).toFixed(1)}s</div></div>
                </div>
              ))}
            </div>
          ) : <p className="text-xs text-neutral-500">No model re-index benchmarks recorded yet.</p>}
        </div>
      </section>

      {/* Latest run readout */}
      <section className="space-y-2">
        <h2 className="text-sm font-medium text-neutral-300">Latest run</h2>
        {latest ? (
          <div className="rounded-lg border border-neutral-800 bg-neutral-900/60 p-4 max-w-2xl space-y-3">
            {latest.config_load_error && (
              <div className="text-xs text-red-400">
                Worker could not load saved config — ran with all features on (fallback).
              </div>
            )}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
              <Stat label="Mode" value={PRESET_NAMES[latest.mode] ?? latest.mode} />
              <Stat label="Files/hour" value={fmt(latest.files_per_hour)} />
              <Stat label="GB/hour" value={fmt(latest.gb_per_hour)} />
              <Stat label="Last duration" value={`${fmt(latest.total_sec)}s`} />
              <Stat label="Files" value={String(latest.files)} />
              <Stat label="GB" value={fmt(latest.gb)} />
              <Stat
                label="When"
                value={new Date(latest.timestamp_utc).toLocaleTimeString()}
              />
            </div>
            <div className="text-xs text-neutral-500">
              Ran with:{" "}
              {Object.entries(latest.enabled_features)
                .filter(([, v]) => v)
                .map(([k]) => FEATURE_LABELS[k] ?? k)
                .join(", ") || "none"}
            </div>
          </div>
        ) : (
          <p className="text-sm text-neutral-500">
            No runs logged yet. Process some photos to populate this.
          </p>
        )}
      </section>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs text-neutral-500">{label}</div>
      <div className="text-neutral-100">{value}</div>
    </div>
  );
}

function RangeControl({
  label,
  value,
  min,
  max,
  step = 1,
  suffix,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  suffix: string;
  onChange: (value: number) => void;
}) {
  return (
    <label className="block">
      <div className="mb-1 flex justify-between text-xs text-neutral-400"><span>{label}</span><span className="text-neutral-200">{value}{suffix}</span></div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} className="w-full accent-indigo-500" />
    </label>
  );
}

function fmt(n: number): string {
  if (!Number.isFinite(n)) return "0";
  return n >= 100 ? Math.round(n).toString() : n.toFixed(2);
}
