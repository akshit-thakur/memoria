// ─── types ──────────────────────────────────────────────────────────────────
// Every shape the frontend shares across components lives here. Keep this
// file aligned with backend response shapes — TypeScript can't catch a drift
// between the two on its own.
//
/**
 * Memoria — Shared TypeScript interfaces
 *
 * Single source of truth for cross-component shapes. MediaItem lives here;
 * everything else imports it rather than re-declaring its own variant.
 *
 * file_size / created_at / source_device are optional because legacy rows
 * (indexed before stat_file() / EXIF pull landed) may have NULL values for
 * those columns.
 */

/** One photo as seen by the UI.
 *
 * phash and exif are absent from list-view responses (keyset pagination) to
 * keep the payload small at 300k-item scale. They are present on the detail
 * endpoint (/api/v1/photos/:id) which InfoPanel fetches on open.
 */
export interface MediaItem {
  id: number;
  filename: string;
  folder: string;
  thumbnailUrl: string;
  previewUrl: string;
  phash?: string;
  blur_score: number;
  exif?: Record<string, string>;
  indexed_at: string | null;
  file_size?: number;
  created_at?: string;
  source_device?: string;
  location?: { lat: number; lng: number; label?: string };
  /** Cosine distance from the group's original (0 = identical, higher = less similar). Only present on duplicate group members. */
  cos_distance?: number | null;
  /** Distance from the active scene/search model. Only present on semantic-search results. */
  distance?: number;
}

export type IntentPreset = "quiet" | "turbo";

/** One immutable scene or face model identity returned by the processing API. */
export interface ModelProfile {
  model_key: string;
  kind?: string;
  driver: string;
  model_name: string;
  model_id?: string;
  weights?: string | null;
  revision?: string | null;
  embedding_dim: number | null;
  input_width?: number;
  input_height?: number;
  input_resolution?: [number, number] | number[];
  actual_provider?: string | null;
  actual_precision?: string | null;
  precision?: string | null;
  default_precision?: string;
  preferred_providers?: string[];
  provider_preferences?: string[];
  distance_metric?: string;
  normalized?: boolean;
  index_kind?: string;
  fingerprint?: string;
  metadata?: Record<string, unknown>;
  benchmark_history?: ModelBenchmark[];
}

/** Backward-compatible name retained for components written during Phase 2. */
export type ModelDescriptor = ModelProfile;

export type ReindexJobStatus =
  | "queued"
  | "running"
  | "indexing"
  | "cancel_requested"
  | "cancelled"
  | "completed"
  | "failed";

/** Persistent progress for a model-scoped background re-index. */
export interface ReindexJob {
  id: string;
  kind?: string;
  model_key: string;
  status: ReindexJobStatus;
  total_items: number;
  processed_items: number;
  failed_items: number;
  skipped_items?: number;
  batch_size?: number | null;
  force_reindex?: boolean;
  cursor_media_id?: number;
  progress_percentage?: number;
  metrics?: Record<string, unknown>;
  error?: string | null;
  created_at?: string;
  requested_at?: string;
  started_at?: string | null;
  finished_at?: string | null;
  updated_at?: string;
}

/** Compact re-index shape embedded in the high-frequency stats response. */
export interface ReindexJobSummary {
  id: string;
  status: ReindexJobStatus;
  total_items: number;
  processed_items: number;
  failed_items: number;
  skipped_items?: number;
  target_model_key: string;
}

export interface ModelBenchmark {
  id?: string;
  reindex_job_id?: string | null;
  model_key: string;
  benchmark_type?: string;
  operation?: string;
  sample_count?: number;
  items?: number;
  duration_ms?: number;
  items_per_second?: number;
  query_latency_ms?: Record<string, number> | null;
  peak_vram_mb?: number | null;
  runtime_provider?: string | null;
  runtime_precision?: string | null;
  provider?: string | null;
  precision?: string | null;
  metrics?: Record<string, unknown>;
  created_at?: string;
}

/** Configured/desired/active scene-model state returned with processing config. */
export interface SceneModelStatus {
  configured: ModelProfile | null;
  active?: ModelProfile | null;
  desired?: ModelProfile | null;
  status: string;
  generation?: number;
  last_error?: string | null;
  reindex_job?: ReindexJob | null;
  profiles?: ModelProfile[];
}

export interface FaceModelStatus {
  configured?: ModelProfile | null;
  configured_embedding?: ModelProfile | null;
  active?: ModelProfile | null;
  desired?: ModelProfile | null;
  status?: string;
  last_error?: string | null;
  profiles?: ModelProfile[];
}

export type ProcessingFeatures = Record<string, boolean>;

export interface ProcessingConfig {
  preset: string;
  features: ProcessingFeatures;
  feature_keys: string[];
  presets: Record<string, ProcessingFeatures>;
  models?: {
    scene?: SceneModelStatus;
    face?: FaceModelStatus;
  };
}

export interface ProcessingRun {
  timestamp_utc: string;
  mode: string;
  config_load_error?: boolean;
  files: number;
  gb: number;
  total_sec: number;
  files_per_hour: number;
  gb_per_hour: number;
  enabled_features: ProcessingFeatures;
  stage_sec: Record<string, number>;
}

export interface ProcessingRunsResponse {
  latest: ProcessingRun | null;
  runs: ProcessingRun[];
}

export interface VramUsage {
  backend: "cuda" | "mps";
  allocated_mb: number;
  driver_mb: number;
}

export interface RamUsage {
  used_mb: number;
  total_mb: number;
  percent: number;
}

export interface IndexingStats {
  total_indexed: number;
  total_bytes_indexed?: number;
  queue_size: number;
  media_read_only?: boolean;
  current_file?: string;
  vram_usage?: VramUsage | null;
  ram_usage?: RamUsage | null;
  faces_indexed?: number;
  faces_total?: number;
  face_clusters?: number;
  face_scan_active?: boolean;
  blur_scored?: number;
  blur_flagged?: number;
  blur_scan_active?: boolean;
  scene_indexed?: number;
  scene_model_key?: string | null;
  scene_reindex?: ReindexJobSummary | null;
  scene_scan_active?: boolean;
}

export interface QueueStats {
  pending: number;
  scheduled?: number;
  running?: number;
  failed: number;
}

export interface MetricSeries {
  n?: number;
  value?: number;
  mean?: number;
  p50?: number;
  p95?: number;
  p99?: number;
  min?: number;
  max?: number;
  last_ts?: number;
  sources?: string[];
}

export interface MetricsSnapshot {
  processes?: Record<string, {
    counters?: Record<string, MetricSeries>;
    histograms?: Record<string, MetricSeries>;
  }>;
  combined?: {
    counters?: Record<string, MetricSeries>;
    histograms?: Record<string, MetricSeries>;
  };
}

export type DiagnosticStageKey = "thumbnail" | "exif" | "phash" | "faces" | "scene";

export interface DiagnosticStageMetrics {
  samples: number;
  mean_ms: number;
  p95_ms: number;
}

export interface ResourceAllocation {
  used_mb: number;
  total_mb: number | null;
  percent: number;
  backend?: string;
}

/** Normalized view model consumed by the diagnostics drawer. */
export interface DiagnosticsMetrics {
  throughput: {
    photos_per_second: number;
    processed_photos: number;
  };
  queue: {
    total: number;
    pending: number;
    running: number;
    failed: number;
  };
  stages: Record<DiagnosticStageKey, DiagnosticStageMetrics>;
  system: {
    vram: ResourceAllocation | null;
    ram: ResourceAllocation | null;
  };
}

/** Browser-retained UI preferences; deployed worker environment remains authoritative. */
export interface UserSettings {
  autoTune: boolean;
  advancedControls: boolean;
  scannedFolders: string[];
  exclusions: string[];
  batchSize: number;
  vramWarningPct: number;
  vramCriticalPct: number;
  hnswEfSearch: number;
}

/** One face cluster ≈ one identified person. */
export interface FaceCluster {
  id: number;
  label: string | null;
  sample_thumbnail: string | null;
  count: number;
  face_count?: number;
  confidence_score?: number;
  representative_media_id?: number | null;
}

/** One face detection on a single photo. */
export interface FaceInstance {
  cluster_id: number;
  label: string | null;
  bbox: { x: number; y: number; w: number; h: number };
  thumbnail: string;
  face_id?: number;
}

/** A hit from `/api/v1/faces/{id}/similar`. `distance` is cosine
 *  distance within the active 512-d face-model namespace (lower = more similar). */
export interface SimilarFace {
  face_id: number;
  media_id: number;
  bbox: { x: number; y: number; w: number; h: number };
  cluster_id: number | null;
  confidence: number;
  cluster_label: string | null;
  filename: string;
  folder: string;
  distance: number;
}

/** Group of photos sharing the same perceptual hash. */
export interface DuplicateGroup {
  id: number;
  phash: string;
  members: MediaItem[];
}

/** Pin on the map view. */
export interface LocationPoint {
  media_id: number;
  lat: number;
  lng: number;
  thumbnail: string;
  label?: string;
}
