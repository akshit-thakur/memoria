// ─── lib/endpoints ──────────────────────────────────────────────────────────
// In plain English: this is the address book. Every backend URL the
// frontend ever calls lives here as either a constant string or a tiny
// function that builds the URL from path/query parameters. If the backend
// renames a route, this is the only file that has to change.
//
/**
 * Memoria — API endpoint builders
 *
 * Every API URL used by the frontend flows through here. Keeps the backend
 * route list discoverable in one file and makes renames a single-site edit.
 */

// Bounding box used by face-crop and editor endpoints.
// Coordinates are in *original image pixels* (not thumbnail pixels), and the
// origin is the top-left corner.
export interface BBox {
    x: number;
    y: number;
    w: number;
    h: number;
}

export const ENDPOINTS = {
    // Photos
    photosList: "/api/v1/photos",
    photoDetail: (id: number | string) => `/api/v1/photos/${id}`,

    // Media blobs (thumbnail, face-crop)
    mediaThumbnail: (id: number | string) => `/api/v1/media/${id}/thumbnail`,
    mediaFaceCrop: (id: number | string, bbox: BBox) =>
        `/api/v1/media/${id}/face-crop?x=${bbox.x}&y=${bbox.y}&w=${bbox.w}&h=${bbox.h}`,

    // Stats + health
    stats: "/api/v1/stats",
    queueStats: "/api/v1/queue/stats",
    metrics: "/api/v1/metrics",
    metricsReset: "/api/v1/metrics/reset",
    libraryVersion: "/api/v1/library/version",
    progress: "/api/v1/progress",

    // Face clusters
    faceClusters: "/api/v1/faces/clusters",
    faceCluster: (id: number | string) => `/api/v1/faces/clusters/${id}`,
    faceClusterLabel: (id: number | string) =>
        `/api/v1/faces/clusters/${id}/label`,
    faceClusterDelete: (id: number | string) => `/api/v1/faces/clusters/${id}`,
    facesForPhoto: (mediaId: number | string) =>
        `/api/v1/faces/photo/${mediaId}`,
    faceSimilar: (faceId: number | string, limit = 24) =>
        `/api/v1/faces/${faceId}/similar?limit=${limit}`,
    faceIndex: "/api/v1/faces/index",
    faceClear: "/api/v1/faces/clear",

    // Blurry photos
    blurryPhotos: "/api/v1/blurry/photos",
    blurrySharpen: (id: number | string) =>
        `/api/v1/blurry/photos/${id}/sharpen`,
    blurryDelete: (id: number | string) => `/api/v1/blurry/photos/${id}`,
    blurryScan: "/api/v1/blurry/scan",

    // Duplicate groups
    dedupGroups: (page = 1, pageSize = 20) =>
        `/api/v1/dedup/groups?page=${page}&page_size=${pageSize}`,
    dedupDeleteDuplicates: (groupId: number | string) =>
        `/api/v1/dedup/groups/${groupId}/delete-duplicates`,

    // Image editor
    editorCrop: "/api/v1/editor/crop",
    editorFilter: "/api/v1/editor/filter",
    editorRotate: "/api/v1/editor/rotate",

    // Location map
    mapLocations: "/api/v1/map/locations",

    // Semantic / scene search (active server-side model)
    searchText: (q: string, limit = 50) =>
        `/api/v1/search/text?q=${encodeURIComponent(q)}&limit=${limit}`,
    searchSimilar: (id: number | string, limit = 50) =>
        `/api/v1/search/similar/${id}?limit=${limit}`,

    // Processing config (per-stage feature toggles + presets) + throughput
    processingConfig: "/api/v1/processing/config",
    processingModels: "/api/v1/processing/models",
    processingReindexScene: (profile?: string, force = false) => {
        const params = new URLSearchParams();
        if (profile) params.set("profile", profile);
        if (force) params.set("force", "true");
        const query = params.toString();
        return `/api/v1/processing/reindex/scene${query ? `?${query}` : ""}`;
    },
    processingReindexJob: (id: string) => `/api/v1/processing/reindex/${id}`,
    processingReindexCancel: (id: string) =>
        `/api/v1/processing/reindex/${id}/cancel`,
    processingBenchmarks: (limit = 50, benchmarkType?: string) => {
        const params = new URLSearchParams({ limit: String(limit) });
        if (benchmarkType) params.set("benchmark_type", benchmarkType);
        return `/api/v1/processing/benchmarks?${params.toString()}`;
    },
    processingRuns: (limit = 20) => `/api/v1/processing/runs?limit=${limit}`,

    // Library on-demand folder browser + scan
    libraryBrowse: (path?: string) => {
        const params = new URLSearchParams();
        if (path) params.set("path", path);
        const query = params.toString();
        return `/api/v1/library/browse${query ? `?${query}` : ""}`;
    },
    libraryScan: "/api/v1/library/scan",
} as const;
