// ─── FaceGrid ───────────────────────────────────────────────────────────────
// In plain English: the "Faces" tab. Backend ML has grouped every detected
// face into clusters (one cluster ≈ one person). This view shows a grid of
// circular face thumbnails (one per cluster). Click one and you see all
// photos that contain that face, with a free-form label field — naming a
// cluster also auto-merges any other clusters that share the same label.
//
// There's a banner at the top with live counts (faces detected, photos
// scanned, clusters formed) plus Scan and Clear buttons.
//
/**
 * Memoria — Face Grid View
 *
 * Responsive grid of face clusters (both labeled and unlabeled).
 * Labeled faces shown first, sorted by photo count descending.
 * Shows face indexing progress when the pipeline is actively scanning.
 * Click a cluster to explore all photos belonging to that face.
 */

"use client";

import { useMemo, useState, useCallback, useEffect, useRef } from "react";
import { useInfiniteQuery, useQuery, useQueryClient } from "@tanstack/react-query";
import type { FaceCluster } from "@/types";
import { Users, ScanSearch, ArrowLeft, Pencil, Check, X, Trash2, Loader2 } from "lucide-react";
import BlobImg from "@/components/BlobImg";
import { apiUrl } from "@/lib/api";
import { ENDPOINTS } from "@/lib/endpoints";
import { useVisibilityInterval } from "@/hooks/useVisibilityInterval";
import { useStats } from "@/hooks/useStats";

const CLUSTER_PAGE_SIZE = 100;
const MEMBERS_INITIAL_CAP = 24;

interface FacesResponse {
  clusters: FaceCluster[];
  total: number;
}

interface ClusterMember {
  face_id: number;
  media_id: number;
  bbox: { x: number; y: number; w: number; h: number };
  confidence: number;
  file_path: string;
}

interface ClusterDetail {
  id: number;
  label: string | null;
  face_count: number;
  confidence_score: number;
  representative_media_id: number | null;
  members: ClusterMember[];
}

const faceCropUrl = ENDPOINTS.mediaFaceCrop;

function FaceIndexingBanner({ onScanClick, isScanning: isScanTriggered, onClear }: {
  onScanClick: () => void;
  isScanning: boolean;
  onClear: () => void;
}) {
  const { data: stats } = useStats();

  if (!stats) return null;

  const isWorkerBusy = stats.queue_size > 0;
  const currentDir = stats.current_file
    ? stats.current_file.split("/").slice(0, -1).join("/")
    : null;
  const hasPhotos = stats.total_indexed > 0;
  const noFaces = stats.faces_total === 0;

  return (
    <div className="mb-4 rounded-lg bg-neutral-900/60 border border-neutral-800 p-3">
      <div className="flex items-center justify-between gap-3 text-sm">
        <div className="flex items-center gap-3 min-w-0">
          <Users size={16} className="text-indigo-400 shrink-0" />
          <div className="min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-neutral-300 font-medium">
                {stats.faces_total} face{stats.faces_total !== 1 ? "s" : ""} detected
              </span>
              <span className="text-neutral-600">·</span>
              <span className="text-neutral-400">
                {stats.faces_indexed} photo{stats.faces_indexed !== 1 ? "s" : ""} scanned
              </span>
              <span className="text-neutral-600">·</span>
              <span className="text-neutral-400">
                {stats.face_clusters} cluster{stats.face_clusters !== 1 ? "s" : ""}
              </span>
            </div>
            {isScanTriggered && currentDir && (
              <p className="text-xs text-indigo-400 mt-1 truncate">
                <span className="inline-block w-2 h-2 rounded-full bg-indigo-400 animate-pulse mr-1.5 align-middle" />
                Scanning {currentDir}
              </p>
            )}
            {isWorkerBusy && (
              <p className="text-xs text-neutral-500 mt-0.5">
                {stats.queue_size} file{stats.queue_size !== 1 ? "s" : ""} remaining in queue
              </p>
            )}
          </div>
        </div>
        {hasPhotos && (
          <div className="flex items-center gap-2 shrink-0">
            {(stats.faces_total ?? 0) > 0 && (
              <button
                onClick={onClear}
                disabled={isScanTriggered}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium
                           bg-neutral-800 hover:bg-red-600/80 text-neutral-400 hover:text-white
                           transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                title="Clear all face data"
              >
                <Trash2 size={14} />
                Clear
              </button>
            )}
            <button
              onClick={onScanClick}
              disabled={isScanTriggered}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium
                         bg-indigo-600 hover:bg-indigo-500 text-white transition-colors
                         disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ScanSearch size={14} className={isScanTriggered ? "animate-spin" : ""} />
              {isScanTriggered ? "Scanning…" : noFaces ? "Scan for Faces" : "Rescan"}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

/* ── Cluster Detail View ─────────────────────────────────────────────── */
function ClusterDetailView({
  clusterId,
  onBack,
  onItemClick,
}: {
  clusterId: number;
  onBack: () => void;
  onItemClick?: (mediaId: number) => void;
}) {
  const [editingLabel, setEditingLabel] = useState(false);
  const [labelInput, setLabelInput] = useState("");
  const [showAllMembers, setShowAllMembers] = useState(false);
  const queryClient = useQueryClient();

  const { data, isLoading } = useQuery<ClusterDetail>({
    queryKey: ["face-cluster", clusterId],
    queryFn: async () => {
      const res = await fetch(apiUrl(ENDPOINTS.faceCluster(clusterId)));
      if (!res.ok) throw new Error("Failed to load cluster");
      return res.json();
    },
    staleTime: 30_000,
  });

  const saveLabel = useCallback(async () => {
    const trimmed = labelInput.trim();
    if (!trimmed) return;
    try {
      const res = await fetch(apiUrl(ENDPOINTS.faceClusterLabel(clusterId)), {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label: trimmed }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      queryClient.invalidateQueries({ queryKey: ["face-cluster", clusterId] });
      queryClient.invalidateQueries({ queryKey: ["face-clusters"] });
      setEditingLabel(false);
    } catch { /* ignore */ }
  }, [clusterId, labelInput, queryClient]);

  const deleteCluster = useCallback(async () => {
    if (!window.confirm("Delete this cluster? This cannot be undone. Future photos of this person will also be hidden.")) return;
    try {
      const res = await fetch(apiUrl(ENDPOINTS.faceClusterDelete(clusterId)), { method: "DELETE" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      queryClient.invalidateQueries({ queryKey: ["face-clusters"] });
      onBack();
    } catch { /* ignore */ }
  }, [clusterId, queryClient, onBack]);

  if (isLoading || !data) {
    return (
      <div className="flex-1 flex items-center justify-center text-neutral-500">
        Loading face details…
      </div>
    );
  }

  const displayName = data.label ?? `Unknown #${data.id}`;

  return (
    <div className="flex-1 overflow-y-auto bg-black p-4">
      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <button
          onClick={onBack}
          className="p-2 rounded-lg hover:bg-neutral-800 transition-colors text-neutral-400 hover:text-white"
        >
          <ArrowLeft size={20} />
        </button>

        {/* Representative face crop */}
        {data.representative_media_id && data.members.length > 0 && (
          <BlobImg
            src={faceCropUrl(
              data.members[0].media_id,
              data.members[0].bbox
            )}
            alt={displayName}
            fallback={
              <div className="w-14 h-14 rounded-full bg-neutral-800" />
            }
            className="w-14 h-14 rounded-full object-cover bg-neutral-800"
          />
        )}

        <div className="min-w-0 flex-1">
          {editingLabel ? (
            <div className="flex items-center gap-2">
              <input
                autoFocus
                value={labelInput}
                onChange={(e) => setLabelInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && saveLabel()}
                placeholder="Enter name…"
                className="bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-1.5
                           text-sm text-white placeholder:text-neutral-500 outline-none
                           focus:border-indigo-500 w-48"
              />
              <button onClick={saveLabel} className="p-1 text-green-400 hover:text-green-300">
                <Check size={16} />
              </button>
              <button onClick={() => setEditingLabel(false)} className="p-1 text-neutral-500 hover:text-neutral-300">
                <X size={16} />
              </button>
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <h2 className="text-lg font-semibold text-white truncate">{displayName}</h2>
              <button
                onClick={() => {
                  setLabelInput(data.label ?? "");
                  setEditingLabel(true);
                }}
                className="p-1 text-neutral-500 hover:text-indigo-400 transition-colors"
                title="Rename this person"
              >
                <Pencil size={14} />
              </button>
              <button
                onClick={deleteCluster}
                className="p-1 text-neutral-500 hover:text-red-400 transition-colors"
                title="Delete this cluster"
              >
                <Trash2 size={14} />
              </button>
            </div>
          )}
          <p className="text-sm text-neutral-400">
            {data.face_count} photo{data.face_count !== 1 ? "s" : ""}
            {data.confidence_score != null && (
              <span className="text-neutral-600 ml-2">
                · {Math.round(data.confidence_score * 100)}% confidence
              </span>
            )}
          </p>
        </div>
      </div>

      {/* Photo grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3">
        {(showAllMembers ? data.members : data.members.slice(0, MEMBERS_INITIAL_CAP)).map((member) => {
          const filename = member.file_path.split("/").pop() ?? "";
          return (
            <div
              key={member.face_id}
              onClick={() => onItemClick?.(member.media_id)}
              className="group relative rounded-xl overflow-hidden bg-neutral-900
                         hover:ring-2 hover:ring-indigo-500/50 transition-all cursor-pointer"
            >
              {/* Full photo thumbnail */}
              <BlobImg
                src={ENDPOINTS.mediaThumbnail(member.media_id)}
                alt={filename}
                loading="lazy"
                fallback={
                  <div className="w-full aspect-square bg-neutral-900" />
                }
                className="w-full aspect-square object-cover"
              />

              {/* Face crop badge (top-right) */}
              <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                <BlobImg
                  src={faceCropUrl(member.media_id, member.bbox)}
                  alt="face"
                  loading="lazy"
                  fallback={
                    <div className="w-10 h-10 rounded-full bg-neutral-800 border-2 border-white/30" />
                  }
                  className="w-10 h-10 rounded-full object-cover border-2 border-white/30 shadow-lg"
                />
              </div>

              {/* File name overlay */}
              <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 to-transparent
                              p-2 opacity-0 group-hover:opacity-100 transition-opacity">
                <p className="text-xs text-neutral-300 truncate">{filename}</p>
                <p className="text-[10px] text-neutral-500">
                  {Math.round(member.confidence * 100)}% match
                </p>
              </div>
            </div>
          );
        })}
      </div>
      {!showAllMembers && data.members.length > MEMBERS_INITIAL_CAP && (
        <button
          onClick={() => setShowAllMembers(true)}
          className="mt-4 w-full py-2 rounded-lg border border-neutral-800 text-sm text-neutral-400
                     hover:bg-neutral-900 hover:text-neutral-200 transition-colors"
        >
          Show all {data.members.length} photos
        </button>
      )}
    </div>
  );
}

/* ── Main FaceGrid ───────────────────────────────────────────────────── */
export default function FaceGrid({ onItemClick }: { onItemClick?: (mediaId: number) => void }) {
  const queryClient = useQueryClient();
  const [scanning, setScanning] = useState(false);
  const [selectedCluster, setSelectedCluster] = useState<number | null>(null);
  const sentinelRef = useRef<HTMLDivElement>(null);

  const {
    data,
    isLoading,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
  } = useInfiniteQuery<FacesResponse>({
    queryKey: ["face-clusters"],
    queryFn: async ({ pageParam = 0 }) => {
      const url = `${ENDPOINTS.faceClusters}?limit=${CLUSTER_PAGE_SIZE}&offset=${pageParam}`;
      const res = await fetch(apiUrl(url));
      if (!res.ok) throw new Error("Failed to load faces");
      return res.json();
    },
    getNextPageParam: (lastPage, allPages) => {
      const loaded = allPages.reduce((n, p) => n + p.clusters.length, 0);
      return loaded < lastPage.total ? loaded : undefined;
    },
    initialPageParam: 0,
    staleTime: 30_000,
    refetchOnWindowFocus: false,
  });

  useEffect(() => {
    const el = sentinelRef.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting && hasNextPage && !isFetchingNextPage) fetchNextPage(); },
      { rootMargin: "200px" },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [hasNextPage, isFetchingNextPage, fetchNextPage]);

  // Auto-refresh clusters while scanning (and the tab is visible).
  useVisibilityInterval(
    () => queryClient.resetQueries({ queryKey: ["face-clusters"] }),
    5000,
    scanning,
  );

  const handleScan = useCallback(async () => {
    setScanning(true);
    try {
      const res = await fetch(apiUrl(ENDPOINTS.faceIndex), { method: "POST" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
    } catch { /* ignore */ }
    setTimeout(() => setScanning(false), 30000);
  }, []);

  const handleClear = useCallback(async () => {
    if (!confirm("Clear all detected faces? You can re-scan afterwards.")) return;
    try {
      const res = await fetch(apiUrl(ENDPOINTS.faceClear), { method: "DELETE" });
      if (res.ok) {
        queryClient.invalidateQueries({ queryKey: ["face-clusters"] });
        setSelectedCluster(null);
      }
    } catch { /* ignore */ }
  }, [queryClient]);

  const allClusters = useMemo(
    () => data?.pages.flatMap((p) => p.clusters) ?? [],
    [data],
  );
  const totalClusters = data?.pages[0]?.total ?? 0;

  const sorted = useMemo(() => {
    const labeled = allClusters.filter((c) => c.label != null).sort((a, b) => b.count - a.count);
    const unlabeled = allClusters.filter((c) => c.label == null).sort((a, b) => b.count - a.count);
    return [...labeled, ...unlabeled];
  }, [allClusters]);

  // Cluster detail view
  if (selectedCluster !== null) {
    return (
      <ClusterDetailView
        clusterId={selectedCluster}
        onBack={() => setSelectedCluster(null)}
        onItemClick={onItemClick}
      />
    );
  }

  if (isLoading) {
    return (
      <div className="flex-1 flex items-center justify-center text-neutral-500">
        Loading faces…
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto bg-black p-4">
      <FaceIndexingBanner onScanClick={handleScan} isScanning={scanning} onClear={handleClear} />
      <h2 className="text-sm font-semibold text-neutral-400 uppercase tracking-wider mb-4">
        Faces ({totalClusters})
      </h2>
      {sorted.length === 0 ? (
        <div className="flex flex-col items-center justify-center text-neutral-500 py-20">
          <Users size={48} className="mb-4 text-neutral-700" />
          <p>No faces detected yet.</p>
          <p className="text-xs mt-1 text-neutral-600">Faces are detected automatically as photos are indexed.</p>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-6">
            {sorted.map((cluster) => (
              <div key={cluster.id} className="relative group/card">
                <button
                  onClick={() => setSelectedCluster(cluster.id)}
                  className="flex flex-col items-center gap-3 p-4 rounded-xl hover:bg-neutral-900/50 transition-colors w-full"
                >
                  {cluster.sample_thumbnail ? (
                    <BlobImg
                      src={cluster.sample_thumbnail}
                      alt={cluster.label ?? `Unknown #${cluster.id}`}
                      fallback={
                        <div className="w-28 h-28 rounded-full bg-neutral-800" />
                      }
                      className="w-28 h-28 rounded-full object-cover bg-neutral-800"
                    />
                  ) : (
                    <div className="w-28 h-28 rounded-full bg-neutral-800 flex items-center justify-center">
                      <Users size={32} className="text-neutral-600" />
                    </div>
                  )}
                  <span className="text-sm text-neutral-200 truncate max-w-full font-medium">
                    {cluster.label ?? `Unknown #${cluster.id}`}
                  </span>
                  <span className="text-xs text-neutral-500">
                    {cluster.count} photo{cluster.count !== 1 ? "s" : ""}
                  </span>
                </button>
                {/* Delete badge — only visible on hover */}
                <button
                  onClick={async (e) => {
                    e.stopPropagation();
                    if (!window.confirm("Delete this cluster? This cannot be undone. Future photos of this person will also be hidden.")) return;
                    try {
                      const res = await fetch(apiUrl(ENDPOINTS.faceClusterDelete(cluster.id)), { method: "DELETE" });
                      if (!res.ok) throw new Error(`HTTP ${res.status}`);
                      queryClient.resetQueries({ queryKey: ["face-clusters"] });
                    } catch { /* ignore */ }
                  }}
                  className="absolute top-2 right-2 p-1 rounded-full bg-neutral-900/80 text-neutral-500
                             hover:text-red-400 hover:bg-neutral-800 transition-all
                             opacity-0 group-hover/card:opacity-100"
                  title="Delete cluster"
                >
                  <X size={14} />
                </button>
              </div>
            ))}
          </div>
          <div ref={sentinelRef} className="h-1" />
          {isFetchingNextPage && (
            <div className="flex justify-center py-6 text-neutral-500">
              <Loader2 size={18} className="animate-spin" />
            </div>
          )}
        </>
      )}
    </div>
  );
}
