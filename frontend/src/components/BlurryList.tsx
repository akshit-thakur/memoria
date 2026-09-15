// ─── BlurryList ─────────────────────────────────────────────────────────────
// In plain English: the "Blurry Photos" tab. Lists every photo whose blur
// score is above the configured threshold (default 0.75), sorted with the
// blurriest first. Each row has two buttons:
//   • Sharpen — runs a Pillow UnsharpMask on the original and writes the
//     result to `~/.memoria/edits/{media_id}/`. Original is left alone.
//   • Delete  — moves the photo to `~/.memoria/trash/` and removes the
//     database row.
// There's a Scan button at the top to (re)compute blur scores for the
// whole library.
//
/**
 * Memoria — Blurry Photos List
 *
 * Displays photos flagged as blurry with placeholder Sharpen/Delete actions.
 * Sorted by blur_score descending (highest score = most blurry first).
 */

"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { useInfiniteQuery, useQueryClient, useMutation } from "@tanstack/react-query";
import { Wand2, Trash2, ScanSearch, Loader2 } from "lucide-react";
import BlobImg from "@/components/BlobImg";
import { apiUrl } from "@/lib/api";
import { ENDPOINTS } from "@/lib/endpoints";
import { useVisibilityInterval } from "@/hooks/useVisibilityInterval";
import { useStats } from "@/hooks/useStats";

interface BlurryItem {
  id: number;
  file_path: string;
  filename: string;
  blur_score: number;
  mime_type: string | null;
  file_size: number | null;
}

interface BlurryResponse {
  items: BlurryItem[];
  total: number;
  threshold: number;
}

interface BlurryListProps {
  onItemClick?: (mediaId: number) => void;
}

/** Tailwind class for the inline blur bar; redder = more blurry.
 *  Note: this scale is 0–1 (normalised), unlike InfoPanel.blurColor which
 *  is on a 0–200 raw-score scale. The two views feed off different
 *  endpoints. */
function blurBarColor(score: number): string {
  if (score > 0.85) return "bg-red-500";
  if (score > 0.75) return "bg-yellow-500";
  return "bg-green-500";
}

/** Human label shown next to the blur bar. Same thresholds as the colour. */
function blurLabel(score: number): string {
  if (score > 0.85) return "Very blurry";
  if (score > 0.75) return "Borderline";
  return "Acceptable";
}

const PAGE_SIZE = 50;

export default function BlurryList({ onItemClick }: BlurryListProps) {
  const queryClient = useQueryClient();
  const { data: runtimeStats } = useStats();
  const mediaReadOnly = runtimeStats?.media_read_only !== false;
  const [scanning, setScanning] = useState(false);
  const [pendingId, setPendingId] = useState<{ id: number; op: "sharpen" | "delete" } | null>(null);
  const sentinelRef = useRef<HTMLDivElement>(null);

  const {
    data,
    isLoading,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
  } = useInfiniteQuery<BlurryResponse>({
    queryKey: ["blurry-photos"],
    queryFn: async ({ pageParam = 0 }) => {
      const url = `${ENDPOINTS.blurryPhotos}?limit=${PAGE_SIZE}&offset=${pageParam}`;
      const res = await fetch(apiUrl(url));
      if (!res.ok) throw new Error("Failed to load blurry photos");
      return res.json();
    },
    getNextPageParam: (lastPage, allPages) => {
      const loaded = allPages.reduce((n, p) => n + p.items.length, 0);
      return loaded < lastPage.total ? loaded : undefined;
    },
    initialPageParam: 0,
    staleTime: 30_000,
    refetchOnWindowFocus: false,
  });

  // Intersection observer — load next page when sentinel enters viewport.
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

  const sharpenMutation = useMutation({
    mutationFn: async (id: number) => {
      const res = await fetch(apiUrl(ENDPOINTS.blurrySharpen(id)), { method: "POST" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json() as Promise<{ output_path: string }>;
    },
    onSettled: () => setPendingId(null),
  });

  const deleteMutation = useMutation({
    mutationFn: async (id: number) => {
      const res = await fetch(apiUrl(ENDPOINTS.blurryDelete(id)), { method: "DELETE" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json() as Promise<{ trashed_to: string | null }>;
    },
    onSuccess: () => {
      queryClient.resetQueries({ queryKey: ["blurry-photos"] });
      queryClient.invalidateQueries({ queryKey: ["photos"] });
    },
    onSettled: () => setPendingId(null),
  });

  const handleScan = useCallback(async () => {
    setScanning(true);
    try {
      const res = await fetch(apiUrl(ENDPOINTS.blurryScan), { method: "POST" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
    } catch {
      setScanning(false);
    }
  }, []);

  // Auto-refresh blurry list every 5s (tab-visible only) while a scan is in
  // flight. Stop automatically after 60s as a safety net.
  useVisibilityInterval(
    () => queryClient.resetQueries({ queryKey: ["blurry-photos"] }),
    5000,
    scanning,
  );
  useEffect(() => {
    if (!scanning) return;
    const timeout = setTimeout(() => {
      setScanning(false);
      queryClient.resetQueries({ queryKey: ["blurry-photos"] });
    }, 60_000);
    return () => clearTimeout(timeout);
  }, [scanning, queryClient]);

  if (isLoading) {
    return (
      <div className="flex-1 flex items-center justify-center text-neutral-500">
        Loading blurry photos…
      </div>
    );
  }

  const items = data?.pages.flatMap((p) => p.items) ?? [];
  const total = data?.pages[0]?.total ?? 0;

  return (
    <div className="flex-1 overflow-y-auto bg-black p-4 space-y-3">
      {/* Scan banner */}
      <div className="flex items-center justify-between mb-2">
        <h2 className="text-sm font-semibold text-neutral-400 uppercase tracking-wider">
          Blurry Photos ({total})
        </h2>
        <button
          onClick={handleScan}
          disabled={scanning}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium
                     bg-indigo-600 hover:bg-indigo-500 text-white transition-colors
                     disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <ScanSearch size={14} className={scanning ? "animate-spin" : ""} />
          {scanning ? "Scanning…" : "Scan for Blur"}
        </button>
      </div>

      {runtimeStats?.media_read_only === true && (
        <div className="rounded-md border border-amber-800/40 bg-amber-950/30 px-3 py-2 text-xs text-amber-200">
          Source media is mounted read-only. Sharpened copies can still be
          saved, but deleting originals is disabled in this deployment.
        </div>
      )}
      {(sharpenMutation.error || deleteMutation.error) && (
        <div className="rounded-md border border-red-800/40 bg-red-950/30 px-3 py-2 text-xs text-red-300">
          {(sharpenMutation.error ?? deleteMutation.error)?.message}
        </div>
      )}

      {items.length === 0 && (
        <div className="flex flex-col items-center justify-center text-neutral-500 py-20">
          <p>No blurry photos detected.</p>
          <p className="text-xs mt-1 text-neutral-600">
            Click &quot;Scan for Blur&quot; to analyze un-scored photos.
          </p>
        </div>
      )}

      {items.map((item) => {
        const folder = item.file_path.split("/").slice(0, -1).join("/");
        return (
          <div
            key={item.id}
            className="flex items-center gap-4 bg-neutral-900 rounded-xl p-3 border border-neutral-800"
          >
            {/* Thumbnail */}
            <BlobImg
              src={ENDPOINTS.mediaThumbnail(item.id)}
              alt={item.filename}
              onClick={() => onItemClick?.(item.id)}
              fallback={
                <div
                  onClick={() => onItemClick?.(item.id)}
                  className="w-16 h-16 rounded-lg bg-neutral-800 shrink-0 cursor-pointer"
                />
              }
              className="w-16 h-16 rounded-lg object-cover bg-neutral-800 shrink-0 cursor-pointer hover:ring-2 hover:ring-indigo-500 transition-all"
            />

            {/* Metadata */}
            <div className="flex-1 min-w-0">
              <p className="text-sm text-neutral-200 truncate">{item.filename}</p>
              <p className="text-xs text-neutral-500 truncate">{folder}</p>
              <div className="mt-1.5 flex items-center gap-2">
                <div className="flex-1 h-1.5 bg-neutral-800 rounded-full overflow-hidden max-w-[120px]">
                  <div
                    className={`h-full rounded-full ${blurBarColor(item.blur_score)}`}
                    style={{ width: `${Math.min(100, item.blur_score * 100)}%` }}
                  />
                </div>
                <span className="text-[10px] text-neutral-500">
                  {(item.blur_score * 100).toFixed(1)}% — {blurLabel(item.blur_score)}
                </span>
              </div>
            </div>

            {/* Actions */}
            <div className="flex gap-1 shrink-0">
              <button
                title="Sharpen (saves to ~/.memoria/edits/)"
                disabled={pendingId?.id === item.id}
                onClick={() => {
                  setPendingId({ id: item.id, op: "sharpen" });
                  sharpenMutation.mutate(item.id);
                }}
                className="p-2 text-neutral-400 hover:text-blue-400 hover:bg-neutral-800 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {pendingId?.id === item.id && pendingId.op === "sharpen" ? (
                  <Loader2 size={16} className="animate-spin" />
                ) : (
                  <Wand2 size={16} />
                )}
              </button>
              <button
                title="Move to ~/.memoria/trash/ and remove from index"
                disabled={mediaReadOnly || pendingId?.id === item.id}
                onClick={() => {
                  if (!confirm(
                    `Move "${item.filename}" to ~/.memoria/trash/? It will be removed from the index.`
                  )) return;
                  setPendingId({ id: item.id, op: "delete" });
                  deleteMutation.mutate(item.id);
                }}
                className="p-2 text-neutral-400 hover:text-red-400 hover:bg-neutral-800 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {pendingId?.id === item.id && pendingId.op === "delete" ? (
                  <Loader2 size={16} className="animate-spin" />
                ) : (
                  <Trash2 size={16} />
                )}
              </button>
            </div>
          </div>
        );
      })}

      {/* Infinite scroll sentinel */}
      <div ref={sentinelRef} className="h-1" />
      {isFetchingNextPage && (
        <div className="flex justify-center py-4 text-neutral-500">
          <Loader2 size={18} className="animate-spin" />
        </div>
      )}
      {!hasNextPage && items.length > 0 && (
        <p className="text-center text-xs text-neutral-600 py-4">
          All {total} blurry photos loaded
        </p>
      )}
    </div>
  );
}
