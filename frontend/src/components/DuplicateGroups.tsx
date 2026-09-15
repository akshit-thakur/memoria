// ─── DuplicateGroups ────────────────────────────────────────────────────────
// In plain English: the "Duplicates" tab. The backend computes SSCD embeddings
// for every photo and groups near-duplicates via HNSW cosine search. This view
// lists 20 groups per page, showing the first member as "Original" (kept) and
// the rest as "Duplicate" (candidates to delete). The % match badge shows how
// visually similar each duplicate is to the original (higher = closer match).
//
/**
 * Memoria — Duplicate Groups View
 *
 * Displays photos grouped by SSCD near-duplicate embedding similarity.
 * Each group shows a horizontal row of member thumbnails with similarity %.
 * Paginated at 20 groups per page.
 */

"use client";

import { useState } from "react";
import { Trash2, ChevronLeft, ChevronRight } from "lucide-react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import BlobImg from "@/components/BlobImg";
import { apiUrl } from "@/lib/api";
import { ENDPOINTS } from "@/lib/endpoints";
import { useStats } from "@/hooks/useStats";
import type { DuplicateGroup, MediaItem } from "@/types";

const PAGE_SIZE = 20;
const MAX_VISIBLE_DUPLICATES = 11; // original + 11 = 12 thumbs max before "show all"

interface DuplicateGroupsProps {
  onItemClick?: (item: MediaItem) => void;
}

interface DuplicatesResponse {
  groups: DuplicateGroup[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
  rebuilding: boolean;
}

function matchPercent(cosDistance: number | null | undefined): number | null {
  if (cosDistance == null) return null;
  return Math.round((1 - cosDistance) * 100);
}

export default function DuplicateGroups({ onItemClick }: DuplicateGroupsProps) {
  const qc = useQueryClient();
  const { data: runtimeStats } = useStats();
  const mediaReadOnly = runtimeStats?.media_read_only !== false;
  const [page, setPage] = useState(1);
  const [dismissed, setDismissed] = useState<Set<number>>(new Set());
  const [pendingId, setPendingId] = useState<number | null>(null);
  const [deletingAll, setDeletingAll] = useState(false);
  const [expandedGroups, setExpandedGroups] = useState<Set<number>>(new Set());
  const [actionError, setActionError] = useState<string | null>(null);

  const { data, isLoading } = useQuery<DuplicatesResponse>({
    queryKey: ["dedup-groups", page],
    queryFn: async () => {
      const res = await fetch(apiUrl(ENDPOINTS.dedupGroups(page, PAGE_SIZE)));
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    },
    staleTime: Infinity,
    refetchOnWindowFocus: false,
    refetchInterval: (query) =>
      query.state.data?.rebuilding ? 10_000 : false,
  });

  const deleteMutation = useMutation({
    mutationFn: async (groupId: number) => {
      const res = await fetch(
        apiUrl(ENDPOINTS.dedupDeleteDuplicates(groupId)),
        { method: "POST" },
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const result = await res.json() as { deleted_ids: number[]; failed_ids?: number[] };
      if (result.failed_ids?.length) {
        throw new Error(`${result.failed_ids.length} file(s) could not be moved to trash`);
      }
      return result;
    },
    onSuccess: (_res, groupId) => {
      setActionError(null);
      setDismissed((prev) => new Set(prev).add(groupId));
      qc.invalidateQueries({ queryKey: ["dedup-groups"] });
      qc.invalidateQueries({ queryKey: ["photos"] });
    },
    onError: (error) => setActionError(error.message),
    onSettled: () => {
      setPendingId(null);
      qc.invalidateQueries({ queryKey: ["dedup-groups"] });
      qc.invalidateQueries({ queryKey: ["photos"] });
    },
  });

  async function deleteAllOnPage() {
    if (!visibleGroups.length) return;
    if (!confirm(
      `Move duplicates from all ${visibleGroups.length} groups on this page to ~/.memoria/trash/? Originals will be kept.`
    )) return;
    setDeletingAll(true);
    setActionError(null);
    for (const g of visibleGroups) {
      try {
        const res = await fetch(apiUrl(ENDPOINTS.dedupDeleteDuplicates(g.id)), { method: "POST" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const result = await res.json() as { failed_ids?: number[] };
        if (result.failed_ids?.length) {
          throw new Error(`${result.failed_ids.length} file(s) could not be moved to trash`);
        }
        setDismissed((prev) => new Set(prev).add(g.id));
      } catch (error) {
        setActionError(error instanceof Error ? error.message : "Duplicate deletion failed");
        // continue with remaining groups even if one fails
      }
    }
    qc.invalidateQueries({ queryKey: ["dedup-groups"] });
    qc.invalidateQueries({ queryKey: ["photos"] });
    setDeletingAll(false);
  }

  const rebuilding = data?.rebuilding ?? false;

  if (isLoading && !data) {
    return (
      <div className="flex-1 flex items-center justify-center text-neutral-500">
        Loading duplicates…
      </div>
    );
  }

  const groups = data?.groups ?? [];
  const totalPages = data?.total_pages ?? 1;
  const total = data?.total ?? 0;

  if (groups.length === 0 && page === 1 && !rebuilding) {
    return (
      <div className="flex-1 flex items-center justify-center text-neutral-500">
        No duplicate groups found.
      </div>
    );
  }

  const visibleGroups = groups.filter((g) => !dismissed.has(g.id));

  return (
    <div className="flex-1 overflow-y-auto bg-black flex flex-col">
      {/* Rebuild-in-progress banner */}
      {rebuilding && (
        <div className="flex items-center gap-2 px-4 py-2 bg-indigo-950/60 border-b border-indigo-800/40 text-xs text-indigo-300">
          <svg className="animate-spin h-3 w-3 shrink-0" viewBox="0 0 24 24" fill="none">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 00-8 8h4z" />
          </svg>
          Scanning for duplicates — groups appear as they are found
        </div>
      )}
      {runtimeStats?.media_read_only === true && (
        <div className="px-4 py-2 border-b border-amber-800/40 bg-amber-950/30 text-xs text-amber-200">
          Source media is mounted read-only. You can review duplicates here,
          but file deletion is disabled in this deployment.
        </div>
      )}
      {actionError && (
        <div className="px-4 py-2 border-b border-red-800/40 bg-red-950/30 text-xs text-red-300">
          {actionError}
        </div>
      )}
      {/* Toolbar */}
      <div className="sticky top-0 z-10 bg-black border-b border-neutral-800 px-4 py-2.5 flex items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <h2 className="text-sm font-semibold text-neutral-400 uppercase tracking-wider">
            Duplicate Groups
          </h2>
          <span className="text-xs text-neutral-600">
            {total} total · page {page}/{totalPages}
          </span>
        </div>
        <button
          disabled={mediaReadOnly || deletingAll || visibleGroups.length === 0}
          onClick={deleteAllOnPage}
          className="flex items-center gap-1.5 text-xs text-red-400 hover:text-red-300 transition-colors px-3 py-1.5 rounded-lg hover:bg-red-500/10 disabled:opacity-40 disabled:cursor-not-allowed border border-red-900/40"
        >
          <Trash2 size={13} />
          {deletingAll ? "Deleting…" : `Delete all ${visibleGroups.length} groups`}
        </button>
      </div>

      {/* Groups */}
      <div className="p-4 space-y-4 flex-1">
        {visibleGroups.length === 0 && rebuilding && (
          <div className="flex items-center justify-center h-32 text-neutral-600 text-sm">
            Finding duplicate groups…
          </div>
        )}
        {visibleGroups.map((group) => {
          const original = group.members[0];
          const allDuplicates = group.members.slice(1);
          const isExpanded = expandedGroups.has(group.id);
          const duplicates = isExpanded
            ? allDuplicates
            : allDuplicates.slice(0, MAX_VISIBLE_DUPLICATES);
          const hiddenCount = allDuplicates.length - MAX_VISIBLE_DUPLICATES;
          return (
            <div
              key={group.id}
              className="rounded-xl border border-neutral-800 overflow-hidden"
            >
              {/* Group header */}
              <div className="flex items-center justify-between px-4 py-2 bg-neutral-900/60">
                <span className="text-xs text-neutral-400">
                  {group.members.length} photos
                </span>
                <button
                  disabled={mediaReadOnly || pendingId === group.id || deletingAll}
                  onClick={() => {
                    if (!confirm(
                      `Move ${allDuplicates.length} duplicate file(s) to ~/.memoria/trash/? The original will be kept.`
                    )) return;
                    setPendingId(group.id);
                    deleteMutation.mutate(group.id);
                  }}
                  className="flex items-center gap-1.5 text-xs text-red-400 hover:text-red-300 transition-colors px-2 py-1 rounded-lg hover:bg-red-500/10 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Trash2 size={13} />
                  {pendingId === group.id ? "Deleting…" : "Delete duplicates"}
                </button>
              </div>

              {/* Members row */}
              <div className="flex gap-3 p-4 overflow-x-auto">
                {/* Original */}
                <div
                  className="relative shrink-0 cursor-pointer"
                  onClick={() => onItemClick?.(original)}
                >
                  <BlobImg
                    src={original.thumbnailUrl}
                    alt={original.filename}
                    fallback={
                      <div className="w-28 h-28 rounded-lg bg-neutral-800 ring-2 ring-indigo-500" />
                    }
                    className="w-28 h-28 rounded-lg object-cover bg-neutral-800 ring-2 ring-indigo-500"
                  />
                  <span className="absolute top-1.5 left-1.5 text-[9px] bg-indigo-600/90 text-white px-1.5 py-0.5 rounded-full">
                    Original
                  </span>
                  <p className="text-[10px] text-neutral-400 mt-1 truncate w-28">
                    {original.filename}
                  </p>
                </div>

                {/* Divider */}
                <div className="w-px bg-neutral-800 shrink-0 my-1" />

                {/* Duplicates */}
                {duplicates.map((member) => {
                  const pct = matchPercent(member.cos_distance);
                  return (
                    <div
                      key={member.id}
                      className="relative shrink-0 cursor-pointer"
                      onClick={() => onItemClick?.(member)}
                    >
                      <BlobImg
                        src={member.thumbnailUrl}
                        alt={member.filename}
                        fallback={
                          <div className="w-28 h-28 rounded-lg bg-neutral-800 opacity-80" />
                        }
                        className="w-28 h-28 rounded-lg object-cover bg-neutral-800 hover:ring-2 hover:ring-neutral-600 transition-all opacity-80"
                      />
                      <span className="absolute top-1.5 left-1.5 text-[9px] bg-neutral-700/90 text-neutral-300 px-1.5 py-0.5 rounded-full">
                        Duplicate
                      </span>
                      {pct !== null && (
                        <span
                          className={`absolute top-1.5 right-1.5 text-[9px] px-1.5 py-0.5 rounded-full font-medium ${
                            pct >= 95
                              ? "bg-emerald-700/90 text-emerald-100"
                              : pct >= 80
                              ? "bg-yellow-700/90 text-yellow-100"
                              : "bg-orange-700/90 text-orange-100"
                          }`}
                        >
                          {pct}%
                        </span>
                      )}
                      <p className="text-[10px] text-neutral-500 mt-1 truncate w-28">
                        {member.filename}
                      </p>
                    </div>
                  );
                })}
              </div>
              {!isExpanded && hiddenCount > 0 && (
                <button
                  onClick={() => setExpandedGroups((s) => new Set([...s, group.id]))}
                  className="w-full py-1.5 text-xs text-neutral-500 hover:text-neutral-300
                             bg-neutral-900/40 hover:bg-neutral-900 transition-colors border-t border-neutral-800"
                >
                  +{hiddenCount} more
                </button>
              )}
            </div>
          );
        })}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="sticky bottom-0 bg-black border-t border-neutral-800 px-4 py-2.5 flex items-center justify-between">
          <button
            disabled={page <= 1}
            onClick={() => { setPage((p) => p - 1); setDismissed(new Set()); setExpandedGroups(new Set()); }}
            className="flex items-center gap-1 text-xs text-neutral-400 hover:text-neutral-200 disabled:opacity-30 disabled:cursor-not-allowed px-2 py-1 rounded-lg hover:bg-neutral-800 transition-colors"
          >
            <ChevronLeft size={14} /> Prev
          </button>
          <span className="text-xs text-neutral-600">
            Page {page} of {totalPages}
          </span>
          <button
            disabled={page >= totalPages}
            onClick={() => { setPage((p) => p + 1); setDismissed(new Set()); setExpandedGroups(new Set()); }}
            className="flex items-center gap-1 text-xs text-neutral-400 hover:text-neutral-200 disabled:opacity-30 disabled:cursor-not-allowed px-2 py-1 rounded-lg hover:bg-neutral-800 transition-colors"
          >
            Next <ChevronRight size={14} />
          </button>
        </div>
      )}
    </div>
  );
}
