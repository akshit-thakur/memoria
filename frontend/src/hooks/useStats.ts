"use client";

import { useQuery } from "@tanstack/react-query";
import { apiUrl } from "@/lib/api";
import { ENDPOINTS } from "@/lib/endpoints";
import type { IndexingStats } from "@/types";

export type { IndexingStats } from "@/types";

function isIdle(stats: IndexingStats | undefined): boolean {
  if (!stats) return true;
  return (
    stats.queue_size === 0 &&
    !stats.current_file &&
    !stats.face_scan_active &&
    !stats.blur_scan_active &&
    !stats.scene_scan_active
  );
}

/**
 * Shared TanStack Query hook for /api/v1/stats.
 * All subscribers share one cache entry — one fetch per interval regardless
 * of how many components are mounted.
 * Polls at 5s while anything is active, 30s while idle.
 * Automatically pauses when the tab is backgrounded (refetchIntervalInBackground: false).
 */
export function useStats() {
  return useQuery<IndexingStats>({
    queryKey: ["stats"],
    queryFn: async ({ signal }) => {
      const res = await fetch(apiUrl(ENDPOINTS.stats), { signal });
      if (!res.ok) throw new Error(`stats HTTP ${res.status}`);
      return res.json() as Promise<IndexingStats>;
    },
    staleTime: 0,
    refetchInterval: (query) =>
      isIdle(query.state.data) ? 30_000 : 5_000,
    refetchIntervalInBackground: false,
  });
}
