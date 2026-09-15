// ─── useMedia ───────────────────────────────────────────────────────────────
// In plain English: the hook that powers the main photo grid. It pulls the
// library 100 photos at a time using "keyset" (cursor) pagination — meaning
// instead of asking for "page 7" it asks for "everything that comes after
// (indexed_at=…, id=…)". That makes pagination stable even when new photos
// arrive between requests.
//
// It also tracks an in-memory set of ids the user has just deleted, so a
// removed photo disappears immediately even before the next refetch.
//
/**
 * Memoria – useMedia hook (Spec 04: Keyset Pagination)
 *
 * Fetches paginated media items using TanStack Query's useInfiniteQuery.
 * Uses cursor-based (keyset) pagination via the FastAPI /api/v1/photos endpoint.
 */

import { useInfiniteQuery, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useCallback, useState, useRef } from "react";
import { apiUrl } from "@/lib/api";
import { ENDPOINTS } from "@/lib/endpoints";
import { useVisibilityInterval } from "@/hooks/useVisibilityInterval";
import type { MediaItem } from "@/types";

// How many photos to request per page. The backend caps this server-side
// (default ~200) but we deliberately stay below that so each page is small
// enough to render in one frame.
const PAGE_SIZE = 100;

// The "where did we leave off" pointer the backend returns. Pairing
// indexed_at with id makes the key unique even if two rows share the same
// timestamp.
interface Cursor {
  indexed_at: string;
  id: number;
}

// Shape of one /api/v1/photos response page.
interface MediaResponse {
  items: MediaItem[];
  next_cursor: Cursor | null;
  has_more: boolean;
}

interface LibraryVersion {
  count: number;
  latest: { id: number; indexed_at: string } | null;
}

/**
 * Fetches one page of photos. `pageParam` is the cursor returned by the
 * previous call (undefined on the first call → first page).
 */
async function fetchPhotos({
  pageParam,
}: {
  pageParam?: Cursor;
}): Promise<MediaResponse> {
  const params = new URLSearchParams({ limit: String(PAGE_SIZE) });
  if (pageParam) {
    // Pass the cursor through as two separate query params — the backend
    // composes them into a `(indexed_at, id) < (cursor_at, cursor_id)`
    // condition against the keyset index.
    params.set("cursor_indexed_at", pageParam.indexed_at);
    params.set("cursor_id", String(pageParam.id));
  }
  const res = await fetch(apiUrl(`${ENDPOINTS.photosList}?${params}`));
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

/**
 * React hook that exposes the photo library as an infinite-scroll list.
 *
 * Returns:
 *   • `items`             — every photo fetched so far, minus any removed locally.
 *   • `total`             — `items.length` (convenience for header chrome).
 *   • `loading`           — true while the *first* page is being fetched.
 *   • `fetchNext`         — call this when the user scrolls near the bottom of the
 *                           virtualized grid; safe to call repeatedly (no-ops while
 *                           a page is in flight or there is nothing left).
 *   • `removeItem`        — locally hide a photo by id without waiting for refetch.
 *   • `newItemsAvailable` — true when the library has changed since last load.
 *   • `refreshLibrary`    — resets to page 1 when user clicks the refresh pill.
 */
export function useMedia() {
  // Locally-deleted photo ids — used to make a Delete action feel instant
  // even though the actual DB delete is async via the FastAPI mutation.
  const [removedIds, setRemovedIds] = useState<Set<number>>(new Set());
  const [newItemsAvailable, setNewItemsAvailable] = useState(false);

  const queryClient = useQueryClient();

  const { data, fetchNextPage, hasNextPage, isFetchingNextPage, isLoading } =
    useInfiniteQuery({
      queryKey: ["photos"],
      queryFn: fetchPhotos,
      initialPageParam: undefined as Cursor | undefined,
      // Tell TanStack Query how to derive the next pageParam from a response.
      // Returning `undefined` here signals "no more pages".
      getNextPageParam: (lastPage) => {
        if (!lastPage.has_more) return undefined;
        return lastPage.next_cursor ?? undefined;
      },
      // Never auto-refetch all pages — we poll /library/version instead and
      // show a pill to trigger a manual reset. This prevents the 3000-request
      // storm that would occur with refetchInterval on a 300k-item library.
      staleTime: Infinity,
      refetchOnWindowFocus: false,
    });

  // Track the version seen at last full load so we can detect changes.
  const seenVersionRef = useRef<{ count: number; latestId: number | null } | null>(null);

  // Poll /library/version every 30 s (single cheap request; pauses when tab hidden).
  useVisibilityInterval(
    useCallback(async () => {
      try {
        const res = await fetch(apiUrl(ENDPOINTS.libraryVersion));
        if (!res.ok) return;
        const ver: LibraryVersion = await res.json();
        const seen = seenVersionRef.current;
        // The photo page is keyset-paginated and does not include the full
        // library count. Treat the first cheap version response as the
        // baseline instead of comparing it with the number of loaded pages.
        if (seen === null) {
          seenVersionRef.current = {
            count: ver.count,
            latestId: ver.latest?.id ?? null,
          };
          return;
        }
        const changed =
          ver.count !== seen.count ||
          (ver.latest?.id ?? null) !== seen.latestId;
        if (changed) setNewItemsAvailable(true);
      } catch {
        // ignore network errors during polling
      }
    }, []),
    30_000,
  );

  // Flatten all fetched pages into a single list, dropping any locally
  // removed ids. Memoized on (data, removedIds) so TanStack Virtual doesn't
  // re-measure on unrelated renders.
  const items = useMemo(() => {
    if (!data) return [];
    const all = data.pages.flatMap((p) => p.items);
    if (removedIds.size === 0) return all;
    return all.filter((i) => !removedIds.has(i.id));
  }, [data, removedIds]);

  const total = items.length;

  // Idempotent next-page trigger. The grid calls this whenever the scroll
  // sentinel becomes visible; guards make repeated calls safe.
  const fetchNext = useCallback(() => {
    if (hasNextPage && !isFetchingNextPage) {
      fetchNextPage();
    }
  }, [hasNextPage, isFetchingNextPage, fetchNextPage]);

  // Optimistic local removal — wrap in a *new* Set so React detects the
  // state change (mutating the existing one would not trigger a re-render).
  const removeItem = useCallback((id: number) => {
    setRemovedIds((prev) => new Set(prev).add(id));
  }, []);

  // Reset to page 1 — drops all cached pages, clears removed ids, hides pill.
  const refreshLibrary = useCallback(() => {
    setRemovedIds(new Set());
    setNewItemsAvailable(false);
    seenVersionRef.current = null;
    queryClient.resetQueries({ queryKey: ["photos"] });
  }, [queryClient]);

  return {
    items,
    total,
    loading: isLoading,
    fetchNext,
    removeItem,
    newItemsAvailable,
    refreshLibrary,
  };
}
