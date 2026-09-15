// ─── useBlobPool ────────────────────────────────────────────────────────────
// In plain English: every photo thumbnail or face crop in the UI is downloaded
// once as a binary blob, then handed to the browser as an "object URL" (a
// short string like `blob:http://localhost:3000/abc-123` that the browser
// resolves to the in-memory bytes). Holding too many of these at once eats
// RAM and crashes Safari. This file is the single source of truth for which
// blobs are alive, how many components are using each, and which to evict
// when we hit the cap.
//
// Vocab cheat-sheet:
//   • blob URL  — the temporary `blob:...` string above. Must be revoked via
//                 URL.revokeObjectURL() once nothing is showing it, or the
//                 browser leaks the bitmap.
//   • refCount  — how many React components currently reference the entry.
//                 0 means safe to evict.
//   • LRU       — "least-recently-used"; when full, drop the entry that
//                 hasn't been touched in the longest time (among those with
//                 refCount = 0).
//
/**
 * Memoria – BlobPool Singleton (Spec 01 A1)
 *
 * Centralized pool that tracks all live blob object URLs with LRU eviction.
 * Caps decoded bitmaps at MAX_POOL_SIZE to prevent Safari bitmap bloat and
 * keep VRAM-cousin RAM usage bounded.
 *
 * The previous version deduplicated in-flight fetches via a `pending` map.
 * That broke under React 18 strict-mode double-mount: the first mount's
 * cleanup aborted the controller; the second mount immediately re-acquired
 * the same key BEFORE the abort's microtask could clean `pending`, so it
 * inherited the about-to-reject promise. AbortError got suppressed and the
 * <img> stayed blank.
 *
 * Fix: each acquire() does its own fetch. After the fetch resolves we race-
 * check the entries map — first writer wins, subsequent winners just bump
 * the refcount. Two cells fetching the same URL pay one extra HTTP call
 * (browser cache absorbs it); aborts no longer share fate across mounts.
 *
 * Usage:
 *   const url = await BlobPool.acquire(id, () => fetch(...).then(r => r.blob()));
 *   // ... use url ...
 *   BlobPool.release(id);
 */

// Hard ceiling on simultaneously-cached blobs. Tuned for Safari, which is the
// browser that crashes first when this is exceeded (the others survive
// silently up to ~2× this number, but it's not free).
import { BLOB_POOL_MAX_ENTRIES } from "@/lib/constants";

// One pooled blob. `lastAccess` is a UTC timestamp (ms) used purely for
// LRU ordering during eviction — it has no functional meaning otherwise.
interface PoolEntry {
  objectUrl: string;
  refCount: number;
  lastAccess: number;
}

// The pool itself. One instance exists per browser tab, exported as the
// `BlobPool` constant at the bottom of this file.
class BlobPoolSingleton {
  private entries = new Map<string, PoolEntry>();

  /**
   * Acquire a blob URL for *id*. Increments refCount if the entry is already
   * pooled; otherwise runs *fetcher* and stores the result. Each caller owns
   * cancellation for its own request. If two callers acquire the same id concurrently, both fetch but
   * only the first to commit creates the object URL — the second bumps the
   * existing entry's refCount and discards its blob.
   */
  async acquire(
    id: string,
    fetcher: () => Promise<Blob>,
  ): Promise<string> {
    // Fast path: someone already downloaded this id. Bump the use count,
    // update its LRU timestamp, hand back the existing URL.
    const existing = this.entries.get(id);
    if (existing) {
      existing.refCount++;
      existing.lastAccess = Date.now();
      return existing.objectUrl;
    }

    const blob = await fetcher();

    // Race check: another caller might have committed an entry while we
    // were fetching. If so, drop ours and join theirs.
    // (In plain English: two grid cells asked for the same thumbnail at
    // the exact same time. Both fetched. Whichever finished first wrote
    // the entry; the loser throws away its copy and increments the
    // winner's refCount so both cells share one blob URL.)
    const raceWinner = this.entries.get(id);
    if (raceWinner) {
      raceWinner.refCount++;
      raceWinner.lastAccess = Date.now();
      return raceWinner.objectUrl;
    }

    // Make room if we're at the cap, then commit our new entry.
    this.evictIfNeeded();
    const objectUrl = URL.createObjectURL(blob);
    this.entries.set(id, {
      objectUrl,
      refCount: 1,
      lastAccess: Date.now(),
    });
    return objectUrl;
  }

  /**
   * Decrement refCount and leave an unused entry eligible for LRU eviction.
   * Called when a component unmounts or swaps to a different image.
   * Keeping zero-reference entries makes repeated virtual-grid mounts reuse
   * the blob. The pool still revokes least-recently-used unused entries before
   * admitting a new item and after a temporary all-in-use overflow subsides.
   */
  release(id: string): void {
    const entry = this.entries.get(id);
    if (!entry) return;
    entry.refCount = Math.max(0, entry.refCount - 1);
    this.evictToSize(BLOB_POOL_MAX_ENTRIES);
  }

  /** Number of currently live pool entries (for debugging / metrics). */
  get size(): number {
    return this.entries.size;
  }

  /** Revoke every entry that is not currently displayed. */
  clearUnused(): void {
    for (const [key, entry] of this.entries) {
      if (entry.refCount > 0) continue;
      URL.revokeObjectURL(entry.objectUrl);
      this.entries.delete(key);
    }
  }

  /**
   * Evict LRU entries with refCount 0 until we're under the cap.
   * Walks every entry, tracks the one with the oldest `lastAccess` whose
   * refCount is 0 (i.e. nothing on screen is still using it), and removes
   * it. Repeats until the map reaches the requested size.
   * If every entry is in use we stop early — the cap is soft in that case.
   */
  private evictIfNeeded(): void {
    this.evictToSize(BLOB_POOL_MAX_ENTRIES - 1);
  }

  private evictToSize(maxEntries: number): void {
    while (this.entries.size > maxEntries) {
      // Find the oldest entry that is safe to evict (nobody is referencing
      // it). Entries with refCount > 0 are skipped — evicting them would
      // leave a live <img> pointing at a revoked URL.
      let oldestKey: string | null = null;
      let oldestTime = Infinity;

      for (const [key, entry] of this.entries) {
        if (entry.refCount <= 0 && entry.lastAccess < oldestTime) {
          oldestTime = entry.lastAccess;
          oldestKey = key;
        }
      }

      if (oldestKey === null) break; // all entries are in use — can't evict
      const entry = this.entries.get(oldestKey)!;
      URL.revokeObjectURL(entry.objectUrl);
      this.entries.delete(oldestKey);
    }
  }
}

/** Module-level singleton — shared across all hook instances. */
export const BlobPool = new BlobPoolSingleton();
