// ─── usePhotoBlob ───────────────────────────────────────────────────────────
// In plain English: a React hook that says "give me the bytes for this photo
// URL as an `<img>`-friendly object URL, and clean up after me when I
// unmount." Internally it routes through BlobPool so later consumers reuse a
// pooled object URL. Concurrent cache misses keep independent, abortable fetches.
//
// Two subtle responsibilities:
//   1. Per-effect AbortController — if the consumer unmounts mid-fetch, we
//      cancel the request rather than leak the response.
//   2. Refcount handoff — BlobPool increments the refcount when acquisition
//      succeeds; this hook decrements it on cleanup or after a late result.
//
/**
 * Memoria – usePhotoBlob hook (Spec 01 A2)
 *
 * Fetches a thumbnail or preview via the centralized BlobPool. Each effect
 * owns its own AbortController and refcount, so React 18 strict-mode double-
 * mount and TanStack Virtual cell churn don't cause "stuck gray cell" bugs
 * (those came from a previous shared-promise dedup model).
 *
 * The blob URL is held until the consumer's cleanup runs, at which point we
 * abort the in-flight fetch (if still pending) and release the entry's
 * refcount. If the fetch had already resolved, BlobPool.release() decrements
 * the refcount and leaves a zero-reference URL eligible for LRU eviction.
 */

import { useState, useEffect, useRef, useCallback } from "react";
import { apiUrl } from "@/lib/api";
import { BlobPool } from "./useBlobPool";

/**
 * Fetches the blob bytes for *url* through BlobPool and returns:
 *   • `blobUrl` — a `blob:...` string suitable for an `<img>` `src`. `null`
 *     until the fetch resolves, or after the hook unmounts.
 *   • `imgRef`  — a callback ref to attach to the `<img>` element. The hook
 *     uses it to clear `src` on unmount so Safari frees the decoded bitmap.
 *
 * Pass `undefined` to disable; the hook becomes a no-op.
 */
export function usePhotoBlob(url: string | undefined) {
  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);

  // Stable ref-setter callback. When React swaps the underlying <img>
  // element, detach any handlers from the old one so they don't fire
  // against a stale DOM node.
  const setImgRef = useCallback((node: HTMLImageElement | null) => {
    const prev = imgRef.current;
    if (prev && prev !== node) {
      prev.onload = null;
      prev.onerror = null;
    }
    imgRef.current = node;
    if (!node) return;
    node.onload = null;
    node.onerror = null;
  }, []);

  useEffect(() => {
    if (!url) return;

    let cancelled = false;
    let acquired = false;
    const poolKey = url;
    const controller = new AbortController();
    const fetchUrl = apiUrl(url);

    BlobPool.acquire(poolKey, () => {
      return fetch(fetchUrl, { signal: controller.signal }).then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.blob();
      });
    })
      .then((objectUrl) => {
        if (cancelled) {
          // We were unmounted before the fetch resolved. The entry has a
          // refcount for us; release it so the LRU can reclaim.
          BlobPool.release(poolKey);
          return;
        }
        acquired = true;
        setBlobUrl(objectUrl);
      })
      .catch((error: unknown) => {
        // Suppress errors that come from our own cancellation. AbortController
        // aborts during the response read surface as `AbortError`, but if the
        // abort fires during DNS / connection setup Chrome and Edge throw
        // `TypeError: Failed to fetch` instead. Either is expected when the
        // consumer unmounted before the request completed — only real network
        // failures (no abort flag set) should reach the console.
        if (cancelled || controller.signal.aborted) return;
        if (!(error instanceof Error && error.name === "AbortError")) {
          // eslint-disable-next-line no-console
          console.error("usePhotoBlob fetch failed:", fetchUrl, error);
        }
      });

    return () => {
      cancelled = true;
      // Abort the in-flight fetch (no-op if already resolved/rejected).
      controller.abort();
      // Drop any blob URL we wired into the <img> so Safari can release
      // its decoded bitmap (Spec 01 A2).
      if (imgRef.current) {
        imgRef.current.onload = null;
        imgRef.current.onerror = null;
        imgRef.current.src = "";
      }
      // If the fetch had already completed, our refcount lives in entries —
      // release it. If still pending or failed, the .then-with-cancelled
      // branch above handles the eventual release.
      if (acquired) {
        BlobPool.release(poolKey);
      }
      setBlobUrl(null);
    };
  }, [url]);

  return { blobUrl, imgRef: setImgRef };
}
