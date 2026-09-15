// ─── useVisibilityInterval ──────────────────────────────────────────────────
// In plain English: a setInterval that pauses itself when the browser tab is
// in the background and resumes (firing immediately) when the user comes
// back. Used for polling endpoints like /stats — no point spending CPU when
// the user isn't looking.
//
/**
 * Memoria — useVisibilityInterval
 *
 * setInterval wrapper that only fires while `document.visibilityState === "visible"`.
 * Pauses when the tab is hidden and resumes (firing immediately) when it becomes
 * visible again. Pass enabled=false to disable the whole loop.
 */

"use client";

import { useEffect, useRef } from "react";

/**
 * Run *callback* every *intervalMs* milliseconds, but only while the tab is
 * visible.
 *
 * Args:
 *   • callback   — fired once immediately on activation and then every tick.
 *                  May return a promise; rejections are swallowed.
 *   • intervalMs — millisecond interval. Changing it restarts the loop.
 *   • enabled    — set to false to disable the whole effect. Default true.
 */
export function useVisibilityInterval(
  callback: () => void | Promise<void>,
  intervalMs: number,
  enabled: boolean = true,
) {
  // Ref-mirror the callback so we can read the latest version inside the
  // interval without restarting the timer every time the parent re-renders.
  const cbRef = useRef(callback);
  cbRef.current = callback;

  useEffect(() => {
    if (!enabled) return;
    let timerId: ReturnType<typeof setInterval> | null = null;

    // `void` discards the promise return so the linter doesn't complain
    // about an unhandled await — rejections are intentionally swallowed.
    const run = () => {
      void cbRef.current();
    };

    // Start the loop. Runs once immediately so the UI has fresh data
    // before the first interval elapses.
    const start = () => {
      if (timerId != null) return;
      run();
      timerId = setInterval(run, intervalMs);
    };
    const stop = () => {
      if (timerId != null) {
        clearInterval(timerId);
        timerId = null;
      }
    };

    // Tab visibility listener — start when the tab comes back, stop when it
    // goes away. Matches the spec's "no work while hidden" rule.
    const onVisibility = () => {
      if (document.visibilityState === "visible") start();
      else stop();
    };

    // SSR guard: `document` is undefined on the server. The visibility check
    // here prevents a stray tick from firing on hidden tabs at first mount.
    if (typeof document !== "undefined" && document.visibilityState === "visible") {
      start();
    }
    document.addEventListener("visibilitychange", onVisibility);

    return () => {
      stop();
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [intervalMs, enabled]);
}
