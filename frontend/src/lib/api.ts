// ─── lib/api ────────────────────────────────────────────────────────────────
// In plain English: every backend URL in the app is routed through `apiUrl()`
// so that, if someone runs the frontend on a different origin from the
// backend (rare; usually they share `localhost`), they can point at it by
// setting NEXT_PUBLIC_API_URL at build time. The fallback "" means "use the
// current origin", which is what next.config.ts's rewrites rely on.
//
/**
 * Memoria — shared API base + URL helpers.
 *
 * Every fetch in the app goes through apiUrl() so the NEXT_PUBLIC_API_URL
 * override (used when the frontend is served from a different origin than
 * the backend) is honored consistently.
 */

// Optional override; resolved at *build* time, not runtime, because Next.js
// inlines NEXT_PUBLIC_* vars into the bundle.
export const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

/** Resolve a relative backend path against API_BASE. Pass-through for absolute/data URLs. */
export function apiUrl(path: string): string {
    // Absolute and data URLs are returned untouched — these come from
    // user-facing inputs (e.g. an external image URL pasted in a future
    // feature) or from inline previews, neither of which the API_BASE
    // should be prepended to.
    if (path.startsWith("http://") || path.startsWith("https://") || path.startsWith("data:")) {
        return path;
    }
    const base = API_BASE.replace(/\/$/, "");
    const relativePath = path.startsWith("/") ? path : `/${path}`;
    return `${base}${relativePath}`;
}
