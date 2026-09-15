// ─── lib/formatters ─────────────────────────────────────────────────────────
// In plain English: tiny pure functions for showing raw values to humans —
// e.g. turning `1572864` into `1.5 MB` and an ISO date into a short readable
// string. Used by InfoPanel and a few list views.
//
/**
 * Memoria — Display-only formatters
 *
 * Pure, UI-facing helpers. Never depends on React or any fetched data.
 */

/** Human-readable byte size. Returns "—" for undefined/zero. */
export function formatFileSize(bytes: number | undefined | null): string {
    if (!bytes) return "—";
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) {
        return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    }
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

/** Localized "Apr 23, 2026, 10:14" date display. "—" for missing. */
export function formatDate(iso: string | undefined | null): string {
    if (!iso) return "—";
    return new Date(iso).toLocaleDateString(undefined, {
        year: "numeric",
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
    });
}
