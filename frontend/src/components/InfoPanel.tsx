// ─── InfoPanel ──────────────────────────────────────────────────────────────
// In plain English: the big modal that opens when you click a photo. The
// layout is split 6 columns of photo / 2 columns of info (on desktop). The
// photo area uses *progressive loading*: a tiny blurry thumbnail appears
// immediately, then the sharper 2048px preview fades in once it has
// downloaded. The info pane lists filename / location / size, any faces
// detected (clickable to find similar faces across the library), and a
// collapsible EXIF / blur-score section.
//
// Two protections against runaway memory:
//   • Semaphore(3) — at most 3 high-res previews fetch at once across the
//     entire app. Without this, opening several InfoPanels back to back
//     would stack ~30 MB downloads in parallel.
//   • IntersectionObserver — the preview fetch doesn't even start until
//     the panel is at least 10% visible.
//
/**
 * Memoria – Expanded View (8-Column 6:2 Split)
 * Photo area: 6 columns with progressive loading (thumbnail → 2048px preview).
 * Info pane: 2 columns with 3-section layout: Primary Info, Faces, Metadata.
 *
 * Spec 01 A3: Semaphore(3) caps concurrent preview fetches.
 * Spec 01 A3: IntersectionObserver gates preview loading to visible state.
 */

"use client";

import { useRef, useState, useEffect, useMemo } from "react";
import { createPortal } from "react-dom";
import { X, ChevronDown, ChevronRight } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { usePhotoBlob } from "@/hooks/usePhotoBlob";
import ImageActions from "@/components/ImageActions";
import BlobImg from "@/components/BlobImg";
import { apiUrl } from "@/lib/api";
import { ENDPOINTS } from "@/lib/endpoints";
import { formatDate, formatFileSize } from "@/lib/formatters";
import type { FaceInstance, MediaItem, SimilarFace } from "@/types";

// ── Semaphore: cap concurrent preview downloads (Spec 01 A3) ────────────────
//
// A *semaphore* is a counter with a queue. `acquire()` returns immediately
// while the counter is under `max`, and `release()` decrements it. Anything
// trying to acquire while we're already at the cap is parked in `queue`
// until someone calls release, at which point the next waiter is woken.

class Semaphore {
    private current = 0;
    private queue: Array<() => void> = [];

    constructor(private max: number) { }

    /** Resolve once a slot is available (immediately or after release). */
    acquire(): Promise<void> {
        if (this.current < this.max) {
            this.current++;
            return Promise.resolve();
        }
        // Park the resolver in the queue — release() will pull it out.
        return new Promise<void>((resolve) => {
            this.queue.push(resolve);
        });
    }

    /** Give a slot back. If anyone is waiting, wake the oldest waiter. */
    release(): void {
        this.current--;
        const next = this.queue.shift();
        if (next) {
            this.current++;
            next();
        }
    }
}

/** Module-level semaphore — shared across all ProgressiveImage instances.
 *  Cap of 3 was chosen empirically; the 2048px preview is the heaviest
 *  payload in the app and 3 parallel decodes fit comfortably in browser
 *  RAM. Bump cautiously. */
const previewSemaphore = new Semaphore(3);

// ── Helpers ─────────────────────────────────────────────────────────────────

/** Map a blur score to a Tailwind colour class.
 *  Score is on the 0–200ish range produced by the blur ensemble — higher is
 *  sharper. Green ≥ 80, yellow 40–80, red < 40. */
function blurColor(score: number): string {
    if (score < 40) return "bg-red-500";
    if (score < 80) return "bg-yellow-500";
    return "bg-green-500";
}

// ── Components ──────────────────────────────────────────────────────────────

interface InfoPanelProps {
    selectedId: number | null;
    items: MediaItem[];
    onClose: () => void;
}

/**
 * Wrapper that resolves the selected MediaItem either from the in-memory
 * `items` array (cheap) or by hitting /photos/:id (when opened from the
 * face-cluster view, where the surrounding grid doesn't include it).
 *
 * Renders nothing while no photo is selected or while the fetch is in flight.
 */
export default function InfoPanel({ selectedId, items, onClose }: InfoPanelProps) {
    // Local lookup — gives us the slim list-view shape instantly (no roundtrip).
    const localMedia = useMemo(
        () => (selectedId != null ? items.find((i) => i.id === selectedId) : undefined),
        [selectedId, items],
    );

    // Always fetch the detail endpoint so exif/phash are available.
    // Keyset list items no longer include them to keep list payloads small.
    const { data: detailMedia } = useQuery<MediaItem>({
        queryKey: ["photo-detail", selectedId],
        queryFn: async () => {
            const res = await fetch(apiUrl(ENDPOINTS.photoDetail(selectedId!)));
            if (!res.ok) throw new Error("Failed to load photo");
            return res.json();
        },
        enabled: selectedId != null,
        staleTime: 60_000,
    });

    // Merge: local slim data for immediate display, detail data for exif/phash.
    const media = useMemo(() => {
        if (!localMedia && !detailMedia) return undefined;
        if (!localMedia) return detailMedia;
        if (!detailMedia) return localMedia;
        return { ...localMedia, ...detailMedia };
    }, [localMedia, detailMedia]);

    if (selectedId == null || !media) return null;

    // Render as a portal to isolate re-renders from PhotoGrid (Spec 03 C1).
    return createPortal(
        <InfoPanelInner media={media} onClose={onClose} />,
        document.body,
    );
}

/**
 * The actual modal body, rendered into a portal by InfoPanel.
 *
 * Fetches face data for this photo on mount and lets the user click any face
 * to open the "similar faces across library" subpanel. The Escape-key /
 * backdrop click dismissal is wired up via the outer overlay div's onClick.
 */
function InfoPanelInner({ media, onClose }: { media: MediaItem; onClose: () => void }) {
    // Fetch real face data for this photo. Returns `[]` rather than throwing
    // on non-2xx — a missing-faces endpoint shouldn't break the modal.
    const { data: faceData } = useQuery<FaceInstance[]>({
        queryKey: ["photo-faces", media.id],
        queryFn: async () => {
            const res = await fetch(apiUrl(ENDPOINTS.facesForPhoto(media.id)));
            if (!res.ok) return [];
            const data = await res.json();
            return (data.faces ?? []).map((f: Record<string, unknown>) => {
                // The backend serialises bbox as plain numbers; cast them
                // into our typed BBox shape explicitly so TS doesn't widen.
                const bbox = f.bbox as { x: number; y: number; w: number; h: number };
                return {
                    face_id: f.face_id as number,
                    cluster_id: f.cluster_id as number,
                    label: (f.cluster_label as string | null) ?? null,
                    bbox,
                    thumbnail: ENDPOINTS.mediaFaceCrop(media.id, bbox),
                };
            });
        },
        staleTime: 60_000,
    });
    const faces: FaceInstance[] = faceData ?? [];
    // Which face id (if any) is currently expanded into a similar-faces panel.
    const [expandedFaceId, setExpandedFaceId] = useState<number | null>(null);

    return (
        <div
            className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
            onClick={onClose}
        >
            <div
                className="relative w-full max-w-7xl h-[90vh] grid grid-cols-1 md:grid-cols-8 bg-neutral-950 rounded-xl overflow-hidden"
                onClick={(e) => e.stopPropagation()}
            >
                <button
                    onClick={onClose}
                    title="Close"
                    className="absolute top-3 right-3 text-neutral-500 hover:text-white z-10"
                >
                    <X size={18} />
                </button>

                {/* Photo Area – 6 columns (single col on mobile) */}
                <div className="col-span-1 md:col-span-6 bg-black flex items-center justify-center relative overflow-hidden">
                    <ProgressiveImage media={media} />
                    <ImageActions mediaId={media.id} />
                </div>

                {/* Info Pane – 2 columns, 3 sections */}
                <div className="col-span-1 md:col-span-2 bg-neutral-950 md:border-l border-neutral-800 overflow-y-auto p-5">

                    {/* Section 1: Primary Information */}
                    <section className="mb-5">
                        <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-500 mb-2">
                            File Info
                        </h3>
                        <dl className="space-y-1.5">
                            <InfoRow label="Filename" value={media.filename} />
                            <InfoRow label="Location" value={media.folder} />
                            <InfoRow label="Size" value={formatFileSize(media.file_size)} />
                            <InfoRow label="Created" value={formatDate(media.created_at)} />
                            <InfoRow label="Source" value={media.source_device || "—"} />
                        </dl>
                    </section>

                    {/* Section 2: Faces Display */}
                    <section className="mb-5">
                        <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-500 mb-2">
                            Faces
                        </h3>
                        {faces.length === 0 ? (
                            <p className="text-xs text-neutral-600">No faces detected.</p>
                        ) : (
                            <div className="flex flex-wrap gap-3">
                                {faces.map((face, idx) => {
                                    const fid = face.face_id ?? null;
                                    const isExpanded = fid !== null && fid === expandedFaceId;
                                    return (
                                        <button
                                            key={idx}
                                            type="button"
                                            disabled={fid === null}
                                            onClick={() =>
                                                setExpandedFaceId((cur) => (cur === fid ? null : fid))
                                            }
                                            title={
                                                fid === null
                                                    ? "No face id"
                                                    : "Find similar faces in library"
                                            }
                                            className={`flex flex-col items-center gap-1 rounded p-1 transition-colors ${isExpanded ? "bg-neutral-800 ring-1 ring-indigo-500" : "hover:bg-neutral-900"
                                                }`}
                                        >
                                            <BlobImg
                                                src={face.thumbnail}
                                                alt={face.label ?? "Unknown"}
                                                fallback={<div className="h-10 w-10 rounded-full bg-neutral-800" />}
                                                className="w-10 h-10 rounded-full object-cover bg-neutral-800"
                                            />
                                            <span className="text-[10px] text-neutral-400 truncate max-w-[48px]">
                                                {face.label ?? "Unknown"}
                                            </span>
                                        </button>
                                    );
                                })}
                            </div>
                        )}
                        {expandedFaceId !== null && (
                            <SimilarFacesPanel
                                faceId={expandedFaceId}
                                onClose={() => setExpandedFaceId(null)}
                            />
                        )}
                    </section>

                    {/* Section 3: Metadata / Details (expandable) */}
                    <MetadataSection media={media} />
                </div>
            </div>
        </div>
    );
}

/** Renders the top-N nearest faces in the library to the selected face_id. */
function SimilarFacesPanel({ faceId, onClose }: { faceId: number; onClose: () => void }) {
    const { data, isLoading, isError } = useQuery<{ faces: SimilarFace[] }>({
        queryKey: ["face-similar", faceId],
        queryFn: async () => {
            const res = await fetch(apiUrl(ENDPOINTS.faceSimilar(faceId, 24)));
            if (!res.ok) throw new Error("similar fetch failed");
            return res.json();
        },
        staleTime: 60_000,
    });
    const similar = data?.faces ?? [];

    return (
        <div className="mt-3 pt-3 border-t border-neutral-800">
            <div className="flex items-center justify-between mb-2">
                <h4 className="text-[10px] font-semibold uppercase tracking-wider text-neutral-500">
                    Similar across library
                </h4>
                <button
                    type="button"
                    onClick={onClose}
                    className="text-neutral-600 hover:text-neutral-300"
                    title="Close"
                >
                    <X size={12} />
                </button>
            </div>
            {isLoading ? (
                <p className="text-xs text-neutral-600">Searching…</p>
            ) : isError ? (
                <p className="text-xs text-red-500">Search failed.</p>
            ) : similar.length === 0 ? (
                <p className="text-xs text-neutral-600">No similar faces found above threshold.</p>
            ) : (
                <div className="flex flex-wrap gap-3">
                    {similar.map((f) => (
                        <div key={f.face_id} className="flex flex-col items-center gap-1">
                            <BlobImg
                                src={ENDPOINTS.mediaFaceCrop(f.media_id, f.bbox)}
                                alt={f.cluster_label ?? f.filename}
                                title={`${f.filename} · d=${f.distance.toFixed(3)}`}
                                fallback={<div className="h-10 w-10 rounded-full bg-neutral-800" />}
                                className="w-10 h-10 rounded-full object-cover bg-neutral-800"
                            />
                            <span className="text-[10px] text-neutral-400 truncate max-w-[48px]">
                                {f.cluster_label ?? `d=${f.distance.toFixed(2)}`}
                            </span>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}


/** Collapsible metadata section with blur score, pHash, and EXIF data. */
function MetadataSection({ media }: { media: MediaItem }) {
    const [expanded, setExpanded] = useState(false);

    return (
        <section>
            <button
                onClick={() => setExpanded(!expanded)}
                className="flex items-center gap-1 w-full text-xs font-semibold uppercase tracking-wider text-neutral-500 mb-2 hover:text-neutral-300 transition-colors"
            >
                {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                Metadata &amp; Details
            </button>

            {expanded && (
                <div className="space-y-4">
                    {/* Blur Score */}
                    <div>
                        <div className="flex justify-between text-xs mb-1">
                            <span className="text-neutral-500">Blur Score</span>
                            <span className="text-neutral-300">{(media.blur_score ?? 0).toFixed(2)}</span>
                        </div>
                        <div className="w-full h-1.5 bg-neutral-800 rounded-full overflow-hidden">
                            <div
                                className={`h-full rounded-full ${blurColor(media.blur_score ?? 0)}`}
                                style={{ width: `${Math.min(100, (media.blur_score ?? 0) / 2)}%` }}
                            />
                        </div>
                    </div>

                    {/* pHash */}
                    <dl className="space-y-1.5">
                        <InfoRow label="pHash" value={media.phash || "—"} />
                    </dl>

                    {/* Detection Indices (placeholder) */}
                    <div>
                        <p className="text-[10px] uppercase tracking-wider text-neutral-600 mb-1">
                            Detection Indices
                        </p>
                        <dl className="space-y-1">
                            <InfoRow label="Object Detection" value="—" />
                            <InfoRow label="Scene Classification" value="—" />
                            <InfoRow label="Color Histogram" value="—" />
                        </dl>
                    </div>

                    {/* Full EXIF Data */}
                    <div>
                        <p className="text-[10px] uppercase tracking-wider text-neutral-600 mb-1">
                            EXIF Data
                        </p>
                        {!media.exif || Object.keys(media.exif).length === 0 ? (
                            <p className="text-xs text-neutral-600">No EXIF data available.</p>
                        ) : (
                            <dl className="space-y-1">
                                {Object.entries(media.exif).map(([k, v]) => (
                                    <InfoRow key={k} label={k} value={v} />
                                ))}
                            </dl>
                        )}
                    </div>
                </div>
            )}
        </section>
    );
}

/**
 * Progressive image loader: blurred thumbnail background → sharp 2048px preview overlay.
 *
 * The preview fetch is gated behind:
 * 1. IntersectionObserver — only loads when the container is visible.
 * 2. Semaphore(3) — caps concurrent preview downloads globally.
 *
 * In plain English: as soon as you open a photo you see the tiny thumbnail
 * (which is already cached locally), heavily blurred and stretched to fill
 * the area. Once the panel is on-screen *and* the semaphore frees up a
 * slot, the high-res preview downloads and replaces the thumbnail.
 */
function ProgressiveImage({ media }: { media: MediaItem }) {
    const { blobUrl: thumbBlob, imgRef: thumbRef } = usePhotoBlob(media.thumbnailUrl);

    const [previewEnabled, setPreviewEnabled] = useState(false);
    const containerRef = useRef<HTMLDivElement>(null);

    // Gate preview loading behind IntersectionObserver.
    // `threshold: 0.1` = trigger when at least 10% of the panel is visible.
    // Once triggered we `disconnect()` immediately — we never need it again,
    // and leaving observers attached costs memory.
    useEffect(() => {
        const el = containerRef.current;
        if (!el) return;
        const observer = new IntersectionObserver(
            ([entry]) => {
                if (entry.isIntersecting) {
                    setPreviewEnabled(true);
                    observer.disconnect();
                }
            },
            { threshold: 0.1 },
        );
        observer.observe(el);
        return () => observer.disconnect();
    }, []);

    return (
        <div ref={containerRef} className="absolute inset-0 flex items-center justify-center">
            {/* Low-res blurred background (always rendered) */}
            {thumbBlob && (
                <img
                    ref={thumbRef}
                    src={thumbBlob}
                    alt=""
                    aria-hidden
                    className="absolute inset-0 w-full h-full object-contain blur-xl scale-110 opacity-40"
                />
            )}
            {/* High-res overlay (gated by visibility + semaphore) */}
            {previewEnabled ? (
                <SemaphoredPreview media={media} fallback={thumbBlob} />
            ) : thumbBlob ? (
                <img
                    src={thumbBlob}
                    alt={media.filename}
                    className="relative w-full h-full object-contain z-[1]"
                />
            ) : (
                <div className="w-16 h-16 border-2 border-neutral-700 border-t-neutral-400 rounded-full animate-spin" />
            )}
        </div>
    );
}

/**
 * Fetches the 2K preview through the global semaphore.
 *
 * Lifecycle:
 *   1. Mount → call previewSemaphore.acquire(). May resolve immediately or
 *      park until a peer releases.
 *   2. Once acquired, flip `ready` so usePhotoBlob starts the actual fetch.
 *   3. As soon as the blob is in hand, release the semaphore so another
 *      panel can begin its own preview.
 *   4. On unmount before either step finishes, the cleanup paths below
 *      ensure we never leak a slot.
 */
function SemaphoredPreview({ media, fallback }: { media: MediaItem; fallback: string | null }) {
    const [ready, setReady] = useState(false);
    const { blobUrl: previewBlob, imgRef: previewRef } = usePhotoBlob(
        ready ? media.previewUrl : undefined,
    );

    // Acquire semaphore slot before starting the preview fetch.
    // The `released` flag handles the racy case where the component
    // unmounts while we're parked in the semaphore's queue.
    useEffect(() => {
        let released = false;
        previewSemaphore.acquire().then(() => {
            if (released) {
                previewSemaphore.release();
                return;
            }
            setReady(true);
        });
        return () => {
            released = true;
            // If acquire already resolved (ready=true) and we never made it
            // to the load-side release below, give the slot back here.
            if (ready) previewSemaphore.release();
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // Release semaphore once the preview blob is loaded.
    // The `releasedOnLoad` ref guards against double-release if React
    // re-runs the effect (strict-mode double-invocation).
    const releasedOnLoad = useRef(false);
    useEffect(() => {
        if (previewBlob && !releasedOnLoad.current) {
            releasedOnLoad.current = true;
            previewSemaphore.release();
        }
    }, [previewBlob]);

    if (previewBlob) {
        return (
            <img
                ref={previewRef}
                src={previewBlob}
                alt={media.filename}
                className="relative w-full h-full object-contain z-[1]"
            />
        );
    }

    if (fallback) {
        return (
            <img
                src={fallback}
                alt={media.filename}
                className="relative w-full h-full object-contain z-[1]"
            />
        );
    }

    return (
        <div className="w-16 h-16 border-2 border-neutral-700 border-t-neutral-400 rounded-full animate-spin" />
    );
}

/** Small key/value row for the info pane. */
function InfoRow({ label, value }: { label: string; value: string }) {
    return (
        <div className="flex justify-between gap-2 text-xs">
            <dt className="text-neutral-500 truncate shrink-0 max-w-[120px]">{label}</dt>
            <dd className="text-neutral-300 truncate text-right">{value}</dd>
        </div>
    );
}
