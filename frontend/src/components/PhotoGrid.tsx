// ─── PhotoGrid ──────────────────────────────────────────────────────────────
// In plain English: the big scrollable grid of photo thumbnails in the
// middle of the screen. Photos are grouped by folder with a thin separator
// between groups. The grid is *virtualized* — only the rows currently on
// screen (plus a couple above and below for smoothness) actually exist in
// the DOM. That's the only way to render a 200k-photo library without
// melting the browser.
//
/**
 * Memoria – Virtualized Photo Grid Component
 * Center pane: borderless, infinite-scroll grid powered by TanStack Virtual.
 * Items are grouped by parent folder with a visual separator between groups.
 */

"use client";

import React, { useRef, useMemo, useState, useEffect } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import BlobImg from "@/components/BlobImg";
import type { MediaItem } from "@/types";

// Two row shapes: a thin folder header row, or a row of image cells.
// The virtualizer treats both the same — only their estimated height differs.
type VirtualRow =
    | { type: "header"; folder: string }
    | { type: "images"; items: MediaItem[] };

interface PhotoGridProps {
    items: MediaItem[];
    columnCount?: number;
    onItemClick?: (item: MediaItem) => void;
    onItemError?: (id: number) => void;
    onLoadMore?: () => void;
}

/**
 * Renders a virtualized grid of photo thumbnails.
 *
 * Props:
 *   • items       — full flat list of MediaItems (already sorted by caller).
 *   • columnCount — cells per image row. Defaults to 6.
 *   • onItemClick — fires when a cell is clicked (opens the InfoPanel).
 *   • onItemError — fires when a thumbnail fails to load (parent typically
 *                   removes the id from useMedia).
 *   • onLoadMore  — fires when the user nears the last loaded row.
 */
export default function PhotoGrid({ items, columnCount = 6, onItemClick, onItemError, onLoadMore }: PhotoGridProps) {
    const parentRef = useRef<HTMLDivElement>(null);

    // Track the container’s pixel width so row height estimates stay accurate
    // when the sidebar is collapsed / expanded (aspect-square rows resize with it).
    const [containerWidth, setContainerWidth] = useState(0);
    useEffect(() => {
        const el = parentRef.current;
        if (!el) return;
        const ro = new ResizeObserver((entries) => {
            setContainerWidth(entries[0].contentRect.width);
        });
        ro.observe(el);
        setContainerWidth(el.clientWidth);
        return () => ro.disconnect();
    }, []);

    // Each image row is a CSS grid of aspect-square cells, so its height equals
    // containerWidth / columnCount.  Header rows are always 40 px.
    const rowHeight = containerWidth > 0 ? Math.floor(containerWidth / columnCount) : 200;

    // Build flat list of virtual rows: folder header + image rows per group.
    // Iterating items twice (group then chunk) avoids re-grouping on each
    // render — the memo's dependency list is just (items, columnCount).
    const virtualRows = useMemo<VirtualRow[]>(() => {
        const groups = new Map<string, MediaItem[]>();
        for (const item of items) {
            const folder = item.folder ?? "Uncategorized";
            if (!groups.has(folder)) groups.set(folder, []);
            groups.get(folder)!.push(item);
        }
        const rows: VirtualRow[] = [];
        for (const [folder, groupItems] of groups) {
            rows.push({ type: "header", folder });
            // Chunk the folder's photos into rows of `columnCount`. The
            // final row may be partial — that's fine, the CSS grid will
            // just leave the trailing cells empty.
            for (let i = 0; i < groupItems.length; i += columnCount) {
                rows.push({ type: "images", items: groupItems.slice(i, i + columnCount) });
            }
        }
        return rows;
    }, [items, columnCount]);

    // The TanStack virtualizer. `overscan: 2` means render two rows above and
    // below the visible viewport so fast scrolls don't flash gray cells.
    const virtualizer = useVirtualizer({
        count: virtualRows.length,
        getScrollElement: () => parentRef.current,
        estimateSize: (i) => (virtualRows[i]?.type === "header" ? 40 : rowHeight),
        overscan: 2,
    });

    // When the container resizes (sidebar toggle), force the virtualizer to
    // re-estimate all row heights so items don’t overlap or leave gaps.
    useEffect(() => {
        if (containerWidth > 0) virtualizer.measure();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [containerWidth]);

    // Trigger onLoadMore when the user scrolls near the last loaded row.
    const virtualItems = virtualizer.getVirtualItems();
    const lastVirtualIndex = virtualItems[virtualItems.length - 1]?.index ?? -1;
    useEffect(() => {
        if (onLoadMore && lastVirtualIndex >= virtualRows.length - 3) {
            onLoadMore();
        }
    }, [lastVirtualIndex, virtualRows.length, onLoadMore]);

    return (
        <div ref={parentRef} className="flex-1 overflow-y-auto bg-black">
            <div
                className="relative w-full"
                style={{ height: `${virtualizer.getTotalSize()}px` }}
            >
                {virtualizer.getVirtualItems().map((vRow) => {
                    const row = virtualRows[vRow.index];
                    if (!row) return null;

                    if (row.type === "header") {
                        return (
                            <div
                                key={vRow.key}
                                className="absolute left-0 w-full px-3 flex items-center gap-3"
                                style={{
                                    height: `${vRow.size}px`,
                                    transform: `translateY(${vRow.start}px)`,
                                }}
                            >
                                <span className="text-xs font-semibold text-neutral-400 truncate shrink-0 max-w-[60%]">
                                    {row.folder}
                                </span>
                                <div className="flex-1 h-px bg-neutral-800" />
                            </div>
                        );
                    }

                    return (
                        <div
                            key={vRow.key}
                            className="absolute left-0 w-full grid gap-0"
                            style={{
                                height: `${vRow.size}px`,
                                transform: `translateY(${vRow.start}px)`,
                                gridTemplateColumns: `repeat(${columnCount}, 1fr)`,
                            }}
                        >
                            {row.items.map((item) => (
                                <GridCell
                                    key={item.id}
                                    item={item}
                                    onItemClick={onItemClick}
                                    onItemError={onItemError}
                                />
                            ))}
                        </div>
                    );
                })}
            </div>
        </div>
    );
}

/**
 * Individual grid cell — renders thumbnail directly via Next.js proxy (/api/*).
 *
 * Memoized with a custom equality function so that re-renders of the parent
 * (e.g. when scrolling produces a new array reference) don't force every
 * visible cell to re-render. We only re-render a cell if its item id,
 * thumbnail URL, or one of the click/error callbacks changed.
 */
const GridCell = React.memo(
    function GridCell({
        item,
        onItemClick,
        onItemError,
    }: {
        item: MediaItem;
        onItemClick?: (item: MediaItem) => void;
        onItemError?: (id: number) => void;
    }) {
        return (
            <div
                className="aspect-square overflow-hidden cursor-pointer"
                onClick={() => onItemClick?.(item)}
            >
                <BlobImg
                    src={item.thumbnailUrl}
                    alt={item.filename}
                    loading="lazy"
                    decoding="async"
                    onError={() => onItemError?.(item.id)}
                    fallback={
                        <div className="w-full h-full bg-neutral-900" />
                    }
                    className="w-full h-full object-cover bg-neutral-900 hover:scale-105 transition-transform duration-150"
                />
            </div>
        );
    },
    (prev, next) =>
        prev.item.id === next.item.id &&
        prev.item.thumbnailUrl === next.item.thumbnailUrl &&
        prev.onItemClick === next.onItemClick &&
        prev.onItemError === next.onItemError,
);
