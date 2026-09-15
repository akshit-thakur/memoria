// ─── SearchView ─────────────────────────────────────────────────────────────
// In plain English: the "Search" tab. The user types a description ("beach
// sunset", "my dog on the couch") and the backend converts that into a
// scene-model text embedding, then finds the photos whose image embedding is
// nearest by cosine distance (powered by pgvector's HNSW index).
//
// The typed query is *debounced*: we wait 350 ms after the user stops
// typing before firing the request. Results render through the same
// PhotoGrid component used by the Library view, so virtualization and
// blob caching come along for free.
//
/**
 * Memoria — Search view (Phase D)
 *
 * Top-bar text input → debounced query against /api/v1/search/text → results
 * rendered through the existing virtualized PhotoGrid (already on BlobPool).
 * Empty query state shows a hint; loading / empty / error states each render
 * a single short message.
 */

"use client";

import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Search as SearchIcon } from "lucide-react";
import PhotoGrid from "@/components/PhotoGrid";
import { apiUrl } from "@/lib/api";
import { ENDPOINTS } from "@/lib/endpoints";
import type { MediaItem } from "@/types";

interface SearchViewProps {
    onItemClick: (item: MediaItem) => void;
}

export default function SearchView({ onItemClick }: SearchViewProps) {
    const [input, setInput] = useState("");
    const [query, setQuery] = useState("");

    // Debounce: don't re-fire on every keystroke.
    useEffect(() => {
        const t = setTimeout(() => setQuery(input.trim()), 350);
        return () => clearTimeout(t);
    }, [input]);

    const { data, isLoading, isError } = useQuery<{ items: MediaItem[] }>({
        queryKey: ["search-text", query],
        queryFn: async ({ signal }) => {
            const res = await fetch(apiUrl(ENDPOINTS.searchText(query)), { signal });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            return res.json();
        },
        enabled: query.length > 0,
        staleTime: 30_000,
    });

    const items = data?.items ?? [];

    return (
        <div className="flex-1 flex flex-col bg-black overflow-hidden">
            <div className="border-b border-neutral-800 px-4 py-3 flex items-center gap-3 shrink-0">
                <SearchIcon size={18} className="text-neutral-500 shrink-0" />
                <input
                    autoFocus
                    type="text"
                    placeholder="Search your library — beach, sunset, my dog…"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    className="flex-1 bg-transparent text-sm text-neutral-200 placeholder-neutral-600 outline-none"
                />
                {input && (
                    <button
                        type="button"
                        onClick={() => setInput("")}
                        className="text-neutral-600 hover:text-neutral-300 text-xs"
                    >
                        clear
                    </button>
                )}
            </div>
            <div className="flex-1 overflow-hidden">
                {!query ? (
                    <div className="flex items-center justify-center h-full text-neutral-500 text-sm px-6 text-center">
                        Type a description above. Powered by the active scene model — works on natural language.
                    </div>
                ) : isLoading ? (
                    <div className="flex items-center justify-center h-full text-neutral-500 text-sm">
                        Searching…
                    </div>
                ) : isError ? (
                    <div className="flex items-center justify-center h-full text-red-500 text-sm px-6 text-center">
                        Search failed. The active scene encoder may not be available — check the backend logs.
                    </div>
                ) : items.length === 0 ? (
                    <div className="flex items-center justify-center h-full text-neutral-500 text-sm">
                        No results. Try a different phrasing.
                    </div>
                ) : (
                    <PhotoGrid
                        items={items}
                        onItemClick={onItemClick}
                        onItemError={() => { /* search results don't auto-prune */ }}
                        onLoadMore={() => { /* search returns a single page */ }}
                    />
                )}
            </div>
        </div>
    );
}
