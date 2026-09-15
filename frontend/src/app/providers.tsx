// ─── app/providers ──────────────────────────────────────────────────────────
// Wraps the app in a single TanStack Query client so all hooks share one
// cache. Two non-default settings:
//   • staleTime: 30s    — data is considered fresh for 30 s; within that
//                         window, queries are not re-fired automatically.
//   • refetchOnWindowFocus: false — don't fire a wave of refetches every
//                         time the user alt-tabs back to the browser.
//
"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";

/**
 * The QueryClient is created lazily inside `useState` so it survives
 * re-renders without being recreated. Recreating would wipe the cache.
 */
export default function Providers({ children }: { children: React.ReactNode }) {
    const [queryClient] = useState(
        () =>
            new QueryClient({
                defaultOptions: {
                    queries: {
                        staleTime: 30_000,
                        refetchOnWindowFocus: false,
                    },
                },
            })
    );

    return (
        <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
}
