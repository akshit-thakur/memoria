// ─── BlobImg ────────────────────────────────────────────────────────────────
// In plain English: a thin wrapper around `<img>` that routes the photo's URL
// through the BlobPool (see hooks/useBlobPool.ts). Every photo, face crop,
// and preview in the UI uses this component — never a raw `<img src={…}>` —
// because that's the only way to keep Safari from leaking decoded bitmaps.
//
/**
 * Memoria — <BlobImg>
 *
 * Shared wrapper that routes a single <img> through the global BlobPool.
 * Prevents Safari bitmap retention when grids/lists render many images by
 * scoping fetch + revocation to this component's lifecycle.
 *
 * Drop-in for <img src={...} />; accepts the usual imaging-related HTML
 * attributes. `src` is the raw API URL (thumbnail/preview path); the pool
 * resolves it, fetches, and produces an object URL.
 */

"use client";

import React from "react";
import { usePhotoBlob } from "@/hooks/usePhotoBlob";

// Same shape as a normal <img>, minus `src` (we typed it ourselves to
// require a string|undefined) and `ref` (we supply our own ref).
interface BlobImgProps
    extends Omit<React.ImgHTMLAttributes<HTMLImageElement>, "src" | "ref"> {
    src: string | undefined;
    /** Element rendered while the blob URL is being fetched. */
    fallback?: React.ReactNode;
}

/**
 * Renders the fallback while the BlobPool fetches the image, then swaps to
 * a real `<img>` with `src` set to the resulting object URL. The `imgRef`
 * comes from `usePhotoBlob` and is what lets the hook reset `src=""` on
 * unmount to release the bitmap.
 */
export default function BlobImg({ src, fallback = null, ...rest }: BlobImgProps) {
    const { blobUrl, imgRef } = usePhotoBlob(src);

    // While the pool is fetching, show whatever the caller passed as the
    // placeholder. `null` is the default — useful when the parent already
    // renders its own skeleton box behind us.
    if (!blobUrl) {
        return <>{fallback}</>;
    }
    return <img ref={imgRef} src={blobUrl} {...rest} />;
}
