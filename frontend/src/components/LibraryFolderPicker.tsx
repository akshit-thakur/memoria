"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { FolderOpen, Loader2, ScanSearch } from "lucide-react";
import { useState } from "react";
import { apiUrl } from "@/lib/api";
import { ENDPOINTS } from "@/lib/endpoints";
import { pickDirectory, type PickedDirectory } from "@/lib/pickDirectory";

interface LibraryFolderPickerProps {
  scannedFolders: string[];
  onScannedFoldersChange: (folders: string[]) => void;
}

interface LibraryScanPayload {
  folder_name: string;
  files: { relative_path: string }[];
  total_files: number;
}

interface LibraryScanResponse {
  status: string;
  folder_name: string;
  path: string;
  matched_files: number | null;
  sampled_files?: number | null;
}

/** Build a small, deterministic fingerprint without serializing the library. */
export function buildLibraryScanPayload(selection: PickedDirectory): LibraryScanPayload {
  return {
    folder_name: selection.folder_name,
    files: selection.files.map((entry) => ({ relative_path: entry.relative_path })),
    total_files: selection.total_files,
  };
}

export default function LibraryFolderPicker({
  scannedFolders,
  onScannedFoldersChange,
}: LibraryFolderPickerProps) {
  const queryClient = useQueryClient();
  const [picked, setPicked] = useState<PickedDirectory | null>(null);
  const [picking, setPicking] = useState(false);
  const [pickProgress, setPickProgress] = useState<number | null>(null);
  const [pickError, setPickError] = useState<string | null>(null);

  const scanMutation = useMutation<
    LibraryScanResponse,
    Error,
    PickedDirectory
  >({
    mutationFn: async (selection) => {
      const res = await fetch(apiUrl(ENDPOINTS.libraryScan), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildLibraryScanPayload(selection)),
      });
      if (!res.ok) {
        const payload = await res.json().catch(() => ({}));
        throw new Error(payload.detail ?? `HTTP ${res.status}`);
      }
      return res.json() as Promise<LibraryScanResponse>;
    },
    onSuccess: (result) => {
      if (!scannedFolders.includes(result.path)) {
        onScannedFoldersChange([result.path, ...scannedFolders]);
      }
      queryClient.invalidateQueries({ queryKey: ["stats"] });
      queryClient.invalidateQueries({ queryKey: ["progress"] });
      queryClient.invalidateQueries({ queryKey: ["processing-runs"] });
    },
  });

  const handlePick = async () => {
    setPickError(null);
    setPicking(true);
    setPickProgress(null);
    try {
      const selection = await pickDirectory((count) => setPickProgress(count));
      setPicked(selection);
    } catch (err) {
      if (err instanceof Error && err.message === "Folder selection cancelled") return;
      setPickError(err instanceof Error ? err.message : "Failed to pick folder");
    } finally {
      setPicking(false);
      setPickProgress(null);
    }
  };

  return (
    <div className="space-y-3">
      <div className="rounded-lg border border-neutral-800 bg-neutral-950/70 p-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="text-xs uppercase tracking-wide text-neutral-500">Library folder</div>
            <div className="mt-1 flex items-center gap-2 text-sm text-neutral-200">
              <FolderOpen size={16} className="shrink-0 text-emerald-400" />
              <span className="truncate" title={picked?.folder_name}>
                {picked?.folder_name ?? "No folder selected"}
              </span>
            </div>
            {picked && (
              <p className="mt-1 text-xs text-neutral-500">
                {picked.total_files} supported media file{picked.total_files === 1 ? "" : "s"} found.
                Only a small sample of relative file names is sent to the server — no uploads.
              </p>
            )}
          </div>
          <button
            type="button"
            disabled={picking || scanMutation.isPending}
            onClick={handlePick}
            className="inline-flex items-center gap-2 rounded-md border border-neutral-700 px-3 py-1.5 text-sm text-neutral-200 hover:bg-neutral-800 disabled:opacity-50"
          >
            {picking && <Loader2 size={14} className="animate-spin" />}
            {picking
              ? pickProgress !== null
                ? `Reading… ${pickProgress}`
                : "Opening picker…"
              : "Choose folder"}
          </button>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          disabled={!picked || !picked.total_files || !picked.files.length || scanMutation.isPending || picking}
          onClick={() => picked && scanMutation.mutate(picked)}
          className="inline-flex items-center gap-2 rounded-md border border-emerald-600/60 bg-emerald-950/30 px-3 py-1.5 text-sm text-emerald-100 hover:bg-emerald-950/50 disabled:opacity-50"
        >
          <ScanSearch size={14} className={scanMutation.isPending ? "animate-spin" : ""} />
          {scanMutation.isPending ? "Queueing scan…" : "Scan selected folder"}
        </button>
      </div>

      {pickError && <p className="text-xs text-red-400">{pickError}</p>}

      {scanMutation.error && (
        <p className="text-xs text-red-400">{scanMutation.error.message}</p>
      )}

      {scannedFolders.length > 0 && (
        <div className="space-y-2">
          <div className="text-xs uppercase tracking-wide text-neutral-500">Recently scanned</div>
          <div className="space-y-1">
            {scannedFolders.map((folder) => (
              <div
                key={folder}
                className="rounded-md border border-neutral-800 bg-neutral-950/50 px-3 py-2 text-xs text-neutral-300"
              >
                {folder}
              </div>
            ))}
          </div>
        </div>
      )}

      <p className="text-[11px] leading-5 text-neutral-500">
        Use your system folder picker to choose a directory. Docker can resolve
        only folders exposed by <code className="text-neutral-400">MEMORIA_MEDIA_PATH</code>{" "}
        or the optional read-only <code className="text-neutral-400">MEMORIA_HOST_MOUNT</code>.
        Restart the stack after changing either bind path.
      </p>
    </div>
  );
}
