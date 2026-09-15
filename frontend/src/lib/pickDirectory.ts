/** Media suffixes aligned with backend/constants.py MEDIA_EXTENSIONS. */
const MEDIA_EXTENSIONS = new Set([
  ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp",
  ".heic", ".heif", ".avif",
  ".raw", ".cr2", ".cr3", ".nef", ".arw", ".dng", ".rw2", ".orf", ".pef", ".raf", ".srw",
  ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv",
]);

export interface PickedFileEntry {
  relative_path: string;
}

export interface PickedDirectory {
  folder_name: string;
  total_files: number;
  /** Deterministic representative sample; never exceeds the exported limit. */
  files: PickedFileEntry[];
}

export const DIRECTORY_FILE_SAMPLE_LIMIT = 50;
const DIRECTORY_PROGRESS_INTERVAL = 1000;

interface ScoredPath extends PickedFileEntry {
  score: number;
}

interface DirectorySampleAccumulator {
  totalFiles: number;
  sample: ScoredPath[];
}

function isMediaFile(name: string): boolean {
  const dot = name.lastIndexOf(".");
  if (dot < 0) return false;
  return MEDIA_EXTENSIONS.has(name.slice(dot).toLowerCase());
}

function normalizeRelativePath(path: string): string {
  return path.replace(/\\/g, "/");
}

/** Stable FNV-1a score used to retain the same sample for the same paths. */
function pathScore(path: string): number {
  let hash = 0x811c9dc5;
  for (let index = 0; index < path.length; index += 1) {
    hash ^= path.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

function compareScoredPaths(left: ScoredPath, right: ScoredPath): number {
  const scoreOrder = left.score - right.score;
  if (scoreOrder) return scoreOrder;
  if (left.relative_path < right.relative_path) return -1;
  if (left.relative_path > right.relative_path) return 1;
  return 0;
}

/** Keep the globally lowest-scoring paths using O(limit) bounded memory. */
function addSampledPath(
  accumulator: DirectorySampleAccumulator,
  rawRelativePath: string,
  onProgress?: (count: number) => void,
): void {
  const relative_path = normalizeRelativePath(rawRelativePath);
  const candidate: ScoredPath = {
    relative_path,
    score: pathScore(relative_path),
  };
  accumulator.totalFiles += 1;
  if (
    accumulator.totalFiles === 1
    || accumulator.totalFiles % DIRECTORY_PROGRESS_INTERVAL === 0
  ) {
    onProgress?.(accumulator.totalFiles);
  }

  if (accumulator.sample.length < DIRECTORY_FILE_SAMPLE_LIMIT) {
    accumulator.sample.push(candidate);
    return;
  }

  let worstIndex = 0;
  for (let index = 1; index < accumulator.sample.length; index += 1) {
    if (compareScoredPaths(accumulator.sample[index], accumulator.sample[worstIndex]) > 0) {
      worstIndex = index;
    }
  }
  if (compareScoredPaths(candidate, accumulator.sample[worstIndex]) < 0) {
    accumulator.sample[worstIndex] = candidate;
  }
}

function pickedDirectory(
  folder_name: string,
  accumulator: DirectorySampleAccumulator,
): PickedDirectory {
  const files = accumulator.sample
    .slice()
    .sort(compareScoredPaths)
    .map(({ relative_path }) => ({ relative_path }));
  return { folder_name, total_files: accumulator.totalFiles, files };
}

async function walkDirectoryHandle(
  handle: FileSystemDirectoryHandle,
  prefix: string,
  accumulator: DirectorySampleAccumulator,
  onProgress?: (count: number) => void,
): Promise<void> {
  const iterator = handle.entries();

  while (true) {
    const next = await iterator.next();
    if (next.done) break;

    const [name, child] = next.value;
    if (name.startsWith(".")) continue;

    const relative = prefix ? `${prefix}/${name}` : name;
    if (child.kind === "directory") {
      await walkDirectoryHandle(
        child as FileSystemDirectoryHandle,
        relative,
        accumulator,
        onProgress,
      );
      continue;
    }

    if (!isMediaFile(name)) continue;
    addSampledPath(accumulator, relative, onProgress);
  }
}

function pickDirectoryViaInput(
  onProgress?: (count: number) => void,
): Promise<PickedDirectory> {
  return new Promise((resolve, reject) => {
    const input = document.createElement("input");
    input.type = "file";
    input.multiple = true;
    input.setAttribute("webkitdirectory", "");
    input.style.display = "none";
    document.body.appendChild(input);

    const cleanup = () => {
      input.remove();
    };

    input.addEventListener("change", () => {
      cleanup();
      const fileList = input.files;
      if (!fileList?.length) {
        reject(new Error("No folder selected"));
        return;
      }

      const firstPath = fileList[0].webkitRelativePath;
      const folderName = firstPath.split("/")[0] ?? "Selected folder";
      const accumulator: DirectorySampleAccumulator = { totalFiles: 0, sample: [] };

      for (const file of fileList) {
        if (!isMediaFile(file.name)) continue;
        const parts = file.webkitRelativePath.split("/");
        const relative = parts.slice(1).join("/");
        addSampledPath(accumulator, relative, onProgress);
      }

      onProgress?.(accumulator.totalFiles);
      resolve(pickedDirectory(folderName, accumulator));
    });

    input.addEventListener("cancel", () => {
      cleanup();
      reject(new Error("Folder selection cancelled"));
    });

    input.click();
  });
}

export function supportsNativeDirectoryPicker(): boolean {
  return typeof window !== "undefined" && "showDirectoryPicker" in window;
}

export async function pickDirectory(
  onProgress?: (count: number) => void,
): Promise<PickedDirectory> {
  if (supportsNativeDirectoryPicker()) {
    const handle = await window.showDirectoryPicker!({ mode: "read" });
    const accumulator: DirectorySampleAccumulator = { totalFiles: 0, sample: [] };
    await walkDirectoryHandle(handle, "", accumulator, onProgress);
    onProgress?.(accumulator.totalFiles);
    return pickedDirectory(handle.name, accumulator);
  }

  return pickDirectoryViaInput(onProgress);
}
