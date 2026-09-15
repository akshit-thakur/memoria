// ─── ImageActions ───────────────────────────────────────────────────────────
// In plain English: the three small icons (Crop, Filter, Rotate) that float
// at the bottom of the open photo modal. Clicking one sends a POST to the
// backend's `/api/v1/editor/...` route, which writes a *new* file under
// `~/.memoria/edits/` — the original photo is never modified. The output
// path is shown as a tiny breadcrumb under the buttons.
//
/**
 * Memoria — Image Actions Toolbar
 *
 * Floating pill bar with Crop / Filter / Rotate buttons.
 * POSTs to the editor routes which write non-destructive edits to
 * ~/.memoria/edits/ and record a row in media_edits.
 */

"use client";

import { useState } from "react";
import {
  Crop,
  SlidersHorizontal,
  RotateCw,
  Check,
  X,
  Loader2,
} from "lucide-react";
import { apiUrl } from "@/lib/api";
import { ENDPOINTS } from "@/lib/endpoints";

interface ImageActionsProps {
  mediaId: number;
}

// Which of the three buttons. Used as map keys for state and endpoints.
type ActionKey = "crop" | "filter" | "rotate";
// Lifecycle of one button: idle → pending (POST in flight) → success/error,
// then back to idle after the 2-second fade in `run()`.
type ActionState = "idle" | "pending" | "success" | "error";

const actions: { key: ActionKey; label: string; Icon: typeof Crop }[] = [
  { key: "crop", label: "Crop (center 80%)", Icon: Crop },
  { key: "filter", label: "Filter (grayscale)", Icon: SlidersHorizontal },
  { key: "rotate", label: "Rotate 90° CW", Icon: RotateCw },
];

const EDITOR_ENDPOINTS: Record<ActionKey, string> = {
  crop: ENDPOINTS.editorCrop,
  filter: ENDPOINTS.editorFilter,
  rotate: ENDPOINTS.editorRotate,
};

/**
 * POST to the chosen editor endpoint and return the saved file path.
 * Throws with the backend's `detail` message if the response is non-2xx.
 */
async function callEditorAction(action: ActionKey, mediaId: number) {
  const res = await fetch(apiUrl(EDITOR_ENDPOINTS[action]), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ photo_id: mediaId }),
  });
  if (!res.ok) {
    // The backend returns `{ "detail": "…" }` on FastAPI HTTPException.
    // Fall through to a generic HTTP-status error if the body isn't JSON.
    const detail = await res.json().catch(() => ({})) as { detail?: string };
    throw new Error(detail?.detail ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<{ output_path: string }>;
}

/**
 * Renders the three-button toolbar. State per button is tracked locally
 * (no global mutation), and each button reverts to "idle" two seconds after
 * the request completes so the user can re-trigger if they want.
 */
export default function ImageActions({ mediaId }: ImageActionsProps) {
  const [states, setStates] = useState<Record<ActionKey, ActionState>>({
    crop: "idle",
    filter: "idle",
    rotate: "idle",
  });
  const [lastOutput, setLastOutput] = useState<string | null>(null);

  const run = async (action: ActionKey) => {
    if (states[action] === "pending") return;
    setStates((s) => ({ ...s, [action]: "pending" }));
    try {
      const data = await callEditorAction(action, mediaId);
      setStates((s) => ({ ...s, [action]: "success" }));
      setLastOutput(data.output_path);
    } catch (err) {
      console.error(`[editor/${action}] failed`, err);
      setStates((s) => ({ ...s, [action]: "error" }));
    }
    setTimeout(
      () => setStates((s) => ({ ...s, [action]: "idle" })),
      2000,
    );
  };

  return (
    <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-10 flex flex-col items-center gap-1">
      <div className="flex items-center gap-1 bg-black/60 backdrop-blur-sm rounded-full px-3 py-2">
        {actions.map(({ key, label, Icon }) => {
          const state = states[key];
          const StatusIcon =
            state === "pending"
              ? Loader2
              : state === "success"
                ? Check
                : state === "error"
                  ? X
                  : Icon;
          return (
            <button
              key={key}
              title={label}
              disabled={state === "pending"}
              onClick={() => run(key)}
              className={`p-2 transition-colors rounded-full ${
                state === "success"
                  ? "text-emerald-400"
                  : state === "error"
                    ? "text-red-400"
                    : "text-neutral-400 hover:text-white hover:bg-white/10"
              } disabled:cursor-not-allowed`}
            >
              <StatusIcon
                size={18}
                className={state === "pending" ? "animate-spin" : ""}
              />
            </button>
          );
        })}
      </div>
      {lastOutput && (
        <p
          className="text-[10px] text-neutral-500 max-w-[400px] truncate"
          title={lastOutput}
        >
          Saved: {lastOutput.replace(/^.*\.memoria\//, "~/.memoria/")}
        </p>
      )}
    </div>
  );
}
