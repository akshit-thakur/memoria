// ─── ErrorBoundary ──────────────────────────────────────────────────────────
// In plain English: a safety net. Wraps a part of the UI; if anything inside
// throws an exception during render, this component catches it and shows a
// small red box with a Retry button instead of letting the whole page go
// blank.
//
// We use one ErrorBoundary per top-level view (Sidebar, Library, Search,
// Duplicates, Faces, Blurry, Map, InfoPanel) so a failure in, say, the face
// grid doesn't kill the photo grid.
//
/**
 * Memoria — ErrorBoundary
 *
 * Catches render/effect errors inside a subtree and shows a fallback UI
 * instead of white-screening the whole app. Labelled per mount-point so
 * errors are scoped (e.g. a failed face cluster doesn't kill the photo grid).
 */

"use client";

import React from "react";
import { AlertTriangle } from "lucide-react";

interface Props {
  children: React.ReactNode;
  /** Short human label like "Library" or "Info Panel" — appears in the
   *  fallback message and in the console error. */
  label?: string;
  /** Optional caller-supplied reset hook (e.g. to clear a selection state
   *  on the parent before Retry tries to render again). */
  onReset?: () => void;
}

interface State {
  error: Error | null;
}

// Class component — required by React for error boundaries; they cannot be
// implemented with hooks alone. This is the only class in the frontend.
export default class ErrorBoundary extends React.Component<Props, State> {
  state: State = { error: null };

  // React calls this when a descendant throws. Returning a state object
  // causes the next render to use the fallback UI.
  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  // Side-effect channel for the same event — used here only for logging.
  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error("[ErrorBoundary]", this.props.label ?? "(unlabeled)", error, info);
  }

  // Retry handler bound to `this` via class-field arrow syntax.
  reset = () => {
    this.setState({ error: null });
    this.props.onReset?.();
  };

  render() {
    if (this.state.error) {
      return (
        <div className="flex-1 flex items-center justify-center bg-black p-6">
          <div className="max-w-md text-center bg-neutral-900 border border-red-900/50 rounded-xl p-6">
            <AlertTriangle className="mx-auto mb-3 text-red-400" size={32} />
            <p className="text-sm font-semibold text-neutral-200 mb-1">
              Something went wrong{this.props.label ? ` in ${this.props.label}` : ""}.
            </p>
            <p className="text-xs text-neutral-500 mb-4 break-all">
              {this.state.error.message}
            </p>
            <button
              onClick={this.reset}
              className="px-4 py-1.5 rounded-lg text-xs font-medium bg-neutral-800 hover:bg-neutral-700 text-neutral-200 transition-colors"
            >
              Retry
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
