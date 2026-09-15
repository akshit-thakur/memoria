#!/usr/bin/env python3
"""Memoria — benchmark harness.

In plain English: a measurement script. Given a folder of sample photos
and a count, it (1) resets the in-process metrics, (2) enqueues N
``process_media`` tasks, (3) waits until the queue empties, (4) hits a
handful of representative API endpoints to populate read-side latency
counters, then (5) snapshots ``/api/v1/metrics`` and emits a markdown
table ready to paste into ``PERFORMANCE.md``.

Reproducible end-to-end measurement of the ingestion pipeline. Snapshots
``/api/v1/metrics``, enqueues N images, polls until the queue drains,
then prints a markdown table you can paste into ``PERFORMANCE.md``.

Prerequisites — the full stack must already be running:

    ./start.sh

Then, in a separate shell:

    python backend/bench.py --root <photos dir> --count 100
    python backend/bench.py --root <photos dir> --count 500 --no-reset
    python backend/bench.py --root <photos dir> --count 100 --output PERFORMANCE_run1.md

Notes:
- ``process_media(path)`` enqueues onto the Huey SQLite queue; the running
  consumer picks it up. The bench process never executes pipeline bodies.
- ``--reset`` (default) calls ``POST /api/v1/metrics/reset``, then waits a
  beat for the worker reset task to fire. Use ``--no-reset`` if you want to
  measure on top of pre-existing samples.
- ``MEMORIA_API`` env var overrides the API base (default
  ``http://localhost:8000``).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))

# Importing process_media has the side effect of registering Huey signal
# hooks and (in the consumer process only) starting the snapshotter. In the
# bench process we just want the enqueue side; both gates are satisfied.
from workers.tasks import process_media  # noqa: E402

API_BASE = os.environ.get("MEMORIA_API", "http://localhost:8000").rstrip("/")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".bmp", ".tiff", ".tif"}


def _http(method: str, path: str, *, timeout: float = 30.0) -> Any:
    url = f"{API_BASE}{path}"
    req = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
        if not body:
            return None
        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return body


def get(path: str, timeout: float = 30.0) -> Any:
    return _http("GET", path, timeout=timeout)


def post(path: str, timeout: float = 30.0) -> Any:
    return _http("POST", path, timeout=timeout)


def find_images(root: Path, count: int) -> list[Path]:
    """Walk *root* and return the first ``count`` image files (deterministic order)."""
    out: list[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        out.append(p)
        if len(out) >= count:
            break
    return out


def wait_for_queue_drain(*, max_wait_s: float, settle_s: float) -> dict:
    """Poll ``/api/v1/queue/stats`` until ``pending`` stays at 0 for ``settle_s``.

    Returns the final ``{pending, failed}`` snapshot.
    """
    deadline = time.monotonic() + max_wait_s
    last_pending: int | None = None
    settled_at: float | None = None
    last_logged = 0.0

    while time.monotonic() < deadline:
        try:
            stats = get("/api/v1/queue/stats")
        except urllib.error.URLError as exc:
            print(f"[bench] queue/stats error: {exc} — retrying", flush=True)
            time.sleep(1.0)
            continue
        pending = int(stats.get("pending", 0))
        failed = int(stats.get("failed", 0))

        now = time.monotonic()
        if pending != last_pending or now - last_logged > 5:
            print(f"[bench] pending={pending} failed={failed}", flush=True)
            last_logged = now
            last_pending = pending

        if pending == 0:
            if settled_at is None:
                settled_at = now
            elif now - settled_at >= settle_s:
                return stats
        else:
            settled_at = None

        time.sleep(1.0)

    raise TimeoutError(
        f"queue did not drain within {max_wait_s}s (last pending={last_pending})"
    )


def warm_api_endpoints(*, n: int = 10) -> None:
    """Hit a handful of API endpoints to populate ``api.request.duration_ms``.

    Without this the histograms would only contain ``/api/v1/queue/stats`` and
    ``/api/v1/metrics`` samples, which isn't what readers want quoted.
    """
    print(f"[bench] warming API ({n} hits each)...", flush=True)
    photo_ids: list[int] = []
    for _ in range(n):
        try:
            data = get("/api/v1/photos?limit=20")
            ids = [item["id"] for item in data.get("items", [])]
            photo_ids.extend(ids)
        except Exception as exc:
            print(f"[bench] /photos failed: {exc}", flush=True)
            break

    sample = list({*photo_ids})[:8]
    for pid in sample:
        try:
            get(f"/api/v1/media/{pid}/thumbnail")
        except Exception as exc:
            print(f"[bench] /thumbnail/{pid}: {exc}", flush=True)
        try:
            get(f"/api/v1/preview/{pid}")
        except Exception as exc:
            print(f"[bench] /preview/{pid}: {exc}", flush=True)

    for path in (
        "/api/v1/faces/clusters",
        "/api/v1/dedup/groups",
        "/api/v1/blurry/photos?limit=20",
        "/api/v1/map/locations",
    ):
        for _ in range(min(5, n)):
            try:
                get(path)
            except Exception as exc:
                print(f"[bench] {path}: {exc}", flush=True)
                break

    # Try semantic search if scene_engine + embeddings are populated.
    for q in ("portrait", "outdoor", "sunset", "people"):
        try:
            get(f"/api/v1/search/text?q={q}&limit=20")
        except Exception as exc:
            print(f"[bench] /search/text?q={q}: {exc}", flush=True)
            break

    # Try /faces/{id}/similar if any face embeddings exist.
    try:
        clusters = get("/api/v1/faces/clusters")
        clist = clusters.get("clusters", []) if isinstance(clusters, dict) else []
        if clist:
            cluster_id = clist[0]["id"]
            detail = get(f"/api/v1/faces/clusters/{cluster_id}")
            members = detail.get("members", []) if isinstance(detail, dict) else []
            for m in members[:5]:
                fid = m.get("face_id")
                if fid:
                    try:
                        get(f"/api/v1/faces/{fid}/similar?limit=10")
                    except Exception as exc:
                        print(f"[bench] /faces/{fid}/similar: {exc}", flush=True)
                        break
    except Exception as exc:
        print(f"[bench] face-similar warmup skipped: {exc}", flush=True)


# ── snapshot diff helpers ───────────────────────────────────────────────────

def _hist_view(hist: dict) -> dict | None:
    if not hist or hist.get("n", 0) == 0:
        return None
    return {
        "n": hist.get("n", 0),
        "mean": hist.get("mean"),
        "p50": hist.get("p50"),
        "p95": hist.get("p95"),
        "p99": hist.get("p99"),
        "max": hist.get("max"),
    }


def collect_per_process(snap: dict) -> dict[str, dict]:
    """Group the snapshot by source process for nicer reporting."""
    by_proc: dict[str, dict] = {}
    for name, payload in snap.get("processes", {}).items():
        by_proc[name] = {
            "histograms": payload.get("histograms", {}),
            "counters": payload.get("counters", {}),
            "written_at": payload.get("_written_ts"),
        }
    return by_proc


def counter_delta(after: dict, before: dict, key: str) -> int:
    a = (after.get(key) or {}).get("value", 0)
    b = (before.get(key) or {}).get("value", 0)
    return int(a - b)


# ── markdown rendering ──────────────────────────────────────────────────────

def fmt_ms(v: float | None) -> str:
    if v is None:
        return "—"
    if v < 1.0:
        return f"{v*1000:.0f} µs"
    if v < 100.0:
        return f"{v:.1f} ms"
    return f"{v:.0f} ms"


def render_markdown(
    *,
    args: argparse.Namespace,
    found_paths: list[Path],
    elapsed_s: float,
    queue_stats: dict,
    snap_after: dict,
    counter_deltas: dict[str, int],
    model_status: dict | None = None,
    model_benchmarks: dict | None = None,
) -> str:
    lines: list[str] = []
    by_proc = collect_per_process(snap_after)
    worker = by_proc.get("worker", {})
    api = by_proc.get("api", {})
    watcher = by_proc.get("watcher", {})

    n = len(found_paths)
    photos_per_sec = n / elapsed_s if elapsed_s > 0 else 0.0
    ms_per_photo = 1000.0 * elapsed_s / n if n > 0 else 0.0

    lines.append(f"# Memoria — Bench Run ({time.strftime('%Y-%m-%d %H:%M:%S')})")
    lines.append("")
    # Reports are commonly published; retain reproducible identity without
    # leaking the caller's absolute home or mount path.
    sample_label = args.root.name or "sample"
    lines.append(f"- **Sample root**: `{sample_label}` (basename only; absolute path redacted)")
    lines.append(f"- **Photos enqueued**: {n}")
    lines.append(f"- **Wall time**: {elapsed_s:.2f} s")
    lines.append(
        f"- **Throughput**: **{photos_per_sec:.2f} photos/sec** "
        f"(≈ {ms_per_photo:.0f} ms/photo end-to-end)"
    )
    lines.append(f"- **Failed (processing_log)**: {queue_stats.get('failed', 0)}")
    scene_status = (model_status or {}).get("configured") or {}
    if scene_status:
        lines.append(
            "- **Scene model**: "
            f"`{scene_status.get('model_key', 'unknown')}` · "
            f"{scene_status.get('embedding_dim', '?')}-d · "
            f"{scene_status.get('actual_provider') or 'provider resolved on load'} · "
            f"{scene_status.get('precision') or scene_status.get('default_precision', 'auto')}"
        )
    lines.append("")
    lines.append("> **BENCH-PROBE blocks are active in the source tree.** They emit the")
    lines.append("> watcher-scan, system-resource, GPU-per-stage, file-extension and")
    lines.append("> RAM-safeguard metrics this report depends on. To remove all probes")
    lines.append("> after benchmarking, run: `grep -rn \"BENCH-PROBE\" backend/` and")
    lines.append("> delete each delimited block. The instrumentation is self-contained")
    lines.append("> and has no callers elsewhere in the codebase.")
    lines.append("")

    # ── Worker pipeline per-stage breakdown ───────────────────────────
    lines.append("## Worker pipeline (per-stage latency)")
    lines.append("")
    lines.append("| Stage | n | mean | p50 | p95 | p99 | max |")
    lines.append("|------|---:|---:|---:|---:|---:|---:|")
    stage_keys = [
        ("worker.thumbnail", "Thumbnail (240×240 JPEG)"),
        ("worker.exif", "EXIF extraction"),
        ("worker.stat", "stat (file size + mime)"),
        ("worker.phash", "pHash (32×32 DCT)"),
        ("worker.blur", "Blur (6-model ensemble)"),
        ("worker.faces", "Faces (multi-model ensemble)"),
        ("worker.scene", "Scene/search image embedding"),
        ("worker.dedup", "Dedup (SSCD 512-d embedding)"),
        ("worker.sync.sidecar_pending", "Sync · sidecar pending"),
        ("worker.sync.db_upsert", "Sync · Postgres upsert"),
        ("worker.sync.sidecar_synced", "Sync · sidecar synced"),
    ]
    worker_h = worker.get("histograms", {})
    for key, label in stage_keys:
        view = _hist_view(worker_h.get(key))
        if not view:
            lines.append(f"| {label} | 0 | — | — | — | — | — |")
            continue
        lines.append(
            f"| {label} | {view['n']} | {fmt_ms(view['mean'])} "
            f"| {fmt_ms(view['p50'])} | {fmt_ms(view['p95'])} "
            f"| {fmt_ms(view['p99'])} | {fmt_ms(view['max'])} |"
        )
    lines.append("")

    # ── Process-media wall (per branch) ───────────────────────────────
    lines.append("## process_media — full vs synced-skip vs pending-retry")
    lines.append("")
    lines.append("| Branch | n | mean | p50 | p95 | p99 |")
    lines.append("|------|---:|---:|---:|---:|---:|")
    for stage in ("full", "synced_skip", "pending_retry"):
        key = f"worker.process_media|stage={stage}"
        view = _hist_view(worker_h.get(key))
        if not view:
            lines.append(f"| `stage={stage}` | 0 | — | — | — | — |")
            continue
        lines.append(
            f"| `stage={stage}` | {view['n']} | {fmt_ms(view['mean'])} "
            f"| {fmt_ms(view['p50'])} | {fmt_ms(view['p95'])} "
            f"| {fmt_ms(view['p99'])} |"
        )
    lines.append("")

    # ── Face sub-model breakdown ─────────────────────────────────────
    lines.append("## Face detection (sub-model breakdown)")
    lines.append("")
    lines.append("| Model | n | mean | p50 | p95 | max |")
    lines.append("|------|---:|---:|---:|---:|---:|")
    for model in ("mtcnn", "yunet", "insightface"):
        key = f"face.detector|model={model}"
        view = _hist_view(worker_h.get(key))
        if not view:
            lines.append(f"| {model} | 0 | — | — | — | — |")
            continue
        lines.append(
            f"| {model} | {view['n']} | {fmt_ms(view['mean'])} "
            f"| {fmt_ms(view['p50'])} | {fmt_ms(view['p95'])} | {fmt_ms(view['max'])} |"
        )
    merge_view = _hist_view(worker_h.get("face.merge"))
    embed_view = _hist_view(worker_h.get("face.embed"))
    if merge_view:
        lines.append(
            f"| consensus merge | {merge_view['n']} | {fmt_ms(merge_view['mean'])} "
            f"| {fmt_ms(merge_view['p50'])} | {fmt_ms(merge_view['p95'])} | {fmt_ms(merge_view['max'])} |"
        )
    if embed_view:
        lines.append(
            f"| embedding (FaceNet) | {embed_view['n']} | {fmt_ms(embed_view['mean'])} "
            f"| {fmt_ms(embed_view['p50'])} | {fmt_ms(embed_view['p95'])} | {fmt_ms(embed_view['max'])} |"
        )
    faces_count = _hist_view(worker_h.get("worker.faces.count"))
    if faces_count:
        lines.append("")
        lines.append(
            f"_faces detected per photo:_ mean={faces_count['mean']:.2f}, "
            f"p50={faces_count['p50']:.0f}, p95={faces_count['p95']:.0f}, "
            f"max={faces_count['max']:.0f} (n={faces_count['n']})"
        )
    lines.append("")

    # ── API endpoints ────────────────────────────────────────────────
    lines.append("## API latency (FastAPI middleware)")
    lines.append("")
    lines.append("| Endpoint | n | p50 | p95 | p99 | max |")
    lines.append("|------|---:|---:|---:|---:|---:|")
    api_h = api.get("histograms", {})
    interesting = sorted(
        (k for k in api_h if k.startswith("api.request.duration_ms|")),
        key=lambda k: api_h[k].get("n", 0),
        reverse=True,
    )
    for key in interesting[:15]:
        view = _hist_view(api_h[key])
        if not view:
            continue
        # key looks like: api.request.duration_ms|method=GET|route=/api/v1/photos
        tags = dict(part.split("=", 1) for part in key.split("|")[1:] if "=" in part)
        label = f"{tags.get('method','?')} {tags.get('route','?')}"
        lines.append(
            f"| `{label}` | {view['n']} | {fmt_ms(view['p50'])} "
            f"| {fmt_ms(view['p95'])} | {fmt_ms(view['p99'])} | {fmt_ms(view['max'])} |"
        )
    lines.append("")

    # ── pgvector kNN breakdown ────────────────────────────────────────
    knn_keys = [k for k in api_h if k.startswith("db.vector_knn|") and not k.startswith("db.vector_knn.hits")]
    if knn_keys:
        lines.append("## pgvector kNN (HNSW index) — DB-only time")
        lines.append("")
        lines.append("| Route | n | p50 | p95 | p99 | max |")
        lines.append("|------|---:|---:|---:|---:|---:|")
        for key in knn_keys:
            view = _hist_view(api_h[key])
            if not view:
                continue
            tags = dict(part.split("=", 1) for part in key.split("|")[1:] if "=" in part)
            lines.append(
                f"| `{tags.get('route','?')}` | {view['n']} "
                f"| {fmt_ms(view['p50'])} | {fmt_ms(view['p95'])} "
                f"| {fmt_ms(view['p99'])} | {fmt_ms(view['max'])} |"
            )
        lines.append("")

    # ── Cache hit rates (counters) ───────────────────────────────────
    api_c = api.get("counters", {})
    cache_lines: list[str] = []
    for tier in ("thumbnail", "preview", "face_crop"):
        hit = (api_c.get(f"cache.{tier}.hit") or {}).get("value", 0)
        miss = (api_c.get(f"cache.{tier}.miss") or {}).get("value", 0)
        total = hit + miss
        rate = (100.0 * hit / total) if total > 0 else 0.0
        cache_lines.append(f"| `{tier}` | {hit} | {miss} | {rate:.1f}% |")
    if cache_lines:
        lines.append("## Cache hit rate (image serving)")
        lines.append("")
        lines.append("| Cache tier | Hits | Misses | Hit rate |")
        lines.append("|------|---:|---:|---:|")
        lines.extend(cache_lines)
        lines.append("")

    # ── Batched pipeline (process_media_batch) ────────────────────────
    lines.append("## Batched pipeline (process_media_batch)")
    lines.append("")
    lines.append(
        "> **In plain English:** When `MEMORIA_BATCH_SIZE > 1`, photos are processed in "
        "groups. Batches of **≥ 8** photos run a 4-stage CPU/GPU double-buffer "
        "(`mode=pipelined`) so CPU decode of the second half overlaps with GPU inference "
        "on the first half. Smaller batches fall back to serial. Effective per-photo "
        "cost in batch mode = `mean / batch size`."
    )
    lines.append("")
    lines.append("| Mode | n batches | mean (batch) | p50 | p95 | max |")
    lines.append("|------|---:|---:|---:|---:|---:|")
    for mode in ("serial", "pipelined"):
        key = f"worker.batch|mode={mode}"
        # The actual tag includes both mode and n; match by prefix.
        matched = [k for k in worker_h if k.startswith(f"worker.batch|mode={mode}")]
        agg_n = 0
        means: list[float] = []
        p50s: list[float] = []
        p95s: list[float] = []
        maxs: list[float] = []
        for k in matched:
            view = _hist_view(worker_h[k])
            if not view:
                continue
            agg_n += view["n"]
            means.append(view.get("mean") or 0.0)
            p50s.append(view.get("p50") or 0.0)
            p95s.append(view.get("p95") or 0.0)
            maxs.append(view.get("max") or 0.0)
        if agg_n == 0:
            lines.append(f"| `{mode}` | 0 | — | — | — | — |")
            continue
        lines.append(
            f"| `{mode}` | {agg_n} | "
            f"{fmt_ms(statistics.mean(means)) if means else '—'} | "
            f"{fmt_ms(statistics.mean(p50s)) if p50s else '—'} | "
            f"{fmt_ms(statistics.mean(p95s)) if p95s else '—'} | "
            f"{fmt_ms(max(maxs)) if maxs else '—'} |"
        )
    # Per-batch photo count distribution.
    batch_size_keys = [k for k in worker_h if k.startswith("worker.batch.size|")]
    if batch_size_keys:
        lines.append("")
        lines.append("Per-batch photo count distribution:")
        lines.append("")
        lines.append("| Mode | n batches | mean | p50 | p95 | max |")
        lines.append("|------|---:|---:|---:|---:|---:|")
        for k in batch_size_keys:
            view = _hist_view(worker_h[k])
            if not view:
                continue
            tags = dict(part.split("=", 1) for part in k.split("|")[1:] if "=" in part)
            lines.append(
                f"| `{tags.get('mode','?')}` | {view['n']} | "
                f"{view['mean']:.1f} | {view['p50']:.0f} | "
                f"{view['p95']:.0f} | {view['max']:.0f} |"
            )
    lines.append("")

    # ── Huey task wall (scheduling overhead) ──────────────────────────
    task_keys = sorted(k for k in worker_h if k.startswith("worker.task.duration_ms|"))
    if task_keys:
        lines.append("## Huey task wall time (queue + execution)")
        lines.append("")
        lines.append(
            "> **In plain English:** End-to-end time per Huey task — includes the "
            "few ms of queue-dequeue overhead on top of the actual pipeline work. "
            "The gap between this and the matching `process_media` stage timer is the "
            "raw queue-scheduling cost. A healthy run shows ≤ 5 % overhead."
        )
        lines.append("")
        lines.append("| Task | n | mean | p50 | p95 | p99 |")
        lines.append("|------|---:|---:|---:|---:|---:|")
        for key in task_keys:
            view = _hist_view(worker_h[key])
            if not view:
                continue
            tags = dict(part.split("=", 1) for part in key.split("|")[1:] if "=" in part)
            lines.append(
                f"| `{tags.get('task','?')}` | {view['n']} | "
                f"{fmt_ms(view['mean'])} | {fmt_ms(view['p50'])} | "
                f"{fmt_ms(view['p95'])} | {fmt_ms(view['p99'])} |"
            )
        lines.append("")

    # ── System resources (CPU / RAM / GPU) — BENCH-PROBE ──────────────
    sys_keys_in_order = [
        ("worker.system.cpu_pct", "CPU utilisation (%)", "%"),
        ("worker.system.ram_pct", "System RAM (%)", "%"),
        ("worker.system.ram_used_gb", "System RAM used (GB)", "GB"),
        ("worker.system.gpu_used_mb", "VRAM allocated (MB)", "MB"),
        ("worker.system.gpu_total_mb", "VRAM total (MB)", "MB"),
        ("worker.system.gpu_util_pct", "GPU compute utilisation (%)", "%"),
        ("worker.system.gpu_mem_util_pct", "GPU mem-controller util (%)", "%"),
    ]
    if any(worker_h.get(k) and worker_h[k].get("n", 0) > 0 for k, _, _ in sys_keys_in_order):
        lines.append("## System resources during the run (BENCH-PROBE)")
        lines.append("")
        lines.append(
            "> **In plain English:** Was the bottleneck CPU, RAM, or GPU? Samples taken "
            "every 30 s by the worker process. Compare p95 VRAM with the total reported "
            "for this run; approaching the limit means lower the batch size. If GPU util "
            "stays low while CPU is pegged, the bottleneck is decode/EXIF on CPU, not "
            "inference."
        )
        lines.append("")
        lines.append("| Metric | n samples | mean | p50 | p95 | max | unit |")
        lines.append("|------|---:|---:|---:|---:|---:|---|")
        for key, label, unit in sys_keys_in_order:
            view = _hist_view(worker_h.get(key))
            if not view:
                lines.append(f"| {label} | 0 | — | — | — | — | {unit} |")
                continue
            lines.append(
                f"| {label} | {view['n']} | "
                f"{view['mean']:.1f} | {view['p50']:.1f} | "
                f"{view['p95']:.1f} | {view['max']:.1f} | {unit} |"
            )
        lines.append("")

    # ── GPU memory per pipeline stage — BENCH-PROBE ───────────────────
    gpu_stage_keys = sorted(
        k for k in worker_h if k.startswith("worker.gpu.allocated_mb|")
    )
    if gpu_stage_keys:
        lines.append("## GPU memory per pipeline stage (BENCH-PROBE)")
        lines.append("")
        lines.append(
            "> **In plain English:** VRAM allocated right after each GPU-bound step "
            "completes. The stage with the highest p95 is your peak; compare it with "
            "the accelerator total captured for the same run when diagnosing the "
            "memory kill-switch."
        )
        lines.append("")
        lines.append("| Stage | n | mean (MB) | p50 | p95 | max |")
        lines.append("|------|---:|---:|---:|---:|---:|")
        for key in gpu_stage_keys:
            view = _hist_view(worker_h[key])
            if not view:
                continue
            tags = dict(part.split("=", 1) for part in key.split("|")[1:] if "=" in part)
            lines.append(
                f"| `{tags.get('stage','?')}` | {view['n']} | "
                f"{view['mean']:.0f} | {view['p50']:.0f} | "
                f"{view['p95']:.0f} | {view['max']:.0f} |"
            )
        lines.append("")

    # ── File distribution (input set) — BENCH-PROBE ───────────────────
    worker_c = worker.get("counters", {})
    file_ext_keys = sorted(k for k in worker_c if k.startswith("worker.file_ext|"))
    file_size_view = _hist_view(worker_h.get("worker.file_size_mb"))
    if file_ext_keys or file_size_view:
        lines.append("## Input file distribution (BENCH-PROBE)")
        lines.append("")
        lines.append(
            "> **In plain English:** What was in the test set? HEIC and RAW formats "
            "decode much slower than JPEG/PNG (RAW especially — rawpy demosaics on "
            "CPU). If `worker.thumbnail` p95 is an outlier, cross-reference this table: "
            "a heavy tail of RAWs explains it."
        )
        lines.append("")
        if file_ext_keys:
            lines.append("| Extension | count |")
            lines.append("|------|---:|")
            for k in file_ext_keys:
                tags = dict(part.split("=", 1) for part in k.split("|")[1:] if "=" in part)
                cnt = worker_c[k].get("value", 0)
                lines.append(f"| `{tags.get('ext','?')}` | {cnt} |")
            lines.append("")
        if file_size_view:
            lines.append(
                f"_Input file size (MB):_ n={file_size_view['n']}, "
                f"mean={file_size_view['mean']:.2f}, "
                f"p50={file_size_view['p50']:.2f}, "
                f"p95={file_size_view['p95']:.2f}, "
                f"max={file_size_view['max']:.2f}"
            )
            lines.append("")

    # ── Watcher scan (BENCH-PROBE) ────────────────────────────────────
    scan_view = _hist_view(watcher.get("histograms", {}).get("watcher.scan.duration_ms"))
    watcher_c = watcher.get("counters", {})
    scan_counters = {
        "files_seen": watcher_c.get("watcher.scan.files_seen", {}).get("value", 0),
        "files_new": watcher_c.get("watcher.scan.files_new", {}).get("value", 0),
        "files_skipped": watcher_c.get("watcher.scan.files_skipped_mtime", {}).get("value", 0),
        "dirs_walked": watcher_c.get("watcher.scan.dirs_walked", {}).get("value", 0),
    }
    if scan_view or any(scan_counters.values()):
        lines.append("## Watcher — initial filesystem scan (BENCH-PROBE)")
        lines.append("")
        lines.append(
            "> **In plain English:** Time and counts for the filesystem discovery "
            "step. The watcher walks every directory under the crawl roots, "
            "consulting `scan_index.db` (SQLite, keyed on path/mtime/size) to skip "
            "files it already processed. On a fresh ingest, `files_new` should equal "
            "`files_seen`. On the second 25 GB run, `files_skipped_mtime` should "
            "dominate — that is the idempotency guarantee working."
        )
        lines.append("")
        if scan_view:
            lines.append(
                f"- **Scan wall time**: n={scan_view['n']} roots · "
                f"mean={fmt_ms(scan_view['mean'])} · "
                f"p95={fmt_ms(scan_view['p95'])} · "
                f"max={fmt_ms(scan_view['max'])}"
            )
        lines.append(f"- **Files seen**: {scan_counters['files_seen']:,}")
        lines.append(f"- **Files new / changed (enqueued)**: {scan_counters['files_new']:,}")
        lines.append(f"- **Files skipped (mtime unchanged)**: {scan_counters['files_skipped']:,}")
        lines.append(f"- **Directories walked**: {scan_counters['dirs_walked']:,}")
        seen = scan_counters["files_seen"]
        skip = scan_counters["files_skipped"]
        if seen > 0:
            lines.append(
                f"- **Idempotency hit rate**: {100.0 * skip / seen:.1f}% "
                f"of seen files were skipped (mtime match)"
            )
        lines.append("")

    # ── Error & RAM-skip summary (BENCH-PROBE + existing counters) ────
    all_counters: dict[str, int] = {}
    for proc_name, proc in by_proc.items():
        for k, v in proc.get("counters", {}).items():
            if any(needle in k for needle in
                   (".error", ".skipped_ram", ".safeguard", ".failed",
                    ".decode_error", ".lock_retry", "ram_safeguard")):
                all_counters[f"{proc_name}/{k}"] = int(v.get("value", 0))
    if all_counters:
        lines.append("## Errors and RAM-skipped steps (BENCH-PROBE)")
        lines.append("")
        lines.append(
            "> **In plain English:** Every per-step error and every step skipped "
            "because RAM was critical. A clean run is zero across the board. "
            "Non-zero `*.skipped_ram` rows mean system RAM was pressured enough that "
            "the safety circuit fired — usually tune by lowering "
            "`MEMORIA_BATCH_SIZE`. Non-zero `*.error` rows point at file corruption "
            "or decoder bugs; check `logs/worker.log` for stack traces."
        )
        lines.append("")
        lines.append("| Event | count |")
        lines.append("|------|---:|")
        for k in sorted(all_counters):
            cnt = all_counters[k]
            if cnt == 0:
                continue
            lines.append(f"| `{k}` | {cnt} |")
        # If everything is zero we still want to show the clean-run row.
        if not any(c > 0 for c in all_counters.values()):
            lines.append("| _all zero — clean run_ | 0 |")
        lines.append("")

    # ── Dedup rebuild + active scene-model text encode ───────────
    dedup_view = _hist_view(api_h.get("db.dedup_rebuild"))
    scene_text_view = _hist_view(api_h.get("scene.encode_text")) or _hist_view(
        api_h.get("clip.encode_text")
    )
    if dedup_view or scene_text_view:
        lines.append("## Misc API timings (dedup rebuild · scene text encode)")
        lines.append("")
        lines.append(
            "> **In plain English:** `db.dedup_rebuild` is the cost of re-running "
            "the HNSW lateral-join + union-find for the dedup endpoint — scales "
            "with library size. `scene.encode_text` is the active model's on-device text "
            "encode (one forward pass through the text tower), independent of "
            "library size."
        )
        lines.append("")
        lines.append("| Metric | n | mean | p50 | p95 | max |")
        lines.append("|------|---:|---:|---:|---:|---:|")
        if dedup_view:
            lines.append(
                f"| `db.dedup_rebuild` | {dedup_view['n']} | {fmt_ms(dedup_view['mean'])} | "
                f"{fmt_ms(dedup_view['p50'])} | {fmt_ms(dedup_view['p95'])} | "
                f"{fmt_ms(dedup_view['max'])} |"
            )
        if scene_text_view:
            lines.append(
                f"| `scene.encode_text` | {scene_text_view['n']} | {fmt_ms(scene_text_view['mean'])} | "
                f"{fmt_ms(scene_text_view['p50'])} | {fmt_ms(scene_text_view['p95'])} | "
                f"{fmt_ms(scene_text_view['max'])} |"
            )
        lines.append("")

    records = (model_benchmarks or {}).get("benchmarks", [])
    if records:
        lines.append("## Model-specific measurements")
        lines.append("")
        lines.append("| Model | operation | items | duration | items/sec | provider | precision |")
        lines.append("|------|------|---:|---:|---:|------|------|")
        for record in records[:20]:
            lines.append(
                f"| `{record.get('model_key', '?')}` | {record.get('operation', '?')} | "
                f"{record.get('items', 0)} | {fmt_ms(record.get('duration_ms'))} | "
                f"{float(record.get('items_per_second') or 0):.2f} | "
                f"{record.get('provider') or '—'} | {record.get('precision') or '—'} |"
            )
        lines.append("")

    # ── Counters (Δ during this run) ─────────────────────────────────
    if counter_deltas:
        lines.append("## Counter deltas during this run")
        lines.append("")
        lines.append("| Series | Δ |")
        lines.append("|------|---:|")
        for k in sorted(counter_deltas):
            v = counter_deltas[k]
            if v == 0:
                continue
            lines.append(f"| `{k}` | {v} |")
        lines.append("")

    # ── Watcher (if it ran during the bench) ──────────────────────────
    if watcher and watcher.get("histograms"):
        lines.append("## Watcher (separate process)")
        lines.append("")
        wh = watcher["histograms"]
        for key in ("watcher.enqueue", "watcher.enqueue_batch"):
            view = _hist_view(wh.get(key))
            if not view:
                continue
            lines.append(
                f"- `{key}` n={view['n']} p50={fmt_ms(view['p50'])} "
                f"p95={fmt_ms(view['p95'])}"
            )
        lines.append("")

    lines.append("---")
    lines.append("_Generated by `python backend/bench.py`. Histograms are over the_")
    lines.append("_last 2048 samples per series; counters are deltas from the start_")
    lines.append("_of this run when `--reset` is set._")
    return "\n".join(lines)


# ── main ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Memoria benchmark harness.")
    parser.add_argument("--root", required=True, type=Path, help="Folder of sample images.")
    parser.add_argument("--count", type=int, default=100, help="Number of photos to enqueue.")
    parser.add_argument("--max-wait", type=float, default=900.0,
                        help="Max seconds to wait for the queue to drain.")
    parser.add_argument("--settle", type=float, default=10.0,
                        help="Seconds the queue must stay at pending=0 before declaring done.")
    parser.add_argument("--no-reset", action="store_true",
                        help="Skip POST /api/v1/metrics/reset (measure on top of existing samples).")
    parser.add_argument("--no-warm-api", action="store_true",
                        help="Skip the API endpoint warm-up step.")
    parser.add_argument("--shuffle", action="store_true",
                        help="Randomise the order of selected images.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write the markdown report to this path (in addition to stdout).")
    args = parser.parse_args()

    if not args.root.exists() or not args.root.is_dir():
        print(f"--root not found or not a directory: {args.root}", file=sys.stderr)
        return 2

    print(f"[bench] Memoria API base: {API_BASE}")
    try:
        get("/health")
    except Exception as exc:
        print(f"[bench] API unreachable at {API_BASE}: {exc}", file=sys.stderr)
        return 3

    model_status: dict = {}
    try:
        response = get("/api/v1/processing/models")
        model_status = response.get("scene", response) if isinstance(response, dict) else {}
        configured = model_status.get("configured", {})
        if configured:
            print(
                "[bench] scene model: "
                f"{configured.get('model_key')} ({configured.get('embedding_dim')}-d)"
            )
    except Exception as exc:
        print(f"[bench] model descriptor unavailable: {exc}", flush=True)

    # Sample selection.
    paths = find_images(args.root, args.count)
    if not paths:
        print("[bench] no image files found under --root", file=sys.stderr)
        return 4
    if args.shuffle:
        random.shuffle(paths)
    print(f"[bench] selected {len(paths)} files under sample root `{args.root.name or 'sample'}`")

    # Optional reset.
    pre_snapshot: dict[str, dict] = {}
    if not args.no_reset:
        try:
            r = post("/api/v1/metrics/reset")
            print(f"[bench] reset: {r}")
            time.sleep(2.0)  # give the worker reset task time to fire
        except Exception as exc:
            print(f"[bench] reset failed: {exc} — continuing without reset", flush=True)
    else:
        try:
            pre_snapshot = collect_per_process(get("/api/v1/metrics"))
        except Exception as exc:
            print(f"[bench] pre-snapshot failed: {exc}", flush=True)

    # Enqueue.
    print("[bench] enqueueing tasks...", flush=True)
    enqueue_t0 = time.perf_counter()
    enqueued = 0
    for p in paths:
        try:
            process_media(str(p))
            enqueued += 1
        except Exception as exc:
            print(f"[bench] enqueue failed for {p}: {exc}", flush=True)
    enqueue_elapsed = time.perf_counter() - enqueue_t0
    print(f"[bench] enqueued {enqueued} tasks in {enqueue_elapsed:.2f}s")

    # Drain.
    drain_t0 = time.perf_counter()
    try:
        queue_stats = wait_for_queue_drain(
            max_wait_s=args.max_wait, settle_s=args.settle,
        )
    except TimeoutError as exc:
        print(f"[bench] {exc}", file=sys.stderr)
        return 5
    drain_elapsed = time.perf_counter() - drain_t0
    print(f"[bench] queue drained in {drain_elapsed:.2f}s; final={queue_stats}")

    # Wall = enqueue + drain (essentially just drain, since enqueue is tiny).
    elapsed_s = enqueue_elapsed + drain_elapsed

    # API warm-up so the latency table has actual data.
    if not args.no_warm_api:
        try:
            warm_api_endpoints(n=20)
        except Exception as exc:
            print(f"[bench] warm-api failed: {exc}", flush=True)

    # Final snapshot — also force the worker to flush by hitting any endpoint
    # that triggers it. The 30s snapshotter cadence already covers it; we
    # additionally sleep briefly so a snapshot generated mid-bench is fresh.
    time.sleep(2.0)
    snap_after_raw = get("/api/v1/metrics")
    snap_after_proc = collect_per_process(snap_after_raw)
    try:
        model_benchmarks = get("/api/v1/processing/benchmarks?limit=20")
    except Exception:
        model_benchmarks = {}

    # Counter deltas (rough — combines all processes).
    deltas: dict[str, int] = {}
    if not args.no_reset or not pre_snapshot:
        # Reset means everything in the post-snapshot is the bench's contribution.
        for proc_name, proc in snap_after_proc.items():
            for k, v in proc.get("counters", {}).items():
                deltas[f"{proc_name}/{k}"] = int(v.get("value", 0))
    else:
        for proc_name, proc in snap_after_proc.items():
            pre = pre_snapshot.get(proc_name, {}).get("counters", {})
            cur = proc.get("counters", {})
            for k in cur:
                deltas[f"{proc_name}/{k}"] = counter_delta(cur, pre, k)

    # If snapshot wasn't refreshed (worker file mtime > 30s old), warn.
    worker_age: float | None = None
    written = snap_after_proc.get("worker", {}).get("written_at")
    if written:
        worker_age = time.time() - float(written)
        if worker_age > 60:
            print(
                f"[bench] WARN: worker snapshot is {worker_age:.0f}s old — "
                f"the consumer may not be running. Some worker stages will read 0.",
                file=sys.stderr,
            )

    md = render_markdown(
        args=args,
        found_paths=paths,
        elapsed_s=elapsed_s,
        queue_stats=queue_stats,
        snap_after=snap_after_raw,
        counter_deltas=deltas,
        model_status=model_status,
        model_benchmarks=model_benchmarks,
    )
    print()
    print(md)

    if args.output:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(md, encoding="utf-8")
            print(f"[bench] wrote markdown to {args.output}", flush=True)
        except OSError as exc:
            print(f"[bench] failed to write {args.output}: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
