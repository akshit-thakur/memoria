"""Memoria – Huey Task Queue Configuration (SqliteHuey).

In plain English: this file configures the background-job system. Huey is
the lightweight task queue library; SqliteHuey is its backend that stores
tasks in a runtime-path-managed SQLite file
(``~/.memoria/data/queue.db`` by default) instead of
needing Redis or RabbitMQ. The watcher pushes work into this queue, and
the ``huey_consumer`` command drains it one task at a time.

Beyond the queue object itself, this module wires up *signal handlers*
that fire when a task starts, completes, errors, retries, or is
interrupted — each one records timings and counters into the in-process
metrics module so the API can surface them at ``/api/v1/metrics``.

A 30-second snapshotter flushes those numbers to disk
(``~/.memoria/metrics/worker.json``) so the FastAPI process can read
them across the process boundary.

Replaces the previous Celery + Redis setup with Huey backed by SQLite,
eliminating the Redis dependency and reducing RAM footprint.
"""

import ctypes
import ctypes.util
import logging
import os
import sys
import time
from pathlib import Path

# glibc malloc_trim — returns unused heap pages to the OS after each task.
# Mitigates Python's tendency to retain brk-allocated pages indefinitely.
_libc = None
try:
    _libc_name = ctypes.util.find_library("c")
    if _libc_name:
        _libc = ctypes.CDLL(_libc_name, use_errno=True)
        _libc.malloc_trim.argtypes = [ctypes.c_size_t]
        _libc.malloc_trim.restype = ctypes.c_int
except Exception:
    _libc = None


def _malloc_trim() -> None:
    if _libc is not None:
        try:
            _libc.malloc_trim(0)
        except Exception:
            pass

# Make sibling backend modules importable before resolving shared runtime
# paths; huey_consumer may start with only backend/workers on sys.path.
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from runtime_paths import CURRENT_FILE_PATH, DATA_DIR, QUEUE_DB_PATH  # noqa: E402
from database_config import DATABASE_URL  # noqa: E402
from huey import SqliteHuey
from huey.signals import (
    SIGNAL_COMPLETE,
    SIGNAL_ERROR,
    SIGNAL_EXECUTING,
    SIGNAL_INTERRUPTED,
    SIGNAL_RETRYING,
)

DATA_DIR.mkdir(parents=True, exist_ok=True)
QUEUE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

QUEUE_DB = str(QUEUE_DB_PATH)

huey = SqliteHuey("memoria", filename=QUEUE_DB, immediate=False)

# Alias kept so existing `from workers import app` references still work
# during migration; new code should import `huey` directly.
app = huey

import metrics  # noqa: E402

_signal_logger = logging.getLogger("memoria.worker.signals")
_task_starts: dict[str, float] = {}


def _clear_active_media_marker() -> None:
    """Clear task-local progress after every terminal Huey signal.

    Media tasks normally clear these files themselves on success, but an
    exception, retry, or consumer interruption exits before their success
    epilogue.  The consumer is deliberately single-threaded, so no other task
    can legitimately own the marker when a terminal signal is emitted.
    """
    try:
        CURRENT_FILE_PATH.write_text("")
    except Exception as exc:
        _signal_logger.debug("current-file cleanup failed: %s", exc)
    try:
        from workers import heartbeat as _hb

        _hb.touch(
            current_file="",
            current_started_ts=None,
            paused_memory=False,
        )
    except Exception as exc:
        _signal_logger.debug("heartbeat cleanup failed: %s", exc)

# Long-running flush so /api/v1/metrics in the API process can see worker-side
# histograms (face detection, scene encoding, per-stage timings). The thread
# dies with the consumer; one final flush happens via atexit. Only the
# `huey_consumer` process starts the snapshotter — other importers (the
# bench script, the watcher, ad-hoc `from workers.tasks import …`) would
# otherwise stomp the same `worker.json` file.
_arg0 = (sys.argv[0] if sys.argv else "").lower()
_IS_HUEY_CONSUMER = (
    "huey_consumer" in _arg0
    or os.environ.get("MEMORIA_HUEY_CONSUMER") == "1"
)
if _IS_HUEY_CONSUMER:
    import atexit  # noqa: E402

    # A hard-killed previous consumer cannot emit a terminal Huey signal.
    # Clear its durable progress markers before this consumer starts polling.
    _clear_active_media_marker()

    _metrics_snapshotter = metrics.PeriodicSnapshotter("worker", interval_s=30.0)
    _metrics_snapshotter.start()
    atexit.register(_metrics_snapshotter.stop)

    # Flip any orphaned 'running' rows to 'failed' on startup — a previous
    # consumer crash could have left rows stuck in 'running' forever.
    try:
        import psycopg as _psycopg
        with _psycopg.connect(DATABASE_URL, autocommit=True) as _conn:
            _conn.execute(
                "UPDATE processing_log SET status='failed', error_msg='orphaned: worker restart'"
                " WHERE status='running'"
            )
    except Exception as _e:
        _signal_logger.warning("Orphan cleanup failed: %s", _e)


# ┌─ BENCH-PROBE ──────────────────────────────────────────────────────────────┐
# │ Periodic CPU / RAM / GPU sampler. Remove after bench run.                  │
# │ Grep "BENCH-PROBE" in backend/ to find every removable block.              │
# │                                                                             │
# │ What this gives the bench report:                                          │
# │   - worker.system.cpu_pct      : CPU utilisation %                          │
# │   - worker.system.ram_pct      : system RAM utilisation %                   │
# │   - worker.system.ram_used_gb  : absolute RAM used                         │
# │   - worker.system.gpu_used_mb  : VRAM allocated on the CUDA card           │
# │   - worker.system.gpu_total_mb : total VRAM the driver reports             │
# │   - worker.system.gpu_util_pct : GPU compute utilisation (nvidia-smi)      │
# │   - worker.system.gpu_mem_util_pct : GPU memory-controller utilisation     │
# │                                                                             │
# │ Samples every 30 s while the worker process is alive.                       │
# └─────────────────────────────────────────────────────────────────────────────┘
import threading as _threading  # noqa: E402
import subprocess as _subprocess  # noqa: E402


def _bench_sample_system_resources():
    """One-shot snapshot of CPU%, RAM%, GPU MB, GPU utilisation %."""
    try:
        import psutil as _psutil
        metrics.record("worker.system.cpu_pct", _psutil.cpu_percent(interval=None))
        vm = _psutil.virtual_memory()
        metrics.record("worker.system.ram_pct", vm.percent)
        metrics.record("worker.system.ram_used_gb", vm.used / (1024 ** 3))
    except Exception:
        pass
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            free_b, total_b = _torch.cuda.mem_get_info()
            used_mb = (total_b - free_b) / (1024 * 1024)
            metrics.record("worker.system.gpu_used_mb", used_mb)
            metrics.record("worker.system.gpu_total_mb", total_b / (1024 * 1024))
    except Exception:
        pass
    try:
        r = _subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            parts = r.stdout.strip().split(",")
            if len(parts) >= 2:
                metrics.record("worker.system.gpu_util_pct", float(parts[0].strip()))
                metrics.record("worker.system.gpu_mem_util_pct", float(parts[1].strip()))
    except Exception:
        pass


def _bench_system_sample_loop():
    from workers import heartbeat as _hb
    from memory_mgmt import monitor as _mem_monitor
    while True:
        _bench_sample_system_resources()
        try:
            _hb.touch(pressure=_mem_monitor.pressure())
        except Exception:
            pass
        time.sleep(30)


if _IS_HUEY_CONSUMER:
    _bench_sys_probe_thread = _threading.Thread(
        target=_bench_system_sample_loop,
        name="bench-system-probe",
        daemon=True,
    )
    _bench_sys_probe_thread.start()
    _signal_logger.info("[BENCH] system resource sampler started (30 s cadence).")
# ── END BENCH-PROBE ───────────────────────────────────────────────────────────


@huey.signal(SIGNAL_EXECUTING)
def _on_executing(signal, task, exc=None):
    try:
        _task_starts[task.id] = time.perf_counter()
    except Exception:
        pass


@huey.signal(SIGNAL_COMPLETE)
def _on_complete(signal, task, exc=None):
    try:
        start = _task_starts.pop(task.id, None)
        if start is not None:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            metrics.record("worker.task.duration_ms", elapsed_ms, task=task.name)
        metrics.count("worker.task.complete", task=task.name)
    except Exception as exc:
        _signal_logger.debug("metrics signal failure (complete): %s", exc)
    finally:
        _clear_active_media_marker()
        _malloc_trim()


@huey.signal(SIGNAL_ERROR)
def _on_error(signal, task, exc=None):
    try:
        _task_starts.pop(task.id, None)
        metrics.count("worker.task.error", task=task.name)
    except Exception as inner:
        _signal_logger.debug("metrics signal failure (error): %s", inner)
    finally:
        _clear_active_media_marker()
        _malloc_trim()


@huey.signal(SIGNAL_RETRYING)
def _on_retrying(signal, task, exc=None):
    try:
        metrics.count("worker.task.retry", task=task.name)
    except Exception as inner:
        _signal_logger.debug("metrics signal failure (retry): %s", inner)
    finally:
        _clear_active_media_marker()


@huey.signal(SIGNAL_INTERRUPTED)
def _on_interrupted(signal, task, *args, **kwargs):
    """Put an abruptly interrupted task back on the durable queue.

    Huey removes a task from SQLite before executing it.  ``SIGTERM`` is its
    immediate-shutdown path and emits ``SIGNAL_INTERRUPTED`` for work that was
    still in flight, so the task supplied by that signal must be explicitly
    re-enqueued.  Normal completion emits ``SIGNAL_COMPLETE`` instead and does
    not pass through this handler.
    """
    try:
        _task_starts.pop(task.id, None)
        try:
            huey.enqueue(task)
        except Exception:
            _signal_logger.exception(
                "Failed to re-enqueue interrupted task %s (%s)",
                task.id,
                task.name,
            )
            raise

        _signal_logger.warning(
            "Re-enqueued interrupted task %s (%s)", task.id, task.name
        )
        try:
            metrics.count("worker.task.interrupted", task=task.name)
            metrics.count("worker.task.requeued", task=task.name)
        except Exception as inner:
            _signal_logger.debug("metrics signal failure (interrupted): %s", inner)
    finally:
        _clear_active_media_marker()


# Import task modules so their @huey.task decorators fire and register
# every task with the Huey registry before the consumer starts polling.
from workers import tasks as _tasks  # noqa: E402, F401
