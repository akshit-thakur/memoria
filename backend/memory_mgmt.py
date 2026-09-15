"""Adaptive memory controller for the GPU worker pipeline.

In plain English: the photo-processing pipeline runs ML models that can
each chew through gigabytes of GPU and system RAM. This module is the
referee. It watches how much memory is in use, classifies it into one of
three tiers (`ok` / `warn` / `critical`), and decides how many photos to
process in the next batch — or whether to pause entirely so the system can
breathe.

Two public touch-points:
  • ``monitor.adaptive_params()`` returns ``{batch_size, sleep}`` — the
    pipeline asks this between steps.
  • ``vram_breathe()`` is the convenience wrapper that actually performs
    ``gc.collect()`` + ``torch.cuda.empty_cache()`` + ``time.sleep`` if
    the current pressure tier says so.

``MemoryMonitor`` reads system RSS and GPU allocation, classifies pressure
as ok / warn / critical, and derives an adaptive batch size. Callers invoke
``vram_breathe()`` between pipeline steps to pause & clear caches when
pressure climbs — the formulas live in one place here so main.py and every
script that uses them stay in sync.

Zero internal imports beyond stdlib + optional psutil/torch, so the module
is safe to import from anywhere without pulling the Huey pipeline graph.
"""

from __future__ import annotations

import gc
import logging
import math
import os
import re
import subprocess
import time
from typing import Literal


def _envf(key: str, default: float) -> float:
    """Read a finite env float, falling back on malformed values."""
    try:
        value = float(os.environ.get(key, str(default)))
    except (ValueError, TypeError):
        return default
    return value if math.isfinite(value) else default


def _envi(
    key: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Read an integer env value and enforce optional safety bounds."""
    try:
        value = int(os.environ.get(key, str(default)))
    except (ValueError, TypeError):
        return default
    if minimum is not None and value < minimum:
        return default
    if maximum is not None and value > maximum:
        return default
    return value


def _env_ratio(key: str, default: float) -> float:
    """Read a finite fractional threshold strictly between zero and one."""
    value = _envf(key, default)
    return value if 0.0 < value < 1.0 else default


def _env_threshold_pair(
    warn_key: str,
    crit_key: str,
    *,
    warn_default: float,
    crit_default: float,
) -> tuple[float, float]:
    """Return ordered warning/critical thresholds or their safe defaults."""
    warn = _env_ratio(warn_key, warn_default)
    critical = _env_ratio(crit_key, crit_default)
    if warn >= critical:
        return warn_default, crit_default
    return warn, critical

# psutil is optional — without it we fall back to parsing macOS vm_stat
# so the kill-switch never silently returns 0.0.
try:
    import psutil  # type: ignore[import-not-found]
    _HAVE_PSUTIL = True
except Exception:
    psutil = None  # type: ignore[assignment]
    _HAVE_PSUTIL = False

logger = logging.getLogger("memoria.ml.memory")


# ── Adaptive Memory Controller (Spec 01) ─────────────────────────────────────

class MemoryMonitor:
    """Reads RSS and CUDA/MPS GPU memory. Computes adaptive batch parameters.

    Batch size formula (continuous, not discrete tiers):
        available_mb = (total_ram_mb * (1 - rss_percent)) - RESERVED_MB
        batch_size   = max(1, min(MEMORIA_BATCH_SIZE,
                                  available_mb // MB_PER_ITEM))

    Guardrails override the formula:
        pressure == 'critical'  → batch_size = 0  (pause + cache clear)
        rss >= RAM_FLOOR_PCT    → batch_size = 1  (minimum forward progress)

    Defaults are conservative and every threshold is environment-driven, so
    deployments can scale from CPU-only systems to CUDA accelerators without
    changing application code:

        MEMORIA_RAM_WARN_PCT        (default 0.85)
        MEMORIA_RAM_CRIT_PCT        (default 0.95)
        MEMORIA_RAM_FLOOR_PCT       (default 0.90)
        MEMORIA_GPU_WARN_PCT        (default 0.85)
        MEMORIA_GPU_CRIT_PCT        (default 0.95)
        MEMORIA_BATCH_SIZE          (default 16; maximum 50)
        MEMORIA_MB_PER_ITEM         (default 512)
        MEMORIA_RESERVED_MB         (default 2048)
        MEMORIA_BREATHE_SLEEP_OK_S  (default 0)
        MEMORIA_BREATHE_SLEEP_WARN_S (default 1)
        MEMORIA_BREATHE_SLEEP_CRIT_S (default 5)
    """

    # System-RAM thresholds (fraction 0–1).
    RAM_WARN_PCT, RAM_CRIT_PCT = _env_threshold_pair(
        "MEMORIA_RAM_WARN_PCT",
        "MEMORIA_RAM_CRIT_PCT",
        warn_default=0.85,
        crit_default=0.95,
    )
    _RAM_FLOOR_PCT: float = _env_ratio("MEMORIA_RAM_FLOOR_PCT", 0.90)

    # GPU-VRAM thresholds (fraction of dedicated VRAM). Only meaningful on
    # CUDA — MPS shares with system RAM, so its pressure is already covered
    # by RSS percentages.
    GPU_WARN_PCT, GPU_CRIT_PCT = _env_threshold_pair(
        "MEMORIA_GPU_WARN_PCT",
        "MEMORIA_GPU_CRIT_PCT",
        warn_default=0.85,
        crit_default=0.95,
    )

    # User-selected upper bound for adaptive dispatch. The default matches the
    # watcher and remains bounded by the pipeline's established safety cap.
    BATCH_SIZE: int = _envi(
        "MEMORIA_BATCH_SIZE", 16, minimum=1, maximum=50
    )

    # Estimated RAM consumed per ML pipeline item (CLIP embed + decode buffer).
    # Bumped from 256 → 512 to account for occasional 50 MP RAW decodes.
    _MB_PER_ITEM: int = _envi("MEMORIA_MB_PER_ITEM", 512, minimum=1)

    # Reserved headroom (Postgres + browser bitmaps + OS pager).
    _RESERVED_MB: int = _envi("MEMORIA_RESERVED_MB", 2048, minimum=0)

    # Sleep durations (seconds) per pressure tier. 0 = no sleep, just
    # gc.collect() + GPU cache clear.
    _SLEEP_OK_S: int = _envi("MEMORIA_BREATHE_SLEEP_OK_S", 0, minimum=0)
    _SLEEP_WARN_S: int = _envi("MEMORIA_BREATHE_SLEEP_WARN_S", 1, minimum=0)
    _SLEEP_CRIT_S: int = _envi("MEMORIA_BREATHE_SLEEP_CRIT_S", 5, minimum=0)

    def rss_percent(self) -> float:
        """Return system RAM usage as a fraction (0.0–1.0).

        Tries psutil first; falls back to macOS ``vm_stat`` parsing.
        Raises RuntimeError if both paths fail.
        """
        if _HAVE_PSUTIL:
            try:
                return psutil.virtual_memory().percent / 100.0
            except Exception:
                pass

        return self._vm_stat_rss_percent()

    def total_ram_mb(self) -> int:
        """Return total physical RAM in MB.

        Tries psutil first; falls back to ``sysctl hw.memsize`` where
        available. Returns a conservative 8 GB estimate if both fail.
        """
        if _HAVE_PSUTIL:
            try:
                return psutil.virtual_memory().total // (1024 * 1024)
            except Exception:
                pass
        try:
            raw = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True, timeout=5
            )
            return int(raw.strip()) // (1024 * 1024)
        except Exception:
            return 8192  # conservative generic fallback when probing is unavailable

    @staticmethod
    def _vm_stat_rss_percent() -> float:
        """Parse ``vm_stat`` on macOS to compute used/total RAM fraction."""
        try:
            raw = subprocess.check_output(["vm_stat"], text=True, timeout=5)
        except Exception as exc:
            raise RuntimeError(
                "Cannot determine memory usage: psutil unavailable and vm_stat failed. "
                "Install psutil (`pip install psutil`) or run on macOS."
            ) from exc

        page_size = 16384  # default Apple Silicon
        try:
            ps_raw = subprocess.check_output(
                ["sysctl", "-n", "hw.pagesize"], text=True, timeout=5
            )
            page_size = int(ps_raw.strip())
        except Exception:
            pass

        pages = {}
        for line in raw.splitlines():
            m = re.match(r"^(.+?):\s+(\d+)\.", line)
            if m:
                pages[m.group(1).strip()] = int(m.group(2))

        free = pages.get("Pages free", 0)
        active = pages.get("Pages active", 0)
        inactive = pages.get("Pages inactive", 0)
        speculative = pages.get("Pages speculative", 0)
        wired = pages.get("Pages wired down", 0)
        compressed = pages.get("Pages occupied by compressor", 0)

        total_pages = free + active + inactive + speculative + wired + compressed
        if total_pages == 0:
            raise RuntimeError("vm_stat returned no usable page counts.")

        used_pages = active + wired + compressed
        return used_pages / total_pages

    def gpu_bytes_used(self) -> int:
        """Return GPU memory currently allocated (bytes).

        Checks CUDA first, MPS second (Apple Silicon).
        Returns 0 if neither backend is available.
        """
        try:
            import torch
            if torch.cuda.is_available():
                return int(torch.cuda.memory_allocated())
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return int(torch.mps.current_allocated_memory())
        except (ImportError, RuntimeError):
            pass
        return 0

    def gpu_pressure(self) -> float:
        """Return GPU VRAM allocation as a fraction of total (0.0–1.0).

        On CUDA: ``memory_allocated / total_memory`` for the active device.
        On MPS: returns 0.0 because MPS shares the system pool — its
        pressure is already captured by ``rss_percent``.
        Returns 0.0 if torch isn't importable or the device probe fails.
        """
        try:
            import torch
            if torch.cuda.is_available():
                idx = torch.cuda.current_device()
                total = torch.cuda.get_device_properties(idx).total_memory
                if total <= 0:
                    return 0.0
                return float(torch.cuda.memory_allocated(idx)) / float(total)
        except (ImportError, RuntimeError, AssertionError):
            pass
        return 0.0

    # How long (seconds) to reuse a cached pressure() result. Each call
    # hits psutil (a syscall) + optionally torch.cuda — caching at 100ms
    # prevents redundant syscalls when pressure() is called multiple times
    # per file (detect_blur/faces/scene/dedup each gate on it).
    _PRESSURE_CACHE_TTL: float = 0.1
    _pressure_cache_result: Literal["ok", "warn", "critical"] = "ok"
    _pressure_cache_ts: float = 0.0

    def pressure(self) -> Literal["ok", "warn", "critical"]:
        """Compute the dominant memory pressure level (max of RSS and GPU).

        Either pool reaching its critical threshold triggers a 'critical'
        return — separate pools, separate failure modes. On MPS,
        ``gpu_pressure`` returns 0 so this collapses to RSS-only.

        Result is cached for _PRESSURE_CACHE_TTL seconds to avoid redundant
        psutil/torch syscalls when multiple pipeline stages call this in a
        tight loop.
        """
        now = time.monotonic()
        if now - self._pressure_cache_ts < self._PRESSURE_CACHE_TTL:
            return self._pressure_cache_result

        try:
            r = self.rss_percent()
        except RuntimeError:
            logger.warning("Memory detection failed — assuming 'warn' pressure.")
            result: Literal["ok", "warn", "critical"] = "warn"
            self._pressure_cache_result = result
            self._pressure_cache_ts = now
            return result

        g = self.gpu_pressure()

        if r >= self.RAM_CRIT_PCT or g >= self.GPU_CRIT_PCT:
            result = "critical"
        elif r >= self.RAM_WARN_PCT or g >= self.GPU_WARN_PCT:
            result = "warn"
        else:
            result = "ok"

        self._pressure_cache_result = result
        self._pressure_cache_ts = now
        return result

    def adaptive_batch_size(self) -> int:
        """Return batch_size using the continuous formula.

        In plain English: how much *free* RAM do we have? Subtract a fixed
        headroom reserve (2 GB by default — keeps Postgres, the browser,
        and ML inference comfortable), then divide by the per-item RAM cost.
        Clamp the result to [1, MEMORIA_BATCH_SIZE] so we never exceed the
        operator-selected batch cap (16 by default, hard-capped at 50).

        Formula: available_mb = (total_ram_mb * free_fraction) - RESERVED_MB
                 batch_size   = max(
                     1, min(MEMORIA_BATCH_SIZE, available_mb // MB_PER_ITEM)
                 )

        Hard guardrails override the formula at high pressure:
            >= RAM_CRIT_PCT (95% by default)  → 0 (pause)
            >= RAM_FLOOR_PCT (90% by default) → 1 (minimum progress)
        """
        try:
            pct = self.rss_percent()
        except RuntimeError:
            logger.warning("Memory detection failed — using batch_size=1.")
            return 1

        if pct >= self.RAM_CRIT_PCT:
            return 0
        if pct >= self._RAM_FLOOR_PCT:
            return 1

        total_mb = self.total_ram_mb()
        free_mb = total_mb * (1.0 - pct)
        available_mb = free_mb - self._RESERVED_MB
        return max(
            1,
            min(self.BATCH_SIZE, int(available_mb // self._MB_PER_ITEM)),
        )

    def adaptive_params(self) -> dict:
        """Return ``{"batch_size": int, "sleep": int}`` for current pressure.

        Sleep duration follows the pressure tier returned by ``pressure()``
        (which already considers both RSS and GPU). Tier-to-sleep mapping
        is configurable via the ``MEMORIA_BREATHE_SLEEP_*_S`` env vars.
        """
        tier = self.pressure()
        if tier == "critical":
            sleep = self._SLEEP_CRIT_S
        elif tier == "warn":
            sleep = self._SLEEP_WARN_S
        else:
            sleep = self._SLEEP_OK_S

        return {"batch_size": self.adaptive_batch_size(), "sleep": sleep}


# Module-level singleton, shared by every caller.
monitor = MemoryMonitor()

# Tracks how many tasks have run since the last breathing pause. Reset
# every time vram_breathe() triggers.
_batch_counter = 0


def gpu_cache_clear() -> None:
    """Release GPU memory (CUDA or MPS depending on hardware)."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass


def vram_breathe() -> None:
    """Adaptive VRAM breathing: batch size and sleep scale with memory pressure.

    Critical pressure triggers an immediate pause + cache clear regardless
    of batch boundary; warn/ok pressure breathes at the boundary.
    """
    global _batch_counter
    _batch_counter += 1

    params = monitor.adaptive_params()
    batch_size = params["batch_size"]
    sleep_secs = params["sleep"]

    pressure = monitor.pressure()
    if pressure == "critical":
        logger.warning(
            "VRAM breathing [CRITICAL]: pausing dispatch, clearing caches, sleeping %ds …",
            sleep_secs,
        )
        gpu_cache_clear()
        gc.collect()
        if sleep_secs > 0:
            time.sleep(sleep_secs)
        _batch_counter = 0
        return

    if batch_size > 0 and _batch_counter >= batch_size:
        _batch_counter = 0
        logger.info(
            "VRAM breathing [%s]: clearing caches and sleeping %ds (batch=%d) …",
            pressure.upper(), sleep_secs, batch_size,
        )
        gpu_cache_clear()
        gc.collect()
        if sleep_secs > 0:
            time.sleep(sleep_secs)
