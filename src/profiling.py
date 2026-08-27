"""Small profiling helpers shared by the EC-SLAM worker threads."""

import csv
import threading
from contextlib import contextmanager
from pathlib import Path

import torch


_NVTX_ENABLED = torch.cuda.is_available()


@contextmanager
def nvtx_range(name: str):
    """Create a stack-nested NVTX range, or a no-op on CPU-only systems.

    EC-SLAM executes mapping and tracking in separate worker threads. Each
    hierarchy is therefore pushed and popped on its worker's own NVTX stack,
    which gives Nsight the parent/child structure from Figure 1. ``finally``
    keeps the stack balanced if a profiled block raises.
    """
    if not _NVTX_ENABLED:
        yield
        return

    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


class IterationBreakdownCudaTimer:
    """Collect per-iteration CUDA timings from EC-SLAM's worker threads.

    Tracking and mapping execute concurrently, so callers supply the complete
    row identity rather than mutating a shared "current frame" context. CUDA
    events are resolved only after both workers finish to avoid synchronizing
    every optimization iteration.
    """

    OUTPUT_COLUMNS = (
        "frame_id",
        "iteration_type",
        "iteration_id",
        "forward_cuda_ms",
        "backward_cuda_ms",
        "full_iteration_cuda_ms",
    )
    ITERATION_TYPES = {"tracking", "bundle_adjustment"}
    METRICS = {
        "forward_cuda_ms",
        "backward_cuda_ms",
        "full_iteration_cuda_ms",
    }

    def __init__(
        self,
        enabled=False,
        output_csv="./profiling/iteration_breakdown.csv",
        warmup_frames=0,
        device=None,
    ):
        self.enabled = bool(enabled)
        self.output_csv = Path(output_csv)
        self.warmup_frames = int(warmup_frames)
        self.device = torch.device(device) if device is not None else None
        self._pending = []
        self._lock = threading.Lock()

    @contextmanager
    def stage(
        self,
        frame_id: int,
        iteration_type: str,
        iteration_id: int,
        metric_name: str,
    ):
        """Time one CUDA component without forcing an immediate sync."""
        if (
            not self.enabled
            or not torch.cuda.is_available()
            or (self.device is not None and self.device.type != "cuda")
            or int(frame_id) < self.warmup_frames
        ):
            yield
            return
        if iteration_type not in self.ITERATION_TYPES:
            raise ValueError(f"Unknown iteration type: {iteration_type}")
        if metric_name not in self.METRICS:
            raise ValueError(f"Unknown timing metric: {metric_name}")

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        stream = (
            torch.cuda.current_stream(self.device)
            if self.device is not None
            else None
        )
        start_event.record(stream)
        try:
            yield
        finally:
            stream = (
                torch.cuda.current_stream(self.device)
                if self.device is not None
                else None
            )
            end_event.record(stream)
            record = (
                int(frame_id),
                iteration_type,
                int(iteration_id),
                metric_name,
                start_event,
                end_event,
            )
            with self._lock:
                self._pending.append(record)

    def flush(self):
        """Synchronize pending events and write one complete CSV row each."""
        if not self.enabled:
            return

        with self._lock:
            pending = self._pending
            self._pending = []

        if pending:
            torch.cuda.synchronize(self.device)

        rows = {}
        for (
            frame_id,
            iteration_type,
            iteration_id,
            metric_name,
            start_event,
            end_event,
        ) in pending:
            row_key = (frame_id, iteration_type, iteration_id)
            row = rows.setdefault(row_key, {})
            if metric_name in row:
                raise RuntimeError(
                    "Duplicate CUDA timing metric "
                    f"'{metric_name}' for {row_key}."
                )
            row[metric_name] = float(start_event.elapsed_time(end_event))

        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_csv, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(self.OUTPUT_COLUMNS)
            for row_key in sorted(rows):
                row = rows[row_key]
                missing = self.METRICS.difference(row)
                if missing:
                    raise RuntimeError(
                        f"Incomplete CUDA timing row {row_key}; missing "
                        + ", ".join(sorted(missing))
                    )
                writer.writerow([
                    *row_key,
                    f"{row['forward_cuda_ms']:.6f}",
                    f"{row['backward_cuda_ms']:.6f}",
                    f"{row['full_iteration_cuda_ms']:.6f}",
                ])
