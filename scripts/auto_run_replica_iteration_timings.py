#!/usr/bin/env python3
"""Collect focused per-iteration CUDA timings for EC-SLAM on Replica.

Each timing CSV contains one row per tracking or bundle-adjustment iteration
with its forward, backward, and full-iteration CUDA-event durations. The
default sweep compares setup A (CoherentPrime, unsorted) with setup D (Morton,
Morton-sorted) at a log2 hash-table size of 19 on all eight Replica scenes.
"""

import argparse
import copy
import csv
import math
import subprocess
import sys
import time
import uuid
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


SETUP_DEFINITIONS = {
    "A": {"hash": "CoherentPrime", "morton_sort": False},
    "D": {"hash": "Morton", "morton_sort": True},
}

DEFAULT_REPLICA_SCENES = [
    "office0",
    "office1",
    "office2",
    "office3",
    "office4",
    "room0",
    "room1",
    "room2",
]

DEFAULT_TIMING_DIR = Path("results/iteration_breakdown_MAXN/Replica")
DEFAULT_LOG_FILE = Path("output/Replica_iteration_timing_runs_MAXN.csv")
LOG_COLUMNS = (
    "Run",
    "Scene",
    "Setup",
    "Hash",
    "MortonSort",
    "HashSizeLog2",
    "FrameLimit",
    "WarmupFrames",
    "TimingCsv",
    "RunWallTimeSeconds",
)
TIMING_COLUMNS = (
    "frame_id",
    "iteration_type",
    "iteration_id",
    "forward_cuda_ms",
    "backward_cuda_ms",
    "full_iteration_cuda_ms",
)


def read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def write_yaml(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def merge_config(base: dict, overrides: dict) -> dict:
    """Recursively merge config dictionaries like ``src.utils.load_config``."""
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_merged_config(
    path: Path,
    default_path: Path | None = Path("configs/EC_SLAM.yaml"),
) -> dict:
    """Load EC-SLAM's inherited YAML without importing its CUDA modules."""
    special = read_yaml(path)
    inherit_from = special.get("inherit_from")
    if inherit_from is not None:
        base = load_merged_config(Path(inherit_from), default_path)
    elif default_path is not None:
        base = read_yaml(default_path)
    else:
        base = {}
    return merge_config(base, special)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_size_arg(value: str) -> int | None:
    value_text = str(value).lower()
    if value_text in {"null", "none", "default"}:
        return None
    size = int(value)
    if size <= 0:
        raise argparse.ArgumentTypeError(
            "--sizes values must be positive integers or 'default'."
        )
    return size


def parse_frame_count_arg(value: str) -> int | None:
    value_text = str(value).lower()
    if value_text in {"all", "null", "none", "default"}:
        return None
    frame_count = int(value)
    if frame_count <= 0:
        raise argparse.ArgumentTypeError(
            "--frame-count must be a positive integer or 'all'."
        )
    return frame_count


def normalize_scene_arg(value: str) -> str:
    scene_name = Path(value).name
    for suffix in (".yaml", ".yml"):
        if scene_name.lower().endswith(suffix):
            return scene_name[: -len(suffix)]
    return scene_name


def size_label(size: int | None) -> str:
    return "DefaultSize" if size is None else f"Size{size}"


def build_exp_name(setup: str, size: int | None, run_idx: int) -> str:
    return f"iteration_breakdownMAXN_{setup}_{size_label(size)}_Run{run_idx}"


def build_timing_csv_path(
    timing_dir: Path,
    scene: str,
    setup: str,
    size: int | None,
    run_idx: int,
) -> Path:
    return timing_dir / (
        f"{scene}_iteration_breakdownMAXN_{setup}_{size_label(size)}"
        f"_Run{run_idx}.csv"
    )


def build_staging_timing_path(timing_csv: Path) -> Path:
    return timing_csv.with_name(
        f".{timing_csv.name}.{uuid.uuid4().hex}.partial"
    )


def format_config_path(path: Path) -> str:
    path_text = str(path)
    if not path.is_absolute() and not path_text.startswith("."):
        return f"./{path_text}"
    return path_text


def override_config(
    base_cfg_path: Path,
    scene: str,
    setup_key: str,
    table_size: int | None,
    run_idx: int,
    frame_count: int | None,
    warmup_frames: int,
    timing_csv: Path,
    disable_eval: bool,
) -> tuple[Path, Path]:
    """Write one EC-SLAM leaf config beside its inherited scene config."""
    cfg = read_yaml(base_cfg_path)
    merged_base_cfg = load_merged_config(base_cfg_path)
    setup = SETUP_DEFINITIONS[setup_key]

    # EC-SLAM reads this section as ``HashGrid`` (Co-SLAM uses ``grid``).
    cfg.setdefault("HashGrid", {})
    cfg["HashGrid"]["hash"] = setup["hash"]
    cfg["HashGrid"]["morton_sort"] = setup["morton_sort"]
    if table_size is not None:
        cfg["HashGrid"]["hash_size"] = table_size

    cfg.setdefault("timing", {})
    cfg["timing"]["mode"] = "iteration_breakdown"
    cfg["timing"]["warmup_frames"] = warmup_frames
    cfg["timing"]["max_frames"] = frame_count
    cfg["timing"]["disable_eval"] = bool(disable_eval)
    cfg["timing"]["iteration_output_csv"] = format_config_path(timing_csv)

    # EC-SLAM does not append data.exp_name to data.output, so the output path
    # itself must identify every setup, size, and repetition.
    exp_name = build_exp_name(setup_key, table_size, run_idx)
    cfg.setdefault("data", {})
    cfg["data"]["exp_name"] = exp_name
    base_output = Path(
        cfg["data"].get(
            "output",
            merged_base_cfg.get("data", {}).get(
                "output",
                f"output/Replica/{scene}",
            ),
        )
    )
    experiment_output = base_output / "iteration_breakdown" / exp_name
    cfg["data"]["output"] = str(experiment_output)

    # Keep the file beside the scene config: its inherit_from entry is
    # intentionally resolved with the same repository-root working directory.
    temp_name = f"temp_{scene}_{exp_name}_{uuid.uuid4().hex[:8]}.yaml"
    temp_path = base_cfg_path.parent / temp_name
    write_yaml(temp_path, cfg)
    return temp_path, experiment_output


def discover_scenes(
    configs_root: Path,
    requested_scenes: list[str] | None,
) -> list[str]:
    if requested_scenes:
        scenes = [normalize_scene_arg(scene) for scene in requested_scenes]
    else:
        scenes = list(DEFAULT_REPLICA_SCENES)

    scenes = list(dict.fromkeys(scenes))
    missing = [
        str(configs_root / f"{scene}.yaml")
        for scene in scenes
        if not (configs_root / f"{scene}.yaml").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing Replica scene config(s): " + ", ".join(missing)
        )
    return scenes


def prepare_log_file(log_file: Path) -> None:
    ensure_dir(log_file.parent)
    if not log_file.exists():
        with open(log_file, "w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(LOG_COLUMNS)
        return

    with open(log_file, "r", encoding="utf-8", newline="") as handle:
        current_header = next(csv.reader(handle), [])
    if current_header != list(LOG_COLUMNS):
        raise RuntimeError(
            f"Existing log file '{log_file}' has an incompatible header. "
            "Use --log-file with a fresh CSV path."
        )


def build_expected_iteration_counts(config: dict) -> dict[tuple[int, str], int]:
    """Derive the exact rows EC-SLAM should emit for one timing run."""
    if config["tracking"].get("gt_camera", False):
        raise ValueError("timing runs require tracking.gt_camera=false")

    tracking_iters = int(config["tracking"]["iters"])
    mapping_iters = int(config["mapping"]["iters"])
    every_frame = int(config["mapping"]["every_frame"])
    if tracking_iters <= 0 or mapping_iters <= 0 or every_frame <= 0:
        raise ValueError(
            "tracking.iters, mapping.iters, and mapping.every_frame must be "
            "positive for iteration timing"
        )

    depth_dir = Path(config["data"]["input_folder"]) / "depth"
    if not depth_dir.is_dir():
        raise FileNotFoundError(f"Replica depth directory not found: {depth_dir}")
    available_frames = sum(1 for _ in depth_dir.iterdir())
    max_frames = config.get("timing", {}).get("max_frames")
    frame_count = (
        available_frames
        if max_frames is None
        else min(available_frames, int(max_frames))
    )
    if frame_count < 2:
        raise ValueError("the effective dataset must contain at least 2 frames")

    warmup_frames = int(config.get("timing", {}).get("warmup_frames", 0))
    expected = {
        (frame_id, "tracking"): tracking_iters
        for frame_id in range(1, frame_count)
        if frame_id >= warmup_frames
    }

    mapping_frames = list(range(0, frame_count, every_frame))
    if frame_count % every_frame != 1:
        mapping_frames.append(frame_count - 1)
    for frame_id in mapping_frames[1:]:
        if frame_id >= warmup_frames:
            expected[(frame_id, "bundle_adjustment")] = mapping_iters

    present_types = {iteration_type for _, iteration_type in expected}
    missing_types = {"tracking", "bundle_adjustment"}.difference(present_types)
    if missing_types:
        raise ValueError(
            "the frame limit and warmup omit all "
            + ", ".join(sorted(missing_types))
            + " iterations"
        )
    return expected


def validate_timing_csv(
    timing_csv: Path,
    expected_counts: dict[tuple[int, str], int] | None = None,
) -> dict[str, int]:
    if not timing_csv.is_file():
        raise RuntimeError(f"timing CSV was not created: {timing_csv}")

    counts = {"tracking": 0, "bundle_adjustment": 0}
    expected_iteration_ids = {}
    with open(timing_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != TIMING_COLUMNS:
            raise RuntimeError(
                f"timing CSV has an unexpected header: {timing_csv}"
            )

        for line_number, row in enumerate(reader, start=2):
            if None in row or any(
                row.get(column) in (None, "") for column in TIMING_COLUMNS
            ):
                raise RuntimeError(
                    f"timing CSV has an incomplete row at line "
                    f"{line_number}: {timing_csv}"
                )

            try:
                frame_id = int(row["frame_id"])
                iteration_id = int(row["iteration_id"])
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    f"timing CSV has a non-integer frame/iteration ID at "
                    f"line {line_number}: {timing_csv}"
                ) from error

            iteration_type = row["iteration_type"]
            if iteration_type not in counts:
                raise RuntimeError(
                    f"timing CSV has an unknown iteration type at line "
                    f"{line_number}: {timing_csv}"
                )
            if frame_id < 0 or iteration_id < 0:
                raise RuntimeError(
                    f"timing CSV has a negative frame/iteration ID at line "
                    f"{line_number}: {timing_csv}"
                )

            sequence_key = (frame_id, iteration_type)
            expected_id = expected_iteration_ids.get(sequence_key, 0)
            if iteration_id != expected_id:
                raise RuntimeError(
                    f"timing CSV iteration IDs are not contiguous at line "
                    f"{line_number}: expected {expected_id}, got "
                    f"{iteration_id} in {timing_csv}"
                )
            expected_iteration_ids[sequence_key] = expected_id + 1

            try:
                forward_ms = float(row["forward_cuda_ms"])
                backward_ms = float(row["backward_cuda_ms"])
                full_ms = float(row["full_iteration_cuda_ms"])
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    f"timing CSV has a non-numeric duration at line "
                    f"{line_number}: {timing_csv}"
                ) from error

            timings = (forward_ms, backward_ms, full_ms)
            if any(not math.isfinite(value) or value < 0 for value in timings):
                raise RuntimeError(
                    f"timing CSV has an invalid duration at line "
                    f"{line_number}: {timing_csv}"
                )
            if full_ms < forward_ms or full_ms < backward_ms:
                raise RuntimeError(
                    f"full iteration is shorter than a component at line "
                    f"{line_number}: {timing_csv}"
                )

            counts[iteration_type] += 1

    missing_types = [
        iteration_type
        for iteration_type, count in counts.items()
        if count == 0
    ]
    if missing_types:
        raise RuntimeError(
            "timing CSV is missing "
            + ", ".join(missing_types)
            + " rows; the frame limit and warmup must include at least one "
            + f"tracking and one BA invocation: {timing_csv}"
        )
    if expected_counts is not None:
        observed_counts = dict(expected_iteration_ids)
        missing_groups = sorted(set(expected_counts).difference(observed_counts))
        unexpected_groups = sorted(set(observed_counts).difference(expected_counts))
        wrong_counts = sorted(
            (key, expected_counts[key], observed_counts[key])
            for key in set(expected_counts).intersection(observed_counts)
            if expected_counts[key] != observed_counts[key]
        )
        if missing_groups or unexpected_groups or wrong_counts:
            details = []
            if missing_groups:
                details.append(f"missing groups={missing_groups[:5]}")
            if unexpected_groups:
                details.append(f"unexpected groups={unexpected_groups[:5]}")
            if wrong_counts:
                details.append(
                    "wrong iteration counts="
                    + str(wrong_counts[:5])
                )
            raise RuntimeError(
                f"timing CSV does not match the scheduled run: {timing_csv}; "
                + "; ".join(details)
            )
    return counts


def append_log_row(
    log_file: Path,
    run_idx: int,
    scene: str,
    setup_key: str,
    size: int | None,
    frame_count: int | None,
    warmup_frames: int,
    timing_csv: Path,
    duration: float,
) -> None:
    setup = SETUP_DEFINITIONS[setup_key]
    with open(log_file, "a", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow([
            run_idx,
            scene,
            setup_key,
            setup["hash"],
            setup["morton_sort"],
            "Default" if size is None else size,
            "All" if frame_count is None else frame_count,
            warmup_frames,
            timing_csv,
            f"{duration:.4f}",
        ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run EC-SLAM on Replica and collect per-iteration forward, "
            "backward, and full-iteration CUDA timings for setups A and D."
        )
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of repetitions of every scene/setup/size combination.",
    )
    parser.add_argument(
        "--start-run",
        type=int,
        default=1,
        help="First run index to use, useful when resuming a sweep.",
    )
    parser.add_argument(
        "--setups",
        nargs="+",
        default=["A", "D"],
        choices=tuple(SETUP_DEFINITIONS),
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=[19],
        type=parse_size_arg,
        metavar="N|default",
        help="Hash-table log2 sizes. Defaults to 19.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Optional Replica scene subset; defaults to all eight scenes.",
    )
    parser.add_argument(
        "--configs-root",
        default=Path("configs/Replica"),
        type=Path,
    )
    parser.add_argument(
        "--frame-count",
        type=parse_frame_count_arg,
        default=None,
        metavar="N|all",
        help=(
            "Maximum frames per scene; defaults to all available frames. "
            "Use at least 6 with the standard Replica config to include BA."
        ),
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=0,
        help="Discard timings whose frame ID is below this value.",
    )
    parser.add_argument(
        "--timing-dir",
        type=Path,
        default=DEFAULT_TIMING_DIR,
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_FILE,
    )
    parser.add_argument("--python-exec", default=sys.executable)
    parser.add_argument(
        "--target-script",
        type=Path,
        default=Path("run.py"),
    )
    parser.add_argument(
        "--enable-eval",
        action="store_true",
        help="Enable EC-SLAM trajectory evaluation during timing runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write and report configs without launching EC-SLAM.",
    )
    parser.add_argument(
        "--keep-temp-configs",
        action="store_true",
        help="Keep generated YAML files after dry or successful runs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow replacing an existing timing CSV and reusing its EC-SLAM "
            "experiment output directory."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if yaml is None:
        print(
            "ERROR: PyYAML is required. Install it with: pip install pyyaml",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.runs <= 0:
        print("ERROR: --runs must be positive.", file=sys.stderr)
        sys.exit(2)
    if args.start_run <= 0:
        print("ERROR: --start-run must be positive.", file=sys.stderr)
        sys.exit(2)
    if args.warmup_frames < 0:
        print(
            "ERROR: --warmup-frames must be non-negative.",
            file=sys.stderr,
        )
        sys.exit(2)
    if args.frame_count is not None and args.frame_count < 6:
        print(
            "ERROR: --frame-count must be at least 6 for the standard "
            "Replica config to include a real bundle-adjustment pass.",
            file=sys.stderr,
        )
        sys.exit(2)
    if (
        args.frame_count is not None
        and args.warmup_frames >= args.frame_count
    ):
        print(
            "ERROR: --warmup-frames must be smaller than --frame-count.",
            file=sys.stderr,
        )
        sys.exit(2)
    if not args.target_script.is_file():
        print(
            f"ERROR: target script not found: {args.target_script}",
            file=sys.stderr,
        )
        sys.exit(2)
    if not args.configs_root.is_dir():
        print(
            f"ERROR: configs root not found: {args.configs_root}",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        scenes = discover_scenes(args.configs_root, args.scenes)
    except FileNotFoundError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)

    setups = list(dict.fromkeys(args.setups))
    sizes = list(dict.fromkeys(args.sizes))
    frame_count_label = (
        "All" if args.frame_count is None else str(args.frame_count)
    )
    total_runs = len(scenes) * len(setups) * len(sizes) * args.runs

    print(f"Total runs scheduled: {total_runs}")
    print(f"Scenes: {', '.join(scenes)}")
    print(f"Setups: {', '.join(setups)}")
    print(
        "Hash sizes: "
        + ", ".join(
            "Default" if size is None else str(size) for size in sizes
        )
    )
    print(f"Frame count limit: {frame_count_label}")
    print(f"Warmup frames: {args.warmup_frames}")
    print("-" * 60)

    if not args.dry_run:
        try:
            prepare_log_file(args.log_file)
        except RuntimeError as error:
            print(f"ERROR: {error}", file=sys.stderr)
            sys.exit(2)

    success_count = 0
    failure_count = 0
    for repetition_idx, run_idx in enumerate(
        range(args.start_run, args.start_run + args.runs),
        start=1,
    ):
        print(
            f"\n=== RUN ITERATION {repetition_idx}/{args.runs} "
            f"(index {run_idx}) ==="
        )
        combinations = []
        for scene_index, scene in enumerate(scenes):
            # Keep A/D adjacent and alternate which one goes first to reduce
            # systematic ordering and thermal bias.
            if (scene_index + run_idx - 1) % 2 == 0:
                ordered_setups = setups
            else:
                ordered_setups = list(reversed(setups))
            for size in sizes:
                combinations.extend(
                    (scene, setup_key, size)
                    for setup_key in ordered_setups
                )

        for scene, setup_key, size in combinations:
            base_cfg = args.configs_root / f"{scene}.yaml"
            timing_csv = build_timing_csv_path(
                args.timing_dir,
                scene,
                setup_key,
                size,
                run_idx,
            )
            ensure_dir(timing_csv.parent)
            timing_output_path = (
                timing_csv
                if args.dry_run
                else build_staging_timing_path(timing_csv)
            )
            try:
                temp_cfg, experiment_output = override_config(
                    base_cfg,
                    scene,
                    setup_key,
                    size,
                    run_idx,
                    args.frame_count,
                    args.warmup_frames,
                    timing_output_path,
                    not args.enable_eval,
                )
            except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
                failure_count += 1
                print(f"    FAILED: could not create config: {error}")
                continue

            collisions = [
                path
                for path in (timing_csv, experiment_output)
                if path.exists()
            ]
            if collisions and not args.overwrite:
                failure_count += 1
                print(
                    "    FAILED: refusing to overwrite existing path(s): "
                    + ", ".join(str(path) for path in collisions)
                )
                print(
                    "    Use --start-run with a new index or pass "
                    "--overwrite explicitly."
                )
                temp_cfg.unlink(missing_ok=True)
                continue

            try:
                merged_config = load_merged_config(temp_cfg)
                expected_counts = build_expected_iteration_counts(merged_config)
            except (
                KeyError,
                OSError,
                TypeError,
                ValueError,
                yaml.YAMLError,
            ) as error:
                failure_count += 1
                print(f"    FAILED: invalid timing config: {error}")
                if not args.keep_temp_configs:
                    temp_cfg.unlink(missing_ok=True)
                continue

            size_text = "Default" if size is None else str(size)
            print(
                f"--> [Run {run_idx}] Scene={scene} | "
                f"Setup={setup_key} | Size={size_text} | "
                f"Timing={timing_csv}"
            )

            if args.dry_run:
                print(f"    DRY RUN: wrote temp config {temp_cfg}")
                if not args.keep_temp_configs:
                    try:
                        temp_cfg.unlink(missing_ok=True)
                    except OSError as error:
                        print(
                            f"    WARNING: could not remove {temp_cfg}: {error}",
                            file=sys.stderr,
                        )
                continue

            start_time = time.perf_counter()
            try:
                subprocess.run(
                    [
                        args.python_exec,
                        str(args.target_script),
                        str(temp_cfg),
                    ],
                    check=True,
                )
                timing_counts = validate_timing_csv(
                    timing_output_path,
                    expected_counts,
                )
                timing_output_path.replace(timing_csv)
                duration = time.perf_counter() - start_time
                append_log_row(
                    args.log_file,
                    run_idx,
                    scene,
                    setup_key,
                    size,
                    args.frame_count,
                    args.warmup_frames,
                    timing_csv,
                    duration,
                )
                success_count += 1
                print(
                    f"    SUCCESS: finished in {duration:.2f}s | "
                    f"tracking iterations={timing_counts['tracking']} | "
                    "bundle-adjustment iterations="
                    f"{timing_counts['bundle_adjustment']}"
                )
                if not args.keep_temp_configs:
                    try:
                        temp_cfg.unlink(missing_ok=True)
                    except OSError as error:
                        print(
                            f"    WARNING: could not remove {temp_cfg}: {error}",
                            file=sys.stderr,
                        )
            except subprocess.CalledProcessError as error:
                failure_count += 1
                print(
                    "    FAILED: execution error "
                    f"(exit code {error.returncode})"
                )
                print(f"    Temp config left for inspection: {temp_cfg}")
                if timing_output_path.exists():
                    print(
                        "    Partial timing data left for inspection: "
                        f"{timing_output_path}"
                    )
            except (OSError, RuntimeError) as error:
                failure_count += 1
                print(f"    FAILED: {error}")
                print(f"    Temp config left for inspection: {temp_cfg}")
                if timing_output_path.exists():
                    print(
                        "    Partial timing data left for inspection: "
                        f"{timing_output_path}"
                    )
            except KeyboardInterrupt:
                print("\n    ABORTED: user interrupted.")
                temp_cfg.unlink(missing_ok=True)
                sys.exit(130)

    if args.dry_run:
        print("\nDry run complete.")
        if failure_count:
            sys.exit(1)
    else:
        print(
            f"\nAll scheduled runs attempted: {success_count} succeeded, "
            f"{failure_count} failed. Successful runs are logged in "
            f"{args.log_file}"
        )
        if failure_count:
            sys.exit(1)


if __name__ == "__main__":
    main()
