#!/usr/bin/env python3
import argparse
import sys
import os
import subprocess
import time
import yaml
import itertools
from pathlib import Path

# --- Configuration Definitions ---
SETUP_DEFINITIONS = {
    "A": {"hash": "CoherentPrime", "morton_sort": False},
    "B": {"hash": "Morton",        "morton_sort": False},
    "C": {"hash": "CoherentPrime", "morton_sort": True},
    "D": {"hash": "Morton",        "morton_sort": True},
}

DEFAULT_EXCLUDES = {"office2"}

def read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def write_yaml(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def build_exp_name(setup: str, size: str) -> str:
    size_str = "DefaultSize" if size is None else f"Size{size}"
    return f"{setup}_{size_str}"

def override_config(base_cfg_path: Path, scene: str, setup_key: str, 
                    table_size: int | None) -> Path:
    cfg = read_yaml(base_cfg_path)

    # --- CRITICAL FIX: Ensure 'HashGrid' section exists ---
    # Model.py explicitly looks for config['HashGrid'], not ['grid']
    cfg.setdefault("HashGrid", {})
    cfg.setdefault("data", {})
    
    # 1. Apply Setup
    setup_params = SETUP_DEFINITIONS[setup_key]
    
    # Write to HashGrid as requested
    cfg["HashGrid"]["hash"] = setup_params["hash"] 
    cfg["HashGrid"]["morton_sort"] = setup_params["morton_sort"]

    # 2. Apply Hash Table Size
    if table_size is not None:
        cfg["HashGrid"]["hash_size"] = table_size

    # 3. Construct Experiment Name and Output Path
    exp_dir_name = build_exp_name(setup_key, table_size)
    cfg["data"]["exp_name"] = exp_dir_name 

    base_output = cfg["data"].get("output", f"output/Replica/{scene}")
    if base_output.endswith("/"):
        base_output = base_output[:-1]
    cfg["data"]["output"] = base_output

    # 4. Write Temporary Config
    # Save in the SAME folder as base config to preserve relative inheritance
    tmp_cfg_name = f"temp_{scene}_{exp_dir_name}.yaml"
    tmp_cfg_path = base_cfg_path.parent / tmp_cfg_name
    
    write_yaml(tmp_cfg_path, cfg)
    
    return tmp_cfg_path

def parse_size_arg(value):
    if str(value).lower() in ["null", "none", "default"]:
        return None
    return int(value)

def main():
    parser = argparse.ArgumentParser(description="Universal Batch Runner for EC-SLAM (HashGrid Fix)")

    # --- Sweep Parameters ---
    parser.add_argument("--setups", nargs="+", default=["A", "B", "C", "D"], choices=["A", "B", "C", "D"],
                        help="List of setups to run.")
    parser.add_argument("--sizes", nargs="+", default=["null"], type=str,
                        help="List of hashmap log2 sizes (e.g., 19). Use 'null' for default.")

    # --- Scene Selection ---
    parser.add_argument("--scenes", nargs="+", default=None,
                        help="Explicit list of scene names to run (e.g. office0 room1).")
    parser.add_argument("--configs-root", default="configs/Replica", type=Path,
                        help="Folder containing base scene YAML files.")
    parser.add_argument("--exclude", nargs="*", default=[], 
                        help="Scenes to exclude if scanning directory.")

    # --- Execution & Logging ---
    parser.add_argument("--log-file", default="output/execution_times.csv", type=Path,
                        help="Path to the global CSV log file.")
    parser.add_argument("--python-exec", default=sys.executable,
                        help="Python executable to use.")

    args = parser.parse_args()

    # --- Check for run.py ---
    target_script = "run.py"
    if not Path(target_script).exists():
        if Path("coslam.py").exists():
             target_script = "coslam.py" 
        else:
            print(f"ERROR: '{target_script}' not found in current directory.")
            sys.exit(1)
        
    if not args.configs_root.exists():
        print(f"ERROR: Configs root not found: {args.configs_root.resolve()}")
        sys.exit(1)

    # 1. Discover Scenes
    if args.scenes:
        scenes = args.scenes
    else:
        excludes = set(map(str, DEFAULT_EXCLUDES)) | set(args.exclude)
        scenes = []
        for yml in sorted(args.configs_root.glob("*.yaml")):
            # Filter out temp files
            if yml.stem not in excludes and not yml.name.startswith("temp_") and yml.stem != "replica":
                scenes.append(yml.stem)

    if not scenes:
        print(f"ERROR: No scenes found in {args.configs_root}.")
        sys.exit(0)

    # 2. Prepare Log File
    ensure_dir(args.log_file.parent)
    if not args.log_file.exists():
        with open(args.log_file, "w", encoding="utf-8") as f:
            f.write("Scene,Setup,TableSize,Time_Seconds\n")

    # 3. Build Combination Grid
    parsed_sizes = [parse_size_arg(s) for s in args.sizes]
    combinations = list(itertools.product(scenes, args.setups, parsed_sizes))

    print(f"Total Runs Scheduled: {len(combinations)}")
    print(f"  > Configs Root: {args.configs_root}")
    print(f"  > Target Script: {target_script}")
    print("-" * 60)

    # 4. Main Execution Loop
    for scene, setup, size in combinations:
        base_cfg = args.configs_root / f"{scene}.yaml"
        
        if not base_cfg.exists():
            print(f"SKIPPING: Config file not found: {base_cfg}")
            continue

        # Create temporary config
        tmp_cfg = override_config(base_cfg, scene, setup, size)

        print(f"\n--> STARTING: Scene={scene} | Setup={setup} | Size={size}")
        print(f"    Config: {tmp_cfg}")

        start_time = time.time()
        try:
            # Positional argument execution: python run.py config_path
            cmd = [args.python_exec, target_script, str(tmp_cfg)]
            
            subprocess.run(cmd, check=True)  
            
            duration = time.time() - start_time
            
            size_log = "Default" if size is None else str(size)
            log_line = f"{scene},{setup},{size_log},{duration:.4f}"
            
            with open(args.log_file, "a", encoding="utf-8") as f:
                f.write(log_line + "\n")
            
            print(f"    SUCCESS: Finished in {duration:.2f}s")
            
            # Optional: Clean up temp file
            if tmp_cfg.exists(): os.remove(tmp_cfg)

        except subprocess.CalledProcessError as e:
            print(f"    FAILED: Execution error (Exit Code: {e.returncode})")
        except KeyboardInterrupt:
            print("\n    ABORTED: User interrupted.")
            if tmp_cfg.exists(): os.remove(tmp_cfg)
            sys.exit(1)
        except Exception as e:
            print(f"    ERROR: Unexpected script error: {e}")

    print(f"\nAll runs completed. Results saved to {args.log_file}")

if __name__ == "__main__":
    main()