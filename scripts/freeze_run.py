import os
import sys
import glob
import json
import shutil
import argparse
from datetime import datetime
import subprocess

def get_latest_run_dir(logs_dir="logs"):
    if not os.path.isdir(logs_dir):
        return None
    date_dirs = [os.path.join(logs_dir, d) for d in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, d))]
    if not date_dirs:
        return None
    
    run_dirs = []
    for d in date_dirs:
        for t in os.listdir(d):
            p = os.path.join(d, t)
            if os.path.isdir(p):
                run_dirs.append(p)
    if not run_dirs:
        return None
    
    return max(run_dirs, key=os.path.getmtime)

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description="Freeze the latest run artifacts for reproducibility.")
    parser.add_argument("--run-dir", default=None, help="Explicit run directory to freeze. If None, uses latest in logs/")
    parser.add_argument("--out-dir", default="results", help="Output directory to store frozen artifacts")
    parser.add_argument("--tag", default="baseline", help="Tag to prepend to the run_id")
    parser.add_argument("--data-dir", default=None, help="Optional dataset directory to store in metadata")
    args = parser.parse_args()

    run_dir = args.run_dir
    if not run_dir:
        run_dir = get_latest_run_dir()
    
    if not run_dir or not os.path.isdir(run_dir):
        print("Error: Could not find a valid run directory under logs/.")
        sys.exit(1)

    print(f"Freezing run: {run_dir}")

    # Derive run_id from run_dir (e.g., logs/2026-03-09/02-46-42 -> 2026-03-09_02-46-42)
    parts = os.path.normpath(run_dir).split(os.sep)
    if len(parts) >= 3 and parts[-3] == "logs":
        run_id = f"{parts[-2]}_{parts[-1]}"
    else:
        run_id = os.path.basename(run_dir)
    
    if args.tag:
        run_id = f"{args.tag}_{run_id}"

    # Metadata collection
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "run_dir": run_dir,
        "git_commit": get_git_commit(),
        "python_version": sys.version,
        "data_dir": args.data_dir
    }

    try:
        import torch
        metadata["torch_version"] = torch.__version__
        metadata["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            metadata["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    # Find configs
    config_src = os.path.join(run_dir, "hydra_configs")
    if not os.path.isdir(config_src):
        config_src = os.path.join(run_dir, ".hydra")
    
    # Extract checkpoints
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    best_ckpt = None
    if os.path.isdir(ckpt_dir):
        ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
        if ckpts:
            best_ckpts = [c for c in ckpts if "best" in c.lower()]
            if best_ckpts:
                best_ckpt = max(best_ckpts, key=os.path.getmtime)
            else:
                best_ckpt = max(ckpts, key=os.path.getmtime)
    
    metadata["checkpoint_path"] = best_ckpt

    # Extract metrics from logs
    val_cer = None
    test_cer = None
    
    log_files = glob.glob(os.path.join(run_dir, "*.log")) + glob.glob(os.path.join(run_dir, "*.txt"))
    for log_file in log_files:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "val/cer" in line.lower() or "val_cer" in line.lower() or "cer" in line.lower():
                    import re
                    # Look for val CER
                    if "val" in line.lower():
                        m = re.search(r"val_?/?cer.*?([0-9]+\.[0-9]+)", line.lower())
                        if m: val_cer = float(m.group(1))
                if "test/cer" in line.lower() or "test_cer" in line.lower():
                    import re
                    m = re.search(r"test_?/?cer.*?([0-9]+\.[0-9]+)", line.lower())
                    if m: test_cer = float(m.group(1))

    metadata["val_cer"] = val_cer
    metadata["test_cer"] = test_cer
    metadata["metrics_note"] = "Extracted from text logs. If null, either not evaluated or log parsing failed."

    # Write artifacts
    run_out_dir = os.path.join(args.out_dir, "runs", run_id)
    os.makedirs(run_out_dir, exist_ok=True)

    if os.path.isdir(config_src):
        dest_config = os.path.join(run_out_dir, "config_dump")
        if os.path.exists(dest_config):
            shutil.rmtree(dest_config)
        shutil.copytree(config_src, dest_config)

    with open(os.path.join(run_out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    summary = f"""# Run Summary: {run_id}
- **Timestamp**: {metadata['timestamp']}
- **Commit**: {metadata['git_commit']}
- **Checkpoint**: {metadata['checkpoint_path']}
- **Val CER**: {metadata['val_cer']}
- **Test CER**: {metadata['test_cer']}
"""
    with open(os.path.join(run_out_dir, "summary.md"), "w") as f:
        f.write(summary)

    # LATEST
    latest_md = os.path.join(args.out_dir, "BASELINE_LATEST.md")
    latest_json = os.path.join(args.out_dir, "BASELINE_LATEST.json")

    with open(latest_md, "w") as f:
        f.write(f"# Latest Baseline\nRun ID: {run_id}\n\nSee `runs/{run_id}` for details.\n\n")
        f.write(f"- **Val CER**: {val_cer}\n- **Test CER**: {test_cer}\n- **Checkpoint**: `{metadata['checkpoint_path']}`\n")
    
    with open(latest_json, "w") as f:
        json.dump({
            "latest_run_id": run_id,
            "val_cer": val_cer,
            "test_cer": test_cer,
            "checkpoint_path": metadata['checkpoint_path']
        }, f, indent=2)

    print(f"Successfully froze run artifacts to {run_out_dir}")
    print(f"Val CER: {val_cer} | Test CER: {test_cer}")

if __name__ == "__main__":
    main()
