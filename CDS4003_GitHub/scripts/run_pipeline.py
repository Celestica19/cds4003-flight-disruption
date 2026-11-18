"""Run the real-data project stages; stop immediately on an error."""
import argparse
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-download", action="store_true", help="Use original CSVs already placed in data/raw")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    stages = ([] if args.skip_download else ["download_data.py"]) + ["prepare_data.py", "train_models.py", "verify_trained_artifact.py", "build_visuals.py", "build_evidence.py"]
    for name in stages:
        command = [sys.executable, str(ROOT / "scripts" / name)]
        if args.config and name not in ["download_data.py", "build_evidence.py", "verify_trained_artifact.py"]:
            command += ["--config", str(args.config.resolve())]
        print(f"Running {name}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
