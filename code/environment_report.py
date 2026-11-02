"""Write results/environment.txt: software versions and machine used (for the reproducibility table of the paper).
Usage: python environment_report.py --out results"""
import argparse, importlib, os, platform, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results")
a = ap.parse_args()
lines = [f"python {sys.version.split()[0]}", f"platform {platform.platform()}", f"processor {platform.processor()}",
         f"logical_cpus {os.cpu_count()}"]
for m in ("numpy", "scipy", "pandas", "sklearn", "joblib", "matplotlib", "qiskit", "qiskit_aer", "qiskit_ibm_runtime", "pennylane"):
    try:
        lines.append(f"{m} {importlib.import_module(m).__version__}")
    except Exception as e:
        lines.append(f"{m} not installed")
Path(a.out).mkdir(exist_ok=True, parents=True)
(Path(a.out) / "environment.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
