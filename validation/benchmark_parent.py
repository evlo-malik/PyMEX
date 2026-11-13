"""Run matched native workflows using the exact parent and optimized sources.

This supplements, rather than overwrites, the original 42-run experiment.
Both source directories and the original references must already be present.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import yaml

from test_independent_shortcut import compare, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    root, scratch = args.project_root, args.scratch
    test = root / "independent-shortcut-validation"
    parent = test / "baseline/src"
    optimized = test / "src"
    files = sorted(p.name for p in optimized.glob("*.py"))
    changed = [name for name in files if digest(parent / name) != digest(optimized / name)]
    assert changed == ["bse.py"], changed
    production = root / "code/PyMEX/src"
    before = {p.name: digest(p) for p in production.glob("*.py")}
    python = root / "code/.venv/bin/python"
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OMP_THREAD_LIMIT="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONUNBUFFERED="1",
               PYTHONDONTWRITEBYTECODE="1", VALIDATION_PROJECT_ROOT=str(root))
    env["PATH"] = str(python.parent) + os.pathsep + env["PATH"]
    records = []
    for material in ("hBN", "MoS2"):
        reference = (root / "mac-archive/optical/results/independent/runs/N030"
                     if material == "hBN" else
                     root / "tmds/MoS2/independent/code/tmds/optical/results/independent/runs/N030")
        for baseline in (True, False):
            name = f"{material}-independent-N030-p1-parent-{'baseline' if baseline else 'optimized'}"
            folder = scratch / name
            folder.mkdir(exist_ok=False)
            config = yaml.safe_load((reference / "pymex_tb.yaml").read_text())
            config["bse"]["tb"]["model"] = "model.py"
            assert config["diagonalize"] == {"library": "lapack"}
            (folder / "pymex_tb.yaml").write_text(yaml.safe_dump(config))
            shutil.copy2(reference / "model.py", folder / "model.py")
            env["VALIDATION_SOURCE"] = str(parent if baseline else optimized)
            command = ["nice", "-n", "10", str(root / "code/.venv/bin/mpiexec"),
                       "-n", "1", str(python), str(test / "driver.py")]
            print("RUN", name, flush=True)
            with (folder / "run.log").open("w") as log:
                subprocess.run(command, cwd=folder, env=env, stdout=log,
                               stderr=subprocess.STDOUT, timeout=900, check=True)
            check = compare(reference, folder, scratch)
            record = dict(case=name, reference=(str(reference.relative_to(root))
                          if reference.is_relative_to(root) else reference.name),
                          material=material, mode="independent", mesh=30, ranks=1,
                          unoptimized_baseline=baseline,
                          timings=json.loads((folder / "timings.json").read_text()),
                          input_hashes={f: digest(folder / f) for f in ("pymex_tb.yaml", "model.py")},
                          native_output_hashes={p.name: digest(p) for p in folder.iterdir()
                                                if p.suffix in (".hdf5", ".txt")}, **check)
            (test / "records" / (name + ".json")).write_text(json.dumps(record, indent=2))
            shutil.copy2(folder / "run.log", test / "records" / (name + ".log"))
            assert check["passed"], record
            records.append(record)
            if baseline:
                reference = folder
            print("PASS", name, flush=True)
    assert before == {p.name: digest(p) for p in production.glob("*.py")}
    report = dict(actual_runs=len(records), all_passed=True,
                  baseline_commit="67165cc", only_changed_source="bse.py",
                  production_sources_unchanged=True,
                  baseline_source_hashes={n: digest(parent / n) for n in files},
                  optimized_source_hashes={n: digest(optimized / n) for n in files},
                  results=records)
    (test / "parent-benchmark.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
