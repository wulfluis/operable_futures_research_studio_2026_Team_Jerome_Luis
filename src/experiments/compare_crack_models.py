"""Two crack models on the same held-out test photos, per group (D100): the old public close-ups (as D32 and D48),
Özgenel building concrete [49], the EPFL buildings after earthquakes [108] and the EPFL lab walls [109]. Missed cracks
first, then false alarms, as the team judged v4 (D48). Usage, from the repository root:
    py src/experiments/compare_crack_models.py models/unet_crack_v4.pt models/unet_crack_v5.pt
Runs src/segmentation/evaluate.py for each model and group into data/results/compare_models/ (local) and prints a table."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GROUPS = {
    "old public close-ups": ["--exclude", "metu_", "sdnet_", "ozg_", "topo_", "dic_"],
    "Özgenel building concrete": ["--only", "ozg_"],
    "EPFL buildings (TOPO)": ["--only", "topo_"],
    "EPFL lab walls (DIC)": ["--only", "dic_"],
}


def run(model: Path, group: str, flags: list) -> dict:
    out = ROOT / "data" / "results" / "compare_models" / model.stem / group.split()[0].lower()
    subprocess.run([sys.executable, str(ROOT / "src" / "segmentation" / "evaluate.py"), str(ROOT / "data" / "processed" /
                    "segmentation"), "--split", "test", "--model", str(model), "--out", str(out)] + flags,
                   check=True, capture_output=True)
    return json.loads((out / "test" / "scores.json").read_text(encoding="utf-8"))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("models", type=Path, nargs="+")
    args = ap.parse_args(argv)
    table = {}
    for g, flags in GROUPS.items():
        for m in args.models:
            table[(g, m.stem)] = run(m, g, flags)
            print(g, m.stem, json.dumps(table[(g, m.stem)])[:300], flush=True)
    out = ROOT / "data" / "results" / "compare_models" / "summary.json"
    old = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}  # earlier models stay in the table
    old.update({f"{g} | {m}": v for (g, m), v in table.items()})
    out.write_text(json.dumps(old, indent=1), encoding="utf-8")
    print("written", out)


if __name__ == "__main__":
    main()
