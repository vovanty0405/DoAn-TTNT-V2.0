"""Parity verification test comparing legacy DoAn models and HeartCare-AI models.

Runs legacy and new inference in separate Python subprocesses so there is zero
module-name collision for `heart_pipeline`.
"""

import json
import math
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEGACY_ROOT = Path(r"D:\Năm 4\Trí tuệ nhân tạo-20260820T130944Z-1-001\DoAn")

LEGACY_SCRIPT = r"""
import json, sys
from pathlib import Path
import joblib
import pandas as pd

legacy_root = Path(sys.argv[1])
sys.path.insert(0, str(legacy_root))
from heart_pipeline import FEATURES, disease_score, normalize_input

exp = json.loads((legacy_root / "results" / "experiment.json").read_text(encoding="utf-8"))
samples = json.loads((legacy_root / "results" / "demo_samples.json").read_text(encoding="utf-8"))
out = {"selected_model": exp["selected_model"], "features": FEATURES, "predictions": {}}

for name in exp["models"]:
    mdl = joblib.load(legacy_root / "results" / "models" / (name.replace(" ", "_") + ".joblib"))
    preds = []
    for s in samples:
        frame = normalize_input(pd.DataFrame([s["features"]], columns=FEATURES))
        preds.append({
            "source_row": s["source_row"],
            "pred": int(mdl.predict(frame)[0]),
            "score": float(disease_score(mdl, frame)[0]),
        })
    out["predictions"][name] = preds

print(json.dumps(out))
"""

NEW_SCRIPT = r"""
import json, sys
from pathlib import Path

project_root = Path(sys.argv[1])
sys.path.insert(0, str(project_root))
from src.config import FEATURES
from src.data_loader import load_demo_samples
from src.prediction import get_default_model_name, load_all_models, predict_single

models = load_all_models()
samples = load_demo_samples()
out = {
    "selected_model": get_default_model_name(),
    "features": FEATURES,
    "predictions": {},
}

for name, mdl in models.items():
    preds = []
    for s in samples:
        pred, score, _ = predict_single(mdl, s["features"])
        preds.append({
            "source_row": s["source_row"],
            "pred": pred,
            "score": score,
        })
    out["predictions"][name] = preds

print(json.dumps(out))
"""


def run_isolated(script: str, root: Path) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", script, str(root)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def main() -> None:
    print(f"Python executable: {sys.executable}")
    legacy_res = run_isolated(LEGACY_SCRIPT, LEGACY_ROOT)
    new_res = run_isolated(NEW_SCRIPT, PROJECT_ROOT)

    assert legacy_res["selected_model"] == new_res["selected_model"], (
        f"Default model mismatch: {legacy_res['selected_model']} vs {new_res['selected_model']}"
    )
    assert legacy_res["features"] == new_res["features"] and len(new_res["features"]) == 13, (
        "Feature list mismatch!"
    )

    print(f"Default model: {new_res['selected_model']} (MATCH)")
    print(f"Features ({len(new_res['features'])}): {new_res['features']} (MATCH)")

    all_ok = True
    for model_name in legacy_res["predictions"]:
        old_list = legacy_res["predictions"][model_name]
        new_list = new_res["predictions"][model_name]
        for o, n in zip(old_list, new_list):
            pred_match = o["pred"] == n["pred"]
            score_match = math.isclose(o["score"], n["score"], rel_tol=1e-12, abs_tol=1e-12)
            status = "PASS" if (pred_match and score_match) else "DIFF"
            if status != "PASS":
                all_ok = False
            print(
                f"  [{status}] {model_name:14s} | row={o['source_row']} | "
                f"pred: old={o['pred']} new={n['pred']} | "
                f"score: old={o['score']:.12f} new={n['score']:.12f}"
            )

    if not all_ok:
        sys.exit(1)
    print("\nALL 3 MODELS PASSED 100% PARITY VERIFICATION.")


if __name__ == "__main__":
    main()

