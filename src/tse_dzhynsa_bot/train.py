"""Train and compare the hand rule and TF-IDF + logistic regression on a time split,
log both runs to MLflow with lineage tags, and register the model as `champion`.

    uv run python -m tse_dzhynsa_bot.train --register
"""

import argparse
import json
import subprocess
from datetime import date
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tse_dzhynsa_bot import rule
from tse_dzhynsa_bot.features import (
    FEATURE_COLUMNS,
    FEATURE_VERSION,
    build_features,
    clean,
    read_raw_range,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "dzhynsa-tfidf"
SPLIT = {  # design.md 5.2: train Jun-Aug, validation Sep, test Oct 1-27
    "train": (date(2026, 6, 1), date(2026, 8, 31)),
    "val": (date(2026, 9, 1), date(2026, 9, 30)),
    "test": (date(2026, 10, 1), date(2026, 10, 27)),
}
PARAMS = {
    "C": 4.0,
    "word_ngrams": (1, 2),
    "char_ngrams": (3, 5),
    "min_df": 3,
    "max_features": 100_000,
    "min_val_precision": 0.70,
}


def data_version(root: Path = ROOT) -> str:
    """md5 of the raw data as DVC recorded it in data/raw.dvc."""
    meta = yaml.safe_load((root / "data" / "raw.dvc").read_text())
    return meta["outs"][0]["md5"]


def git_commit(root: Path = ROOT) -> str:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "src", "configs"], cwd=root)
    return sha + ("-dirty" if dirty.returncode else "")


def load_dataset(raw_dir: Path) -> pd.DataFrame:
    raw = read_raw_range(raw_dir, SPLIT["train"][0], SPLIT["test"][1])
    return build_features(clean(raw))


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {k: df[(df["day"] >= s) & (df["day"] <= e)] for k, (s, e) in SPLIT.items()}


def make_model(p: dict) -> Pipeline:
    text = ColumnTransformer(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=p["word_ngrams"],
                    min_df=p["min_df"],
                    max_features=p["max_features"],
                    sublinear_tf=True,
                ),
                "text_clean",
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=p["char_ngrams"],
                    min_df=p["min_df"],
                    max_features=p["max_features"],
                    sublinear_tf=True,
                ),
                "text_clean",
            ),
            ("num", StandardScaler(), FEATURE_COLUMNS),
        ]
    )
    clf = LogisticRegression(C=p["C"], class_weight="balanced", max_iter=2000, random_state=0)
    return Pipeline([("features", text), ("clf", clf)])


def pick_threshold(y: np.ndarray, p: np.ndarray, min_precision: float) -> float:
    """Highest recall on validation while `ad` precision stays >= min_precision."""
    best = (0.5, -1.0)
    for t in np.linspace(0.05, 0.95, 91):
        pred = p >= t
        if pred.sum() and precision_score(y, pred) >= min_precision:
            r = recall_score(y, pred)
            if r > best[1]:
                best = (float(t), r)
    return best[0]


def scores(y, pred, prob=None) -> dict:
    return {
        "pr_auc": float(average_precision_score(y, pred if prob is None else prob)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--register", action="store_true", help="register the model as champion")
    ap.add_argument("--reproduces", default="", help="run id this run reproduces")
    args = ap.parse_args()

    tags = {
        "data_version": data_version(),
        "git_commit": git_commit(),
        "feature_version": FEATURE_VERSION,
    }
    if args.reproduces:
        tags["reproduces"] = args.reproduces
    parts = split(load_dataset(args.raw))
    sizes = {f"n_{k}": len(v) for k, v in parts.items()}
    ad_share = {f"ad_share_{k}": float(v["label_ad"].mean()) for k, v in parts.items()}
    test = parts["test"]
    mlflow.set_experiment("dzhynsa-lab2")

    with mlflow.start_run(run_name="rule"):
        mlflow.set_tags(tags | {"model": "rule"})
        mlflow.log_params({"split": json.dumps(SPLIT, default=str)} | sizes)
        rule_scores = scores(test["label_ad"], rule.predict(test))
        mlflow.log_metrics({f"test_{k}": v for k, v in rule_scores.items()} | ad_share)

    with mlflow.start_run(run_name="tfidf_logreg") as run:
        mlflow.set_tags(tags | {"model": "tfidf_logreg"})
        model = make_model(PARAMS).fit(parts["train"], parts["train"]["label_ad"])
        thr = pick_threshold(
            parts["val"]["label_ad"].to_numpy(),
            model.predict_proba(parts["val"])[:, 1],
            PARAMS["min_val_precision"],
        )
        prob = model.predict_proba(test)[:, 1]
        model_scores = scores(test["label_ad"], prob >= thr, prob)
        mlflow.log_params({k: str(v) for k, v in PARAMS.items()} | sizes | {"threshold": thr})
        mlflow.log_metrics({f"test_{k}": v for k, v in model_scores.items()} | ad_share)
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=test.head(3),
            serialization_format="cloudpickle",
        )

    if args.register:
        mv = mlflow.register_model(info.model_uri, MODEL_NAME)
        client = MlflowClient()
        for k, v in tags.items():
            client.set_model_version_tag(MODEL_NAME, mv.version, k, v)
        client.set_model_version_tag(MODEL_NAME, mv.version, "threshold", str(thr))
        client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
        print(f"registered {MODEL_NAME} v{mv.version} as champion")

    print(
        json.dumps({"run_id": run.info.run_id, "rule": rule_scores, "tfidf_logreg": model_scores})
    )


if __name__ == "__main__":
    main()
