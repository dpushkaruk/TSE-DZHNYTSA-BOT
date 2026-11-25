"""Import the XLM-R model trained on Kaggle into MLflow and compare it with the champion.

    uv run --env-file .env python -m tse_dzhynsa_bot.import_xlmr --zip xlmr_dzhynsa.zip
    uv run --env-file .env python -m tse_dzhynsa_bot.import_xlmr --zip xlmr_dzhynsa.zip --promote

Metrics are computed here, not trusted from Kaggle: Kaggle predictions are joined to the
local labels by (channel, id), the threshold is picked on validation exactly like train.py,
and a sample of test posts is re-scored with the int8 ONNX model on this CPU to check that
the logged artifact gives the same numbers. The model is registered as dzhynsa-xlmr
(alias challenger); with --promote it also gets alias champion if its test PR-AUC beats
dzhynsa-tfidf@champion. Serving switches with MODEL_NAME=dzhynsa-xlmr in .env.
"""

import argparse
import json
import tempfile
import zipfile
from importlib.metadata import version
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from tse_dzhynsa_bot.train import (
    MODEL_NAME as BASELINE_NAME,
)
from tse_dzhynsa_bot.train import (
    PARAMS,
    ROOT,
    data_version,
    git_commit,
    load_dataset,
    pick_threshold,
    scores,
    split,
)
from tse_dzhynsa_bot.xlmr_model import XlmrOnnxClassifier

MODEL_NAME = "dzhynsa-xlmr"
EXPERIMENT = "dzhynsa-lab2"
MAX_VERIFY_DIFF = 0.05


def attach_preds(part: pd.DataFrame, preds: pd.DataFrame, name: str) -> np.ndarray:
    """Kaggle probabilities in the order of the local rows; every local post must have one."""
    merged = part[["channel", "id"]].merge(preds, on=["channel", "id"], how="left")
    missing = int(merged["prob_ad"].isna().sum())
    if missing or len(preds) != len(part):
        raise SystemExit(
            f"{name}: {missing} local posts without a Kaggle prediction, "
            f"{len(preds)} predictions for {len(part)} posts. Re-export the data and retrain."
        )
    return merged["prob_ad"].to_numpy()


def verify_onnx(art: Path, test: pd.DataFrame, p_kaggle: np.ndarray, n: int) -> dict:
    """Re-score up to n test posts (all ads first) with the artifact that gets logged."""
    clf = XlmrOnnxClassifier()

    class Ctx:
        artifacts = {
            "onnx_model": str(art / "model.onnx"),
            "tokenizer": str(art / "tokenizer.json"),
        }

    clf.load_context(Ctx)
    pos = np.flatnonzero(test["label_ad"].to_numpy() == 1)[: n // 2]
    neg = np.flatnonzero(test["label_ad"].to_numpy() == 0)
    neg = np.random.default_rng(0).choice(neg, size=min(n - len(pos), len(neg)), replace=False)
    idx = np.concatenate([pos, neg])
    p_local = clf.predict_texts(test["text_clean"].iloc[idx].tolist())
    diff = np.abs(p_local - p_kaggle[idx])
    return {
        "verify_n": len(idx),
        "verify_max_abs_diff": float(diff.max()),
        "verify_mean_abs_diff": float(diff.mean()),
    }


def champion_pr_auc(client: MlflowClient) -> tuple[str, float | None]:
    try:
        mv = client.get_model_version_by_alias(BASELINE_NAME, "champion")
    except mlflow.exceptions.MlflowException:
        return "none", None
    run = client.get_run(mv.run_id)
    return f"{BASELINE_NAME}/{mv.version}", run.data.metrics.get("test_pr_auc")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, help="xlmr_dzhynsa.zip downloaded from Kaggle")
    ap.add_argument("--dir", type=Path, help="or the unzipped folder")
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--verify-n", type=int, default=200)
    ap.add_argument("--promote", action="store_true", help="alias champion if it beats TF-IDF")
    ap.add_argument("--no-register", action="store_true")
    args = ap.parse_args()
    if not (args.zip or args.dir):
        ap.error("pass --zip or --dir")

    with tempfile.TemporaryDirectory() as tmp:
        art = args.dir
        if args.zip:
            art = Path(tmp)
            zipfile.ZipFile(args.zip).extractall(art)
        run(art, args)


def run(art: Path, args: argparse.Namespace) -> None:
    meta = json.loads((art / "train_meta.json").read_text())
    dv = data_version()
    if meta["manifest"]["data_version"] != dv:
        raise SystemExit(
            f"Kaggle trained on data {meta['manifest']['data_version']}, local data is {dv}"
        )
    if meta.get("max_rows"):
        print(f"WARNING: smoke run on {meta['max_rows']} rows per split")

    parts = split(load_dataset(args.raw))
    if meta.get("max_rows"):
        parts = {k: v.head(meta["max_rows"]) for k, v in parts.items()}
    val, test = parts["val"], parts["test"]
    p_val = attach_preds(val, pd.read_parquet(art / "preds_val.parquet"), "val")
    p_test = attach_preds(test, pd.read_parquet(art / "preds_test.parquet"), "test")

    thr = pick_threshold(val["label_ad"].to_numpy(), p_val, PARAMS["min_val_precision"])
    test_scores = scores(test["label_ad"], p_test >= thr, p_test)
    val_pr_auc = scores(val["label_ad"], p_val >= thr, p_val)["pr_auc"]
    check = verify_onnx(art, test, p_test, args.verify_n)
    if check["verify_max_abs_diff"] > MAX_VERIFY_DIFF:
        raise SystemExit(f"ONNX on this CPU disagrees with Kaggle: {check}")

    client = MlflowClient()
    champ, champ_pr_auc = champion_pr_auc(client)
    beats = champ_pr_auc is None or test_scores["pr_auc"] > champ_pr_auc
    tags = {
        "data_version": dv,
        "git_commit": git_commit(),
        "feature_version": "text_clean",
        "model": "xlmr_onnx_int8",
        "base_model": meta["params"]["model_name"],
        "trained_on": f"kaggle {meta['versions']['gpu']}",
        "kaggle_git_commit": meta["manifest"]["git_commit"],
        "compared_with": champ,
        "beats_champion": str(beats),
    }
    sizes = {f"n_{k}": len(v) for k, v in parts.items()}

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="xlmr_onnx") as r:
        mlflow.set_tags(tags)
        mlflow.log_params(
            {k: str(v) for k, v in meta["params"].items()}
            | sizes
            | {
                "threshold": thr,
                "best_epoch": meta["best_epoch"],
                "min_val_precision": PARAMS["min_val_precision"],
            }
        )
        for h in meta["history"]:
            mlflow.log_metrics(
                {"train_loss": h["train_loss"], "epoch_val_pr_auc": h["val_pr_auc"]},
                step=h["epoch"],
            )
        mlflow.log_metrics(
            {f"test_{k}": v for k, v in test_scores.items()}
            | {"val_pr_auc": val_pr_auc, "train_minutes": meta["train_minutes"]}
            | {f"onnx_{k}": v for k, v in meta["onnx_check"].items()}
            | check
            | {"model_size_mb": (art / "model.onnx").stat().st_size / 1e6}
            | ({"champion_test_pr_auc": champ_pr_auc} if champ_pr_auc is not None else {})
        )
        mlflow.log_artifact(str(art / "train_meta.json"))
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=XlmrOnnxClassifier(),
            artifacts={
                "onnx_model": str(art / "model.onnx"),
                "tokenizer": str(art / "tokenizer.json"),
            },
            input_example=test[["text_clean"]].head(3),
            pip_requirements=[
                f"onnxruntime=={version('onnxruntime')}",
                f"tokenizers=={version('tokenizers')}",
                f"pandas=={version('pandas')}",
                f"numpy=={version('numpy')}",
            ],
        )

    if not args.no_register:
        mv = mlflow.register_model(info.model_uri, MODEL_NAME)
        for k, v in tags.items():
            client.set_model_version_tag(MODEL_NAME, mv.version, k, v)
        client.set_model_version_tag(MODEL_NAME, mv.version, "threshold", str(thr))
        client.set_registered_model_alias(MODEL_NAME, "challenger", mv.version)
        msg = f"registered {MODEL_NAME} v{mv.version} as challenger"
        if args.promote and beats:
            client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
            msg += "; promoted to champion (set MODEL_NAME=dzhynsa-xlmr in .env to serve it)"
        elif args.promote:
            msg += f"; NOT promoted: PR-AUC {test_scores['pr_auc']:.3f} <= {champ_pr_auc:.3f}"
        print(msg)

    print(
        json.dumps(
            {
                "run_id": r.info.run_id,
                "threshold": thr,
                "xlmr": test_scores,
                "champion": {"model": champ, "test_pr_auc": champ_pr_auc},
                "onnx_verify": check,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
