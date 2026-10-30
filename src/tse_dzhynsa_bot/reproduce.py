"""Rebuild a registered model version from its lineage tags and check the metric.

    uv run python -m tse_dzhynsa_bot.reproduce --alias champion

Checks out the commit in a git worktree, pulls that data version with DVC,
retrains, and fails if test PR-AUC differs from the registered run by more than
the tolerance (README states it).
"""

import argparse
import json
import subprocess
import sys

from mlflow import MlflowClient

from tse_dzhynsa_bot.train import MODEL_NAME, ROOT, data_version

TOLERANCE = 0.01


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alias", default="champion")
    ap.add_argument("--version", help="model version; overrides --alias")
    args = ap.parse_args()

    client = MlflowClient()
    mv = (
        client.get_model_version(MODEL_NAME, args.version)
        if args.version
        else client.get_model_version_by_alias(MODEL_NAME, args.alias)
    )
    commit, data_md5 = mv.tags["git_commit"].removesuffix("-dirty"), mv.tags["data_version"]
    expected = client.get_run(mv.run_id).data.metrics["test_pr_auc"]
    print(
        f"{MODEL_NAME} v{mv.version}: commit {commit[:8]}, data {data_md5}, pr_auc {expected:.4f}"
    )

    wt = ROOT / ".repro" / commit[:12]
    if not wt.exists():
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(wt), commit], cwd=ROOT, check=True
        )
    subprocess.run(["dvc", "pull", "data/raw.dvc"], cwd=wt, check=True)
    if data_version(wt) != data_md5:
        sys.exit(f"data version mismatch: {data_version(wt)} != {data_md5}")
    out = (
        subprocess.run(
            ["uv", "run", "python", "-m", "tse_dzhynsa_bot.train", "--reproduces", mv.run_id],
            cwd=wt,
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .splitlines()[-1]
    )
    got = json.loads(out)["tfidf_logreg"]["pr_auc"]
    diff = abs(got - expected)
    print(
        f"reproduced pr_auc {got:.4f}, registered {expected:.4f}, diff {diff:.4f}, tol {TOLERANCE}"
    )
    sys.exit(0 if diff <= TOLERANCE else 1)


if __name__ == "__main__":
    main()
