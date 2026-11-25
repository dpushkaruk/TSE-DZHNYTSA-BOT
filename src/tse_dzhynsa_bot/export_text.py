"""Export the train/val/test text split for fine-tuning XLM-R on Kaggle.

    uv run python -m tse_dzhynsa_bot.export_text

Writes data/xlmr/{train,val,test}.parquet (channel, id, day, text_clean, label_ad) and
manifest.json with the data version, so the Kaggle run is tied to one DVC snapshot.
Upload the folder as a private Kaggle dataset named `dzhynsa-text`.
"""

import argparse
import json
from pathlib import Path

from tse_dzhynsa_bot.train import ROOT, SPLIT, data_version, git_commit, load_dataset, split

COLUMNS = ["channel", "id", "day", "text_clean", "label_ad"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "xlmr")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"data_version": data_version(), "git_commit": git_commit(), "split": {}}
    for name, part in split(load_dataset(args.raw)).items():
        part = part[COLUMNS].assign(day=lambda d: d["day"].astype(str))
        part.to_parquet(args.out / f"{name}.parquet", index=False)
        manifest["split"][name] = {
            "range": [str(d) for d in SPLIT[name]],
            "rows": len(part),
            "ads": int(part["label_ad"].sum()),
        }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.out / "dataset-metadata.json").write_text(
        json.dumps({"title": "dzhynsa-text", "id": "KAGGLE_USER/dzhynsa-text", "licenses": []})
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
