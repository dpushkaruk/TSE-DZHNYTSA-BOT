"""Serving side of the fine-tuned XLM-R classifier: ONNX int8 + tokenizer, no torch.

The Kaggle notebook (notebooks/xlmr_finetune_kaggle.ipynb) trains the model and exports
`model.onnx` and `tokenizer.json`; import_xlmr.py logs them to MLflow with this class.
`encode` is copied into the notebook as is, so training and serving cut texts the same way.
"""

import numpy as np
import pandas as pd

try:  # mlflow is a project dependency; the notebook imports this file without it
    from mlflow.pyfunc import PythonModel
except ImportError:  # pragma: no cover
    PythonModel = object

MAX_LEN = 256
HEAD = 128  # long posts: keep the first HEAD tokens and the end (links, promo codes)


def encode(tokenizer, texts: list[str], max_len: int = MAX_LEN) -> tuple[np.ndarray, np.ndarray]:
    """tokenizers.Tokenizer -> padded (input_ids, attention_mask), head + tail truncation."""
    cls, sep, pad = (tokenizer.token_to_id(t) for t in ("<s>", "</s>", "<pad>"))
    body = max_len - 2
    rows = []
    for enc in tokenizer.encode_batch([t or "" for t in texts], add_special_tokens=False):
        ids = enc.ids
        if len(ids) > body:
            ids = ids[:HEAD] + ids[-(body - HEAD) :]
        rows.append([cls, *ids, sep])
    width = max(len(r) for r in rows)
    input_ids = np.full((len(rows), width), pad, dtype=np.int64)
    mask = np.zeros((len(rows), width), dtype=np.int64)
    for i, r in enumerate(rows):
        input_ids[i, : len(r)] = r
        mask[i, : len(r)] = 1
    return input_ids, mask


def softmax_pos(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e[:, 1] / e.sum(axis=1)


class XlmrOnnxClassifier(PythonModel):
    """MLflow pyfunc: DataFrame with `text_clean` -> DataFrame with `prob_ad`."""

    batch_size = 16

    def load_context(self, context) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(context.artifacts["tokenizer"])
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            context.artifacts["onnx_model"], opts, providers=["CPUExecutionProvider"]
        )

    def __getstate__(self) -> dict:
        # MLflow calls load_context while inferring the signature, then pickles the
        # instance: the ONNX session is rebuilt from the artifacts on load, never pickled.
        return {k: v for k, v in self.__dict__.items() if k not in ("session", "tokenizer")}

    def predict_texts(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch_size):
            ids, mask = encode(self.tokenizer, texts[i : i + self.batch_size])
            (logits,) = self.session.run(["logits"], {"input_ids": ids, "attention_mask": mask})
            out.append(softmax_pos(logits))
        return np.concatenate(out) if out else np.zeros(0)

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        texts = model_input["text_clean"].fillna("").astype(str).tolist()
        return pd.DataFrame({"prob_ad": self.predict_texts(texts)}, index=model_input.index)
