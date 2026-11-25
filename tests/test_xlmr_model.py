import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers

from tse_dzhynsa_bot.xlmr_model import HEAD, encode, softmax_pos


def tiny_tokenizer() -> Tokenizer:
    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3} | {f"w{i}": 4 + i for i in range(400)}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    return tk


def test_encode_pads_and_masks():
    ids, mask = encode(tiny_tokenizer(), ["w1 w2 w3", "w4"])
    assert ids.tolist() == [[0, 5, 6, 7, 2], [0, 8, 2, 1, 1]]
    assert mask.tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]


def test_encode_keeps_head_and_tail_of_long_posts():
    text = " ".join(f"w{i}" for i in range(300))
    ids, mask = encode(tiny_tokenizer(), [text], max_len=256)
    assert ids.shape == (1, 256) and mask.all()
    assert ids[0, 1] == 4 and ids[0, HEAD] == 4 + HEAD - 1  # first HEAD tokens
    assert ids[0, -2] == 4 + 299 and ids[0, -1] == 2  # last token, then </s>


def test_softmax_pos():
    p = softmax_pos(np.array([[0.0, 0.0], [0.0, 10.0]]))
    assert np.allclose(p, [0.5, 1 / (1 + np.exp(-10))])
