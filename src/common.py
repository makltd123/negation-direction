"""Общие константы, загрузка данных и эмбеддингов."""
import json
import os
from pathlib import Path

import numpy as np

from lexicon import TOPICS as TOPICS_ORDER  # noqa: F401  (порядок тем)

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIG = RESULTS / "figures"
ARTIFACTS = ROOT / "artifacts"


def _cache_dir() -> Path:
    if os.environ.get("NEGDIR_CACHE"):
        return Path(os.environ["NEGDIR_CACHE"])
    d = Path("D:/ml-cache/negation_direction")  # вне OneDrive: кэш эмбеддингов ~1 ГБ
    return d if d.parent.exists() else ROOT / "cache"


CACHE = _cache_dir()

# key: (HF id, языки, префикс для симметричных задач)
MODELS = {
    "mpnet":   ("sentence-transformers/all-mpnet-base-v2", ("en",), ""),
    "bge":     ("BAAI/bge-base-en-v1.5", ("en",), ""),
    "e5":      ("intfloat/e5-base-v2", ("en",), "query: "),
    "me5":     ("intfloat/multilingual-e5-base", ("en", "ru"), "query: "),
    "mminilm": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", ("en", "ru"), ""),
    "bgem3":   ("BAAI/bge-m3", ("en", "ru"), ""),
    "rubert":  ("cointegrated/rubert-tiny2", ("ru",), ""),
    "user":    ("deepvk/USER-base", ("ru",), "query: "),
}
MODEL_LABEL = {
    "mpnet": "all-mpnet-base-v2", "bge": "bge-base-en-v1.5", "e5": "e5-base-v2",
    "me5": "multilingual-e5-base", "mminilm": "paraphrase-multilingual-MiniLM-L12-v2",
    "bgem3": "bge-m3", "rubert": "rubert-tiny2", "user": "USER-base",
}
LANGS = ("en", "ru")


def model_langs():
    return [(m, l) for m, (_, langs, _) in MODELS.items() for l in langs]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_data(lang):
    return dict(
        pairs=read_jsonl(DATA / f"pairs_{lang}.jsonl"),
        topics=read_jsonl(DATA / f"topics_{lang}.jsonl"),
        h4=read_jsonl(DATA / f"h4_{lang}.jsonl"),
        h4_triples=read_jsonl(DATA / f"h4_triples_{lang}.jsonl"),
        natural=read_jsonl(DATA / f"natural_{lang}.jsonl"),
    )


def all_sentences(lang):
    """Детерминированный список уникальных предложений языка (порядок = порядок в кэше)."""
    d = load_data(lang)
    seen, out = set(), []

    def add(s):
        if s not in seen:
            seen.add(s)
            out.append(s)

    for r in d["pairs"]:
        for k in ("pos", "neg", "edit", "ant"):
            add(r[k])
    for r in d["topics"]:
        add(r["text"])
    for r in d["h4"]:
        for k in ("anchor", "para", "neg", "ant", "neg_ant"):
            add(r[k])
    for r in d["natural"]:
        add(r["text"])
    return out


class Emb:
    """Эмбеддинги одной модели на одном языке: layers[l] (N, d) и final (N, d), всё L2-нормировано."""

    def __init__(self, model, lang, normalize=True):
        z = np.load(CACHE / f"{model}_{lang}.npz", allow_pickle=False)
        self.sentences = list(z["sentences"])
        assert self.sentences == all_sentences(lang), "кэш устарел: перезапустите s02_embed.py"
        self.idx = {s: i for i, s in enumerate(self.sentences)}
        layers = z["layers"].astype(np.float32)
        final = z["final"].astype(np.float32)
        if normalize:
            layers /= np.linalg.norm(layers, axis=-1, keepdims=True)
            final /= np.linalg.norm(final, axis=-1, keepdims=True)
        self.layers, self.final = layers, final
        self.n_layers = layers.shape[0] - 1  # слой 0 = выход эмбеддинг-слоя
        # у CLS-пулинга слой 0 — эмбеддинг токена [CLS] без контекста: одинаков для всех предложений
        self.degenerate0 = bool(np.max(np.std(layers[0], axis=0)) < 1e-4)
        self.model, self.lang = model, lang

    def rep(self, layer):
        """layer: int (0..L) или 'final'."""
        return self.final if layer == "final" else self.layers[layer]

    def get(self, texts, layer="final"):
        X = self.rep(layer)
        return X[[self.idx[t] for t in texts]]

    def layer_keys(self):
        first = 1 if self.degenerate0 else 0
        return list(range(first, self.n_layers + 1)) + ["final"]


def unit(v, axis=-1):
    return v / np.linalg.norm(v, axis=axis, keepdims=True)
