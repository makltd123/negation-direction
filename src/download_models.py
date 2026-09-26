"""Скачивает все модели из common.MODELS в кэш HuggingFace (необязательно: s02_embed.py скачает их сам)."""
from sentence_transformers import SentenceTransformer

from common import MODELS

for key, (name, _, _) in MODELS.items():
    SentenceTransformer(name, device="cpu")
    print(key, name, "ok", flush=True)
