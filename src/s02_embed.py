"""Шаг 1a. Эмбеддинги всех предложений: каждый слой энкодера (тем же пулингом, что у модели) + финальный выход.

Для каждой пары (модель, язык) пишет CACHE/{model}_{lang}.npz:
  layers   (L+1, N, d) float16 — пулинг hidden_states[l] (l=0 — выход эмбеддинг-слоя), без нормировки
  final    (N, d)      float32 — выход модели как в SentenceTransformer.encode (пулинг последнего слоя
                                  + Normalize, если он есть); сверяется с encode на 256 фразах
  sentences (N,)
и results/embed_log.json с проверкой «пулинг последнего слоя == финальный выход».

Запуск: python s02_embed.py [model_key ...]
"""
import json
import sys
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Normalize, Pooling, Transformer

from common import CACHE, MODELS, RESULTS, SEED, all_sentences

BATCH = 64


@torch.no_grad()
def embed_all_layers(model: SentenceTransformer, texts: list[str]):
    mods = list(model)
    assert isinstance(mods[0], Transformer) and isinstance(mods[1], Pooling), mods
    assert all(isinstance(m, Normalize) for m in mods[2:]), mods  # других слоёв (Dense) быть не должно
    auto, pool = mods[0].auto_model, mods[1]
    order = np.argsort([len(t) for t in texts])  # сортировка по длине ускоряет батчи
    layers_out, final_out = [None] * len(texts), [None] * len(texts)
    for s in range(0, len(texts), BATCH):
        ids = order[s:s + BATCH]
        batch = [texts[i] for i in ids]
        feats = model.tokenize(batch)
        out = auto(**{k: v for k, v in feats.items() if k in ("input_ids", "attention_mask", "token_type_ids")},
                   output_hidden_states=True, return_dict=True)
        pooled = []
        for h in out.hidden_states:
            f = {"token_embeddings": h, "attention_mask": feats["attention_mask"]}
            pooled.append(pool(f)["sentence_embedding"].float().numpy())
        pooled = np.stack(pooled, 1)  # (B, L+1, d)
        fin = pooled[:, -1]
        if any(isinstance(m, Normalize) for m in mods[2:]):
            fin = fin / np.linalg.norm(fin, axis=1, keepdims=True)
        for j, i in enumerate(ids):
            layers_out[i] = pooled[j]
            final_out[i] = fin[j]
    # сверка с «настоящим» SentenceTransformer.encode на 256 фразах: final должен совпасть
    chk = np.random.default_rng(SEED).choice(len(texts), size=min(256, len(texts)), replace=False)
    # prompt="" — не дать encode() добавить default prompt модели повторно (у USER-base он задан в конфиге)
    ref = model.encode([texts[i] for i in chk], batch_size=BATCH, convert_to_numpy=True, normalize_embeddings=False,
                       prompt="")
    got = np.stack([final_out[i] for i in chk])
    max_abs = float(np.max(np.abs(ref - got)))
    return np.stack(layers_out, 1), np.stack(final_out), max_abs  # (L+1, N, d), (N, d)


def main(keys):
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    CACHE.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS / "embed_log.json"
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {}
    for key in keys:
        name, langs, prefix = MODELS[key]
        model = SentenceTransformer(name, device="cpu")
        model.eval()
        for lang in langs:
            texts = all_sentences(lang)
            t0 = time.time()
            layers, final, max_abs = embed_all_layers(model, [prefix + t for t in texts])
            dt = time.time() - t0
            last = layers[-1] / np.linalg.norm(layers[-1], axis=1, keepdims=True)
            fin = final / np.linalg.norm(final, axis=1, keepdims=True)
            check = float(np.min(np.sum(last * fin, 1)))
            np.savez(CACHE / f"{key}_{lang}.npz", layers=layers.astype(np.float16),
                     final=final.astype(np.float32), sentences=np.array(texts))
            log[f"{key}_{lang}"] = dict(
                model=name, revision=getattr(model[0].auto_model.config, "_commit_hash", None),
                pooling=model[1].get_pooling_mode_str(), normalize=any(isinstance(m, Normalize) for m in model),
                prefix=prefix, n_sentences=len(texts), n_layers=layers.shape[0] - 1, dim=layers.shape[2],
                min_cos_lastlayer_vs_final=round(check, 6), max_abs_diff_vs_encode=max_abs, seconds=round(dt, 1),
            )
            print(key, lang, log[f"{key}_{lang}"], flush=True)
            log_path.write_text(json.dumps(log, indent=1, ensure_ascii=False), encoding="utf-8")
        del model


if __name__ == "__main__":
    main(sys.argv[1:] or list(MODELS))
