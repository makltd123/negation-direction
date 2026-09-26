"""Шаг 4. Ортогональность v_neg направлениям темы (H3), по всем слоям.

Подпространство тем: 8 средних векторов по темам из topics_{lang}.jsonl, центрированных ->
ортонормированный базис Q ранга 7 (оно же натянуто на все 28 направлений «тема A − тема B»).

Выход results/h3_topic.csv:
  frac_{kind}   — доля квадрата нормы направления в подпространстве тем (kind: neg, negB, edit, ant, rp)
  null_iso      — 7/d, ожидание для случайного изотропного направления
  null_aniso    — tr(QᵀΣQ)/tr(Σ), Σ — ковариация естественного корпуса (случайное направление «как данные»)
  max_abs_cos_topic_pair — max |cos(v_neg, v_topic(A,B))| по 28 парам тем
  topic_consistency_{kind} и его перцентиль среди случайных разбиений на 8 групп по 6
"""
import itertools

import numpy as np
import pandas as pd

from common import RESULTS, SEED, TOPICS_ORDER, Emb, load_data, model_langs, unit
from directions import PairSet, random_partition_consistency, subset_consistency

N_PERM = 300


def main():
    rows = []
    for lang in ("en", "ru"):
        data = load_data(lang)
        pairs, topics = data["pairs"], data["topics"]
        nat = [r["text"] for r in data["natural"]]
        for model, l in model_langs():
            if l != lang:
                continue
            emb = Emb(model, lang)
            for layer in emb.layer_keys():
                T = emb.get([r["text"] for r in topics], layer)
                lab = np.array([r["topic"] for r in topics])
                means = np.stack([T[lab == t].mean(0) for t in TOPICS_ORDER])
                Mc = means - means.mean(0)
                U, s, _ = np.linalg.svd(Mc.T, full_matrices=False)
                Q = U[:, : len(TOPICS_ORDER) - 1]  # (d, 7)
                d = Q.shape[0]
                N = emb.get(nat, layer)
                Nc = N - N.mean(0)
                null_aniso = float(np.sum((Nc @ Q) ** 2) / np.sum(Nc ** 2))
                ps = PairSet(emb, pairs, layer)
                rng = np.random.default_rng([SEED, 11, len(rows)])
                row = dict(model=model, lang=lang, layer=str(layer), n_layers=emb.n_layers, dim=d,
                           null_iso=(len(TOPICS_ORDER) - 1) / d, null_aniso=null_aniso)
                for kind in ("neg", "negB", "edit", "ant", "rp"):
                    v = ps.direction(kind)
                    row[f"frac_{kind}"] = float(np.sum((Q.T @ v) ** 2))
                vneg = ps.direction("neg")
                tdirs = [unit(means[i] - means[j]) for i, j in itertools.combinations(range(len(TOPICS_ORDER)), 2)]
                row["max_abs_cos_topic_pair"] = float(max(abs(t @ vneg) for t in tdirs))
                for kind in ("neg", "negB", "edit", "ant"):
                    obs, _, _ = subset_consistency(ps, kind, ps.g_topic)
                    null = random_partition_consistency(ps, kind, ps.g_topic, N_PERM, rng)
                    row[f"topic_consistency_{kind}"] = obs
                    row[f"topic_consistency_{kind}_null_mean"] = float(null.mean())
                    row[f"topic_consistency_{kind}_pct"] = float(np.mean(null <= obs))
                rows.append(row)
            f = rows[-1]
            print(model, lang, f"final: frac_neg={f['frac_neg']:.4f} iso={f['null_iso']:.4f} "
                               f"aniso={f['null_aniso']:.4f} frac_edit={f['frac_edit']:.4f} "
                               f"frac_rp={f['frac_rp']:.4f} topic_cons_neg={f['topic_consistency_neg']:.3f} "
                               f"(null {f['topic_consistency_neg_null_mean']:.3f}, pct {f['topic_consistency_neg_pct']:.2f})",
                  flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "h3_topic.csv", index=False)


if __name__ == "__main__":
    main()
