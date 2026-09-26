"""Шаг 6. Межъязыковое и межмодельное сравнение направлений (H5).
(Послойные кривые устойчивости строятся из results/h1_*.csv и h2_*.csv в s08_figures.py.)

A) Внутри одной мультиязычной модели (me5, mminilm, bgem3), по всем слоям:
   cos(v_kind^EN, v_kind'^RU) для kind, kind' ∈ {neg, edit, ant}. Выравнивание не нужно: пространство общее.
   Поправка на шум оценки: cos / sqrt(SB_en * SB_ru), SB — Spearman–Brown из split-half (s03).

B) Между моделями одного языка (final): линейное отображение A -> B (ridge), обученное ТОЛЬКО на
   предложениях без отрицания (естественный корпус без отрицаний + корпус тем). Затем
   cos(unit(v_kind^A W), v_kind'^B) — переносится ли негация как негация, лучше ли, чем контроли.
   И per-pair: cos(d_i^A W, d_i^B) — насколько отображение сохраняет сами сдвиги пар каждого вида.

Выход: results/h5_crosslingual.csv, results/h5_crossmodel.csv, results/h5_crossmodel_pairs.csv
"""
import itertools

import numpy as np
import pandas as pd

from common import MODELS, RESULTS, SEED, Emb, load_data, unit
from directions import PairSet, spearman_brown

KINDS3 = ("neg", "negB", "edit", "ant")
ALPHAS = (1e-3, 1e-2, 1e-1, 1.0)


def ridge_fit(X, Y, rng):
    """Ridge с центрированием; α (в долях tr(XᵀX)/d) по 5-fold CV на среднем cos(pred, true)."""
    n = len(X)
    fold = rng.permutation(n) % 5
    best, best_a = -1, None
    for a in ALPHAS:
        sc = []
        for k in range(5):
            tr, te = fold != k, fold == k
            W, mx, my = _ridge(X[tr], Y[tr], a)
            pred = (X[te] - mx) @ W + my
            sc.append(np.mean(np.sum(unit(pred) * unit(Y[te]), 1)))
        if np.mean(sc) > best:
            best, best_a = float(np.mean(sc)), a
    W, mx, my = _ridge(X, Y, best_a)
    return W, mx, my, best_a, best


def _ridge(X, Y, a):
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    G = Xc.T @ Xc
    lam = a * np.trace(G) / G.shape[0]
    W = np.linalg.solve(G + lam * np.eye(G.shape[0]), Xc.T @ Yc)
    return W, mx, my


def main():
    h1 = pd.read_csv(RESULTS / "h1_stability.csv")
    sb = {(r.model, r.lang, str(r.layer), r.kind): r.spearman_brown for r in h1.itertuples()}

    # ---------------- A) EN vs RU внутри одной модели
    rows = []
    pairs = {l: load_data(l)["pairs"] for l in ("en", "ru")}
    for model in [m for m, (_, langs, _) in MODELS.items() if set(langs) == {"en", "ru"}]:
        e_en, e_ru = Emb(model, "en"), Emb(model, "ru")
        for layer in e_en.layer_keys():
            p_en, p_ru = PairSet(e_en, pairs["en"], layer), PairSet(e_ru, pairs["ru"], layer)
            for k1, k2 in itertools.product(KINDS3, KINDS3):
                c = float(p_en.direction(k1) @ p_ru.direction(k2))
                s1, s2 = sb[(model, "en", str(layer), k1)], sb[(model, "ru", str(layer), k2)]
                rows.append(dict(model=model, layer=str(layer), n_layers=e_en.n_layers, kind_en=k1, kind_ru=k2,
                                 cos=c, cos_corrected=c / np.sqrt(s1 * s2) if s1 > 0 and s2 > 0 else np.nan))
        f = {(r["kind_en"], r["kind_ru"]): r["cos"] for r in rows if r["model"] == model and r["layer"] == "final"}
        print("crossling", model, {f"{a}-{b}": round(v, 3) for (a, b), v in f.items()}, flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "h5_crosslingual.csv", index=False)

    # ---------------- B) между моделями (final)
    rows, prow = [], []
    for lang in ("en", "ru"):
        data = load_data(lang)
        fit_texts = [r["text"] for r in data["natural"] if not r["has_neg"]] + [r["text"] for r in data["topics"]]
        models = [m for m, (_, langs, _) in MODELS.items() if lang in langs]
        embs = {m: Emb(m, lang) for m in models}
        pss = {m: PairSet(embs[m], data["pairs"], "final") for m in models}
        for a, b in itertools.permutations(models, 2):
            X, Y = embs[a].get(fit_texts), embs[b].get(fit_texts)
            W, mx, my, alpha, cvcos = ridge_fit(X, Y, np.random.default_rng(SEED))
            # качество отображения на предложениях основного набора (в обучении их не было)
            Xp, Yp = pss[a].Xs["pos"], pss[b].Xs["pos"]
            map_cos_pos = float(np.mean(np.sum(unit((Xp - mx) @ W + my) * unit(Yp), 1)))
            for k1, k2 in itertools.product(KINDS3, KINDS3):
                t = unit(pss[a].direction(k1) @ W)
                rows.append(dict(lang=lang, src=a, dst=b, kind_src=k1, kind_dst=k2,
                                 cos=float(t @ pss[b].direction(k2)), alpha=alpha, cv_cos=cvcos,
                                 map_cos_pos=map_cos_pos))
            for k in ("neg", "edit", "ant"):
                dm = pss[a].D[k] @ W
                prow.append(dict(lang=lang, src=a, dst=b, kind=k,
                                 pair_cos=float(np.mean(np.sum(unit(dm) * unit(pss[b].D[k]), 1)))))
        print("crossmodel", lang, "done", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "h5_crossmodel.csv", index=False)
    pd.DataFrame(prow).to_csv(RESULTS / "h5_crossmodel_pairs.csv", index=False)


if __name__ == "__main__":
    main()
