"""Шаг 1 (извлечение v_neg, DiffMean) и Шаг 2 (устойчивость, H1) — для всех моделей и слоёв.

Считает то же самое и для контрольных направлений (edit, ant, rp): они нужны в H2, — и для
сбалансированных по тональности negB/editB/antB (разбиения на половины для них стратифицированы).

Выход:
  results/h1_stability.csv       — модель, язык, слой, вид направления, все метрики устойчивости
  results/h1_crossfit.csv        — align / nearest / probe AUC (кросс-фит по 5 фолдам) + вне шаблонов:
                                   natural_auc_{kind} — x·v отделяет предложения STS-B с отрицанием от прочих;
                                   h4set_auc_{kind}   — то же на предложениях отложенного набора H4
  results/h1_fold_matrices.json  — матрицы 5x5 для final (heatmap), процедура ТЗ и disjoint
  artifacts/v_neg/{model}_{lang}.npz — v_neg, v_negB, v_edit, v_ant, ... по всем слоям + final
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import ARTIFACTS, RESULTS, SEED, Emb, load_data, model_langs
from directions import (KINDS, PairSet, base_of, crossfit, fold_matrix, group_folds, half_weights,
                        half_weights_strat, is_bal, offdiag_mean, signflip, snr_bootstrap_ci, snr_reliability,
                        split_half, spearman_brown)

R_SPLITS, N_NULL, N_BOOT, R_BOOT = 200, 200, 300, 20


def layer_stats(ps: PairSet, fold, splits, rng):
    """splits: {"plain": (WA, WB), "strat": (WA, WB)} — стратифицированные по тональности для *B-видов."""
    rows = []
    G = ps.G
    for kind in KINDS:
        S = ps.Sk(kind)
        bal = is_bal(kind)
        cls = ps.g_valence if bal else None
        A0, B0 = splits["strat" if bal else "plain"]
        WA, WB = ps.W(kind, A0), ps.W(kind, B0)
        sh_each = np.sum(_u(WA @ S) * _u(WB @ S), 1)
        C_ovl = fold_matrix(S, fold, cls=cls)
        C_dis = fold_matrix(S, fold, disjoint=True, cls=cls)
        null_sh, null_ovl, null_dis = [], [], []
        for _ in range(N_NULL):
            Sn = signflip(S, rng)
            null_sh.append(split_half(Sn, WA[:50], WB[:50]))
            null_ovl.append(offdiag_mean(fold_matrix(Sn, fold, cls=cls)))
            null_dis.append(offdiag_mean(fold_matrix(Sn, fold, disjoint=True, cls=cls)))
        boot_sh, boot_ovl = [], []
        for _ in range(N_BOOT):
            counts = np.bincount(rng.integers(0, G, G), minlength=G).astype(float)
            A, B = half_weights(G, R_BOOT, rng, counts)
            boot_sh.append(split_half(S, ps.W(kind, A), ps.W(kind, B)))
            present = np.flatnonzero(counts > 0)
            f = np.full(G, -1)
            f[rng.permutation(present)] = np.arange(len(present)) % 5
            W = ps.W(kind, np.stack([(f != i) & (f >= 0) for i in range(5)]) * counts)
            V = _u(W @ S)
            boot_ovl.append(offdiag_mean(V @ V.T))
        sh = float(np.mean(sh_each))
        if bal:  # моментная оценка выведена для равных весов групп — для *B не считаем
            snr = a = b = snr_lo = snr_hi = np.nan
        else:
            snr, a, b = snr_reliability(S)
            _, snr_lo, snr_hi = snr_bootstrap_ci(S, rng, N_BOOT)
        w_all = ps.W(kind, np.ones(G))[0]
        rows.append(dict(
            kind=kind,
            split_half=sh, split_half_p05=float(np.percentile(sh_each, 5)),
            # основная оценка и ДИ: аналитическая split-half через моменты (корректный бутстрап)
            snr_half=snr, snr_half_lo=snr_lo, snr_half_hi=snr_hi, snr_signal=a, snr_noise=b,
            split_half_null_mean=float(np.mean(null_sh)), split_half_null_p99=float(np.percentile(null_sh, 99)),
            # наивный бутстрап split-half (копии утверждения в одной половине) — смещён вниз, для справки
            split_half_boot_lo=float(np.percentile(boot_sh, 2.5)), split_half_boot_hi=float(np.percentile(boot_sh, 97.5)),
            spearman_brown=spearman_brown(sh),
            tz5fold=offdiag_mean(C_ovl), tz5fold_null_mean=float(np.mean(null_ovl)),
            tz5fold_null_p99=float(np.percentile(null_ovl, 99)),
            tz5fold_boot_lo=float(np.percentile(boot_ovl, 2.5)), tz5fold_boot_hi=float(np.percentile(boot_ovl, 97.5)),
            disj5fold=offdiag_mean(C_dis), disj5fold_null_p99=float(np.percentile(null_dis, 99)),
            mean_diff_norm=float(np.linalg.norm(w_all @ S) / (5 * w_all.sum())),
            mean_pair_dist=float(np.mean(np.linalg.norm(ps.D[base_of(kind)], axis=1))),
        ))
    return rows


def _u(A):
    return A / np.linalg.norm(A, axis=-1, keepdims=True)


def main():
    rows, cf_rows, mats = [], [], {}
    (ARTIFACTS / "v_neg").mkdir(parents=True, exist_ok=True)
    for lang in ("en", "ru"):
        data = load_data(lang)
        pairs = data["pairs"]
        nat_texts = [r["text"] for r in data["natural"]]
        nat_y = np.array([r["has_neg"] for r in data["natural"]])
        h4_texts = [r[k] for r in data["h4"] for k in ("anchor", "para", "ant", "neg")]
        h4_y = np.array([k == "neg" for r in data["h4"] for k in ("anchor", "para", "ant", "neg")])
        G = 1 + max(r["base_id"] for r in pairs)
        split_rng = np.random.default_rng(SEED)
        WA, WB = half_weights(G, R_SPLITS, split_rng)  # одни и те же разбиения для всех моделей/слоёв
        g_val = np.array([next(r["valence"] for r in pairs if r["base_id"] == g) for g in range(G)])
        splits = {"plain": (WA, WB), "strat": half_weights_strat(g_val, R_SPLITS, np.random.default_rng(SEED + 2))}
        fold = group_folds(G, 5, np.random.default_rng(SEED + 1))
        for model, l in model_langs():
            if l != lang:
                continue
            emb = Emb(model, lang)
            dirs = {k: [] for k in KINDS}
            for layer in emb.layer_keys():
                ps = PairSet(emb, pairs, layer)
                rng = np.random.default_rng([SEED, len(rows)])
                for r in layer_stats(ps, fold, splits, rng):
                    rows.append(dict(model=model, lang=lang, layer=str(layer), n_layers=emb.n_layers, **r))
                cf = crossfit(ps, fold)
                Xn, Xh = emb.get(nat_texts, layer), emb.get(h4_texts, layer)
                for kind in ("neg", "negB", "edit", "ant"):
                    v = ps.direction(kind)
                    cf[f"natural_auc_{kind}"] = float(roc_auc_score(nat_y, Xn @ v))
                    cf[f"h4set_auc_{kind}"] = float(roc_auc_score(h4_y, Xh @ v))
                cf_rows.append(dict(model=model, lang=lang, layer=str(layer), n_layers=emb.n_layers,
                                    **{k: v for k, v in cf.items() if not isinstance(v, dict)},
                                    **{f"align_{k}": v for k, v in cf["align"].items()},
                                    **{f"align_sq_{k}": v for k, v in cf["align_sq"].items()}))
                for k in KINDS:
                    dirs[k].append(ps.direction(k))
                if layer == "final":
                    mats[f"{model}_{lang}"] = {
                        k: dict(tz=fold_matrix(ps.Sk(k), fold, cls=ps.g_valence if is_bal(k) else None).round(4).tolist(),
                                disjoint=fold_matrix(ps.Sk(k), fold, disjoint=True,
                                                     cls=ps.g_valence if is_bal(k) else None).round(4).tolist())
                        for k in KINDS}
            np.savez(ARTIFACTS / "v_neg" / f"{model}_{lang}.npz",
                     layers=np.array([str(x) for x in emb.layer_keys()]),
                     **{f"v_{k}": np.stack(v).astype(np.float32) for k, v in dirs.items()})
            fin = [r for r in rows if r["model"] == model and r["lang"] == lang and r["layer"] == "final"]
            print(model, lang, " | ".join(f"{r['kind']}: sh={r['split_half']:.3f} "
                                          f"snr={r['snr_half']:.3f}[{r['snr_half_lo']:.3f},{r['snr_half_hi']:.3f}] "
                                          f"null99={r['split_half_null_p99']:.3f}" for r in fin), flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "h1_stability.csv", index=False)
    pd.DataFrame(cf_rows).to_csv(RESULTS / "h1_crossfit.csv", index=False)
    (RESULTS / "h1_fold_matrices.json").write_text(json.dumps(mats), encoding="utf-8")


if __name__ == "__main__":
    main()
