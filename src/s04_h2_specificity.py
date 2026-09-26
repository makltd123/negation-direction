"""Шаг 3. Специфичность v_neg относительно контрольных направлений (H2), по всем слоям.

Выход:
  results/h2_specificity.csv — пары (направление, контроль): neg–edit, neg–ant, neg–rp (по плану) и
      negB–editB, negB–antB, negB–rp (сбалансированные по тональности, добавлено после §6 журнала):
      self_a / self_b   — split-half устойчивость каждого направления
      cross             — cos(v_a по половине A, v_b по половине B), симметризовано
      disattenuated     — cross / sqrt(self_a * self_b): похожесть с поправкой на шум оценки
      cos_full          — cos полных оценок
      gap = self_a - cross и его бутстрап-ДИ (кластерный бутстрап по утверждениям)
  results/h2_valence.csv — по всем слоям: направления, посчитанные отдельно по позитивным / негативным /
      нейтральным утверждениям; нулевое распределение для pos-vs-neg (перестановки групп тех же размеров);
      cos с «осью тональности» v_sent = unit(v_ant[pos] − v_ant[neg]).
  results/h2_construction.csv — то же для типов конструкций (copula vs do-support / verb), final и середина.
"""
import numpy as np
import pandas as pd

from common import RESULTS, SEED, Emb, load_data, model_langs, unit
from directions import PairSet, half_weights, half_weights_strat, is_bal

R_SPLITS, N_BOOT, R_BOOT, N_PERM = 200, 300, 20, 200
PAIRS = (("neg", "edit"), ("neg", "ant"), ("neg", "rp"), ("negB", "editB"), ("negB", "antB"), ("negB", "rp"))


def _u(A):
    return A / np.linalg.norm(A, axis=-1, keepdims=True)


def mean_cos(A, B):
    return float(np.mean(np.sum(_u(A) * _u(B), -1)))


def pair_stats(ps, a, b, splits):
    """self_a, self_b, cross на одних и тех же разбиениях групп."""
    strat = is_bal(a) or is_bal(b)
    A0, B0 = splits["strat" if strat else "plain"]
    Sa, Sb = ps.Sk(a), ps.Sk(b)
    WaA, WaB, WbA, WbB = ps.W(a, A0), ps.W(a, B0), ps.W(b, A0), ps.W(b, B0)
    self_a = mean_cos(WaA @ Sa, WaB @ Sa)
    self_b = mean_cos(WbA @ Sb, WbB @ Sb)
    cross = 0.5 * (mean_cos(WaA @ Sa, WbB @ Sb) + mean_cos(WbA @ Sb, WaB @ Sa))
    return self_a, self_b, cross


def valence_rows(ps, rng):
    out = {}
    lab = ps.g_valence
    for kind in ("neg", "edit", "ant"):
        V = {c: ps.direction(kind, (lab == c).astype(float)) for c in ("pos", "neg", "neu")}
        out[f"{kind}_pos_vs_neg"] = float(V["pos"] @ V["neg"])
        out[f"{kind}_pos_vs_neu"] = float(V["pos"] @ V["neu"])
        out[f"{kind}_neg_vs_neu"] = float(V["neg"] @ V["neu"])
        idx = np.flatnonzero(np.isin(lab, ["pos", "neg"]))
        null = []
        for _ in range(N_PERM):
            perm = rng.permutation(lab[idx])
            wa, wb = np.zeros(ps.G), np.zeros(ps.G)
            wa[idx[perm == "pos"]] = 1
            wb[idx[perm == "neg"]] = 1
            null.append(float(ps.direction(kind, wa) @ ps.direction(kind, wb)))
        out[f"{kind}_pos_vs_neg_null_mean"] = float(np.mean(null))
        out[f"{kind}_pos_vs_neg_null_p05"] = float(np.percentile(null, 5))
    # отрицание против антонима внутри одного класса тональности
    for c in ("pos", "neg", "neu"):
        w = (lab == c).astype(float)
        out[f"neg_vs_ant_within_{c}"] = float(ps.direction("neg", w) @ ps.direction("ant", w))
    # ось тональности: антоним позитивного утверждения ведёт «в негатив», негативного — «в позитив»
    v_sent = unit(ps.direction("ant", (lab == "pos").astype(float)) - ps.direction("ant", (lab == "neg").astype(float)))
    for kind in ("neg", "negB", "edit", "ant", "antB"):
        out[f"cos_sent_{kind}"] = float(ps.direction(kind) @ v_sent)
    return out


def construction_rows(ps, a, b, rng):
    lab = ps.g_constr
    out = []
    for kind in ("neg", "negB", "edit", "ant"):
        wa, wb = (lab == a).astype(float), (lab == b).astype(float)
        obs = float(ps.direction(kind, wa) @ ps.direction(kind, wb))
        idx = np.flatnonzero(np.isin(lab, [a, b]))
        null = []
        for _ in range(N_PERM):
            perm = rng.permutation(lab[idx])
            pa, pb = np.zeros(ps.G), np.zeros(ps.G)
            pa[idx[perm == a]] = 1
            pb[idx[perm == b]] = 1
            null.append(float(ps.direction(kind, pa) @ ps.direction(kind, pb)))
        out.append(dict(a=a, b=b, n_a=int(wa.sum()), n_b=int(wb.sum()), kind=kind, cos=obs,
                        null_mean=float(np.mean(null)), null_p05=float(np.percentile(null, 5)),
                        pct_rank=float(np.mean(np.array(null) <= obs))))
    return out


def main():
    spec, val, con = [], [], []
    for lang in ("en", "ru"):
        pairs = load_data(lang)["pairs"]
        G = 1 + max(r["base_id"] for r in pairs)
        g_val = np.array([next(r["valence"] for r in pairs if r["base_id"] == g) for g in range(G)])
        splits = {"plain": half_weights(G, R_SPLITS, np.random.default_rng(SEED)),
                  "strat": half_weights_strat(g_val, R_SPLITS, np.random.default_rng(SEED + 2))}
        cb = ("copula", "do-support") if lang == "en" else ("copula", "verb")
        for model, l in model_langs():
            if l != lang:
                continue
            emb = Emb(model, lang)
            for layer in emb.layer_keys():
                ps = PairSet(emb, pairs, layer)
                rng = np.random.default_rng([SEED, 7, len(spec)])
                boot = {p: [] for p in PAIRS}
                for _ in range(N_BOOT):
                    counts = np.bincount(rng.integers(0, G, G), minlength=G).astype(float)
                    A, B = half_weights(G, R_BOOT, rng, counts)
                    bs = {"plain": (A, B), "strat": (A, B)}
                    for a, b in PAIRS:
                        sa, _, cr = pair_stats(ps, a, b, bs)
                        boot[(a, b)].append(sa - cr)
                for a, b in PAIRS:
                    sa, sb, cr = pair_stats(ps, a, b, splits)
                    dis = cr / np.sqrt(sa * sb) if sa > 0 and sb > 0 else np.nan
                    spec.append(dict(model=model, lang=lang, layer=str(layer), n_layers=emb.n_layers, a=a, b=b,
                                     self_a=sa, self_b=sb, cross=cr, disattenuated=dis,
                                     cos_full=float(ps.direction(a) @ ps.direction(b)), gap=sa - cr,
                                     gap_lo=float(np.percentile(boot[(a, b)], 2.5)),
                                     gap_hi=float(np.percentile(boot[(a, b)], 97.5))))
                val.append(dict(model=model, lang=lang, layer=str(layer), n_layers=emb.n_layers,
                                **valence_rows(ps, rng)))
                if layer in ("final", emb.n_layers // 2):
                    for r in construction_rows(ps, cb[0], cb[1], rng):
                        con.append(dict(model=model, lang=lang, layer=str(layer), **r))
            f = [r for r in spec if r["model"] == model and r["lang"] == lang and r["layer"] == "final"]
            v = val[-1]
            print(model, lang, " | ".join(f"{r['a']}-{r['b']}: cross={r['cross']:.3f} dis={r['disattenuated']:.3f} "
                                          f"gap=[{r['gap_lo']:.3f},{r['gap_hi']:.3f}]" for r in f),
                  f"|| valence neg pos-vs-neg={v['neg_pos_vs_neg']:.3f} (null {v['neg_pos_vs_neg_null_mean']:.3f}) "
                  f"cos_sent neg={v['cos_sent_neg']:.3f} negB={v['cos_sent_negB']:.3f}", flush=True)
    pd.DataFrame(spec).to_csv(RESULTS / "h2_specificity.csv", index=False)
    pd.DataFrame(val).to_csv(RESULTS / "h2_valence.csv", index=False)
    pd.DataFrame(con).to_csv(RESULTS / "h2_construction.csv", index=False)


if __name__ == "__main__":
    main()
