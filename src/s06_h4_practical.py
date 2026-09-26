"""Шаг 5. Практическая польза v_neg (H4) на отложенных тройках (anchor, positive, negative).

Провал = score(anchor, negative) >= score(anchor, positive). Все вектора — final (как в поиске).
v_neg и v_edit берутся из основного набора пар (s03), тройки H4 в нём не участвовали.

Методы:
  M0_raw          — косинус сырых эмбеддингов (baseline)
  M1_projout      — e' = e − (e·v_neg)v_neg, затем косинус
  M1B_projout     — то же с v_negB (сбалансированное по тональности направление, см. RESEARCH_LOG §6)
  M1_ctrl_edit    — то же, но вычитается v_edit (контроль: помогает ли вычитание ЛЮБОГО направления)
  M1_ctrl_random  — то же для 50 случайных направлений (средняя доля провалов)
  M2_feature      — cos(a,c) − λ·|a·v_neg − c·v_neg|, λ подбирается на train-фолдах (5 фолдов по пунктам)
                    по макро-среднему T1–T4
  M2_feature_T1   — то же, но λ подбирается только по T1 («хак»: оптимизация под один тип ловушки)
  M2B_feature     — M2 с v_negB
  M2_ctrl_edit    — M2 с v_edit вместо v_neg
  M5_lexical      — контроль без эмбеддингов: cos(a,c) − λ·|neg(a) − neg(c)|, neg() = регулярка «есть
                    ли отрицание» (not/n't/no/never… ; не/нет/ни…); λ по CV так же, как в M2
  M5_lexical_T1   — то же, λ только по T1
  M3_center       — вычесть среднее естественного корпуса (STS-B), косинус
  M4_whiten       — mean-centering + whitening (PCA, k=256) по естественному корпусу

Выход: results/h4_methods.csv (доли провалов по типам + McNemar против M0), results/h4_lambda.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from common import RESULTS, SEED, Emb, load_data, model_langs, unit
from directions import PairSet
from s01_build_data import NEG_RE

TYPES = ("T1_neg", "T2_ant", "T3_negpara", "T4_neganchor")
LAMBDAS = np.round(np.arange(0, 5.001, 0.05), 3)
K_WHITEN = 256


def cos_triples(A, P, Nn):
    A, P, Nn = unit(A), unit(P), unit(Nn)
    return np.sum(A * P, 1), np.sum(A * Nn, 1)


def projout(X, v):
    return X - np.outer(X @ v, v)


def fail_vec(sp, sn):
    return sn >= sp


def feature_scores(A, P, Nn, v, lam):
    sp, sn = cos_triples(A, P, Nn)
    if isinstance(v, tuple):  # лексический контроль: v = (neg(a), neg(p), neg(n)) из регулярки
        pa, pp, pn = v
    else:
        pa, pp, pn = unit(A) @ v, unit(P) @ v, unit(Nn) @ v
    return sp - lam * np.abs(pa - pp), sn - lam * np.abs(pa - pn)


def macro_acc(fails, types, weights_types=TYPES):
    return np.mean([1 - fails[types == t].mean() for t in weights_types if np.any(types == t)])


def cv_feature(A, P, Nn, v, types, items, objective_types, rng):
    """5 фолдов по пунктам: λ по train-фолдам, предсказания на test."""
    uniq = np.unique(items)
    fold_of = dict(zip(rng.permutation(uniq), np.arange(len(uniq)) % 5))
    fold = np.array([fold_of[i] for i in items])
    pre = {lam: fail_vec(*feature_scores(A, P, Nn, v, lam)) for lam in LAMBDAS}
    fails = np.zeros(len(items), dtype=bool)
    chosen = []
    for k in range(5):
        tr, te = fold != k, fold == k
        accs = [macro_acc(pre[lam][tr], types[tr], objective_types) for lam in LAMBDAS]
        best = LAMBDAS[int(np.argmax(accs))]  # при равенстве — наименьшая λ
        chosen.append(best)
        fails[te] = pre[best][te]
    return fails, chosen


def whitening(Ref, k):
    mu = Ref.mean(0)
    U, s, Vt = np.linalg.svd(Ref - mu, full_matrices=False)
    k = min(k, Vt.shape[0])
    W = Vt[:k].T / (s[:k] / np.sqrt(len(Ref) - 1))
    return mu, W


def mcnemar(f0, f1):
    """Точный тест на несовпадающих парах: b = провал только у baseline, c = только у метода."""
    b = int(np.sum(f0 & ~f1))
    c = int(np.sum(~f0 & f1))
    p = binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0
    return b, c, float(p)


def main():
    rows, lam_rows = [], []
    for lang in ("en", "ru"):
        data = load_data(lang)
        tr = data["h4_triples"]
        types = np.array([t["type"] for t in tr])
        items = np.array([t["item_id"] for t in tr])
        nat = [r["text"] for r in data["natural"]]
        for model, l in model_langs():
            if l != lang:
                continue
            emb = Emb(model, lang)
            A = emb.get([t["anchor"] for t in tr])
            P = emb.get([t["positive"] for t in tr])
            Nn = emb.get([t["negative"] for t in tr])
            ps = PairSet(emb, data["pairs"], "final")
            v_neg, v_edit, v_negB = ps.direction("neg"), ps.direction("edit"), ps.direction("negB")
            Ref = emb.get(nat)
            rng = np.random.default_rng([SEED, 5])
            lex = tuple(np.array([bool(NEG_RE[lang].search(t[k])) for t in tr], dtype=float)
                        for k in ("anchor", "positive", "negative"))
            res = {}
            res["M0_raw"] = fail_vec(*cos_triples(A, P, Nn))
            res["M1_projout"] = fail_vec(*cos_triples(projout(A, v_neg), projout(P, v_neg), projout(Nn, v_neg)))
            res["M1B_projout"] = fail_vec(*cos_triples(projout(A, v_negB), projout(P, v_negB), projout(Nn, v_negB)))
            res["M1_ctrl_edit"] = fail_vec(*cos_triples(projout(A, v_edit), projout(P, v_edit), projout(Nn, v_edit)))
            rand_fails = []
            for _ in range(50):
                r = unit(rng.standard_normal(A.shape[1]))
                rand_fails.append(fail_vec(*cos_triples(projout(A, r), projout(P, r), projout(Nn, r))))
            res["M1_ctrl_random"] = np.mean(rand_fails, 0)  # доля провалов, усреднённая по направлениям
            for name, v, obj in (("M2_feature", v_neg, TYPES), ("M2_feature_T1", v_neg, ("T1_neg",)),
                                 ("M2B_feature", v_negB, TYPES), ("M2_ctrl_edit", v_edit, TYPES),
                                 ("M5_lexical", lex, TYPES), ("M5_lexical_T1", lex, ("T1_neg",))):
                f, chosen = cv_feature(A, P, Nn, v, types, items, obj, np.random.default_rng([SEED, 9]))
                res[name] = f
                lam_rows.append(dict(model=model, lang=lang, method=name, lambdas=" ".join(map(str, chosen))))
            mu = Ref.mean(0)
            res["M3_center"] = fail_vec(*cos_triples(A - mu, P - mu, Nn - mu))
            mu, W = whitening(Ref, K_WHITEN)
            res["M4_whiten"] = fail_vec(*cos_triples((A - mu) @ W, (P - mu) @ W, (Nn - mu) @ W))

            f0 = res["M0_raw"].astype(bool)
            for meth, f in res.items():
                for t in TYPES + ("ALL",):
                    m = np.ones(len(types), bool) if t == "ALL" else types == t
                    row = dict(model=model, lang=lang, method=meth, type=t, n=int(m.sum()),
                               fail_rate=float(np.mean(f[m])))
                    if meth != "M0_raw" and f.dtype == bool:
                        row["b_fixed"], row["c_broken"], row["p_mcnemar"] = mcnemar(f0[m], f[m])
                    rows.append(row)
                rows.append(dict(model=model, lang=lang, method=meth, type="MACRO", n=len(types),
                                 fail_rate=float(np.mean([np.mean(f[types == t]) for t in TYPES]))))
            summ = {m: f"{np.mean(res[m][types == 'T1_neg']):.2f}/{np.mean([np.mean(res[m][types == t]) for t in TYPES]):.2f}"
                    for m in res}
            print(model, lang, "T1/macro fail:", summ, flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "h4_methods.csv", index=False)
    pd.DataFrame(lam_rows).to_csv(RESULTS / "h4_lambda.csv", index=False)


if __name__ == "__main__":
    main()
