"""Шаг 7 (продолжение, «Что дальше» п.1). Признак M2 на внешнем бенчмарке отрицаний в поиске — NevIR.

NevIR (Weller et al., 2023, HF: orionweller/NevIR): в каждом пункте два запроса q1, q2 и два документа
doc1, doc2; doc1 и doc2 отличаются отрицанием, q1 релевантен doc1, q2 — doc2. Метрика бенчмарка —
pairwise accuracy: пункт засчитан, только если ОБА ранжирования верны,
  s(q1,doc1) > s(q1,doc2)  и  s(q2,doc2) > s(q2,doc1)
(случайное угадывание — 25%). Равенство — провал, как в H4. Дополнительно — доля верных запросов.

Отличия от H4: документы — абзацы, а не одно предложение; отрицания любых видов; данные чужие.
Направления берутся из artifacts/v_neg/{model}_en.npz (строка final) — на NevIR они не подбирались.

Методы (s(q,d) = q·d на L2-нормированных векторах):
  M0_raw          — косинус (baseline)
  M1_projout      — вычесть проекцию на v_neg, затем косинус
  M2_feature      — q·d − λ·|q·v_neg − d·v_neg|, λ по сетке на train+validation NevIR
  M2_feature_h4   — то же, но λ = медиана λ из H4 (results/h4_lambda.csv): перенос без подстройки
  M2B_feature     — M2 с v_negB (сбалансированное по тональности направление)
  M2B_feature_h4  — M2B с λ из H4
  M2_ctrl_edit    — M2 с v_edit (контроль: признак на «любом» направлении правки)
  M2_ctrl_random  — M2 с 20 случайными направлениями (λ подбирается для каждого), средняя точность
  M5_lexical      — q·d − λ·|neg(q) − neg(d)|, neg() — регулярка «есть ли отрицание» (как в H4)

Префиксы: e5 и multilingual-e5 — «query: » для запросов и «passage: » для документов (так их
используют в поиске; v_neg в основном исследовании считался с «query: », это расхождение
записано в results/nevir_log.json). Остальные модели — без префиксов.

Выход: results/nevir_methods.csv (точность на test + McNemar и бутстрап-ДИ разности против M0),
       results/nevir_lambda.csv (кривые точности по λ на dev и test — test только для описания),
       results/nevir_log.json (ревизия датасета, размеры, доля обрезанных документов).
Эмбеддинги кэшируются в CACHE/nevir_{model}.npz.

Запуск: python s10_nevir.py [model_key ...]      (по умолчанию все модели с английским)
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from common import ARTIFACTS, CACHE, MODELS, RESULTS, SEED, unit
from s01_build_data import NEG_RE

DATASET = "orionweller/NevIR"
FIELDS = ("q1", "q2", "doc1", "doc2")
LAMBDAS = np.round(np.arange(0, 5.001, 0.05), 3)
N_RANDOM = 20
N_BOOT = 10000
BATCH = 32
DOC_PREFIX = {"e5": "passage: ", "me5": "passage: "}  # для запросов — префикс из MODELS


def load_nevir():
    from datasets import load_dataset

    ds = load_dataset(DATASET)
    out = {}
    for split in ("train", "validation", "test"):
        rows = ds[split]
        missing = [f for f in FIELDS if f not in rows.column_names]
        assert not missing, f"в {DATASET}/{split} нет полей {missing}: {rows.column_names}"
        out[split] = [{f: r[f] for f in FIELDS} for r in rows]
    info = dict(dataset=DATASET, fingerprint={s: ds[s]._fingerprint for s in out},
                sizes={s: len(v) for s, v in out.items()})
    return out, info


def embed(key, texts_q, texts_d):
    """Эмбеддинги уникальных запросов и документов (L2-нормированы), с кэшем на диске."""
    path = CACHE / f"nevir_{key}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if list(z["queries"]) == texts_q and list(z["docs"]) == texts_d:
            return z["Q"], z["D"], json.loads(str(z["meta"]))
    import torch
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    name, _, qpre = MODELS[key]
    dpre = DOC_PREFIX.get(key, "")
    model = SentenceTransformer(name, device="cpu")
    model.eval()
    # prompt="" — не дать encode() добавить default prompt модели поверх нашего префикса (см. s02)
    enc = dict(batch_size=BATCH, convert_to_numpy=True, normalize_embeddings=True, prompt="")
    Q = model.encode([qpre + t for t in texts_q], **enc).astype(np.float32)
    D = model.encode([dpre + t for t in texts_d], **enc).astype(np.float32)
    n_tok = [len(ids) for ids in model.tokenizer([dpre + t for t in texts_d])["input_ids"]]
    meta = dict(model=name, query_prefix=qpre, doc_prefix=dpre, max_seq_length=model.max_seq_length,
                doc_tokens_median=int(np.median(n_tok)), doc_tokens_max=int(np.max(n_tok)),
                docs_truncated_frac=float(np.mean(np.array(n_tok) > model.max_seq_length)))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, Q=Q, D=D, queries=np.array(texts_q), docs=np.array(texts_d), meta=json.dumps(meta))
    return Q, D, meta


class Split:
    """Векторы одного сплита: q1, q2, d1, d2 (n, dim) и лексические флаги отрицания."""

    def __init__(self, rows, qix, dix, Q, D):
        self.n = len(rows)
        self.q1, self.q2 = Q[[qix[r["q1"]] for r in rows]], Q[[qix[r["q2"]] for r in rows]]
        self.d1, self.d2 = D[[dix[r["doc1"]] for r in rows]], D[[dix[r["doc2"]] for r in rows]]
        self.lex = {f: np.array([bool(NEG_RE["en"].search(r[f])) for r in rows], dtype=float) for f in FIELDS}

    def projout(self, v):
        s = object.__new__(Split)
        s.n, s.lex = self.n, self.lex
        for f in ("q1", "q2", "d1", "d2"):
            X = getattr(self, f)
            setattr(s, f, unit(X - np.outer(X @ v, v)))
        return s


def scores(sp, v, lam):
    """4 оценки на пункт: s(q1,d1), s(q1,d2), s(q2,d1), s(q2,d2). v=None — сырой косинус, v='lex' — регулярка."""
    cos = [np.sum(a * b, 1) for a, b in ((sp.q1, sp.d1), (sp.q1, sp.d2), (sp.q2, sp.d1), (sp.q2, sp.d2))]
    if v is None or lam == 0:
        return cos
    if isinstance(v, str):
        p = {"q1": sp.lex["q1"], "q2": sp.lex["q2"], "d1": sp.lex["doc1"], "d2": sp.lex["doc2"]}
    else:
        p = {f: getattr(sp, f) @ v for f in ("q1", "q2", "d1", "d2")}
    pen = [np.abs(p[a] - p[b]) for a, b in (("q1", "d1"), ("q1", "d2"), ("q2", "d1"), ("q2", "d2"))]
    return [c - lam * x for c, x in zip(cos, pen)]


def correct(sp, v=None, lam=0.0):
    """(пункт верен: оба запроса, запрос q1 верен, запрос q2 верен) — булевы массивы (n,)."""
    s11, s12, s21, s22 = scores(sp, v, lam)
    ok1, ok2 = s11 > s12, s22 > s21
    return ok1 & ok2, ok1, ok2


def pick_lambda(dev, v):
    """λ с максимальной pairwise accuracy на dev; при равенстве — наименьшая."""
    accs = np.array([correct(dev, v, lam)[0].mean() for lam in LAMBDAS])
    return float(LAMBDAS[int(np.argmax(accs))]), accs


def mcnemar(ok0, ok1):
    """Точный тест: b = исправлено методом, c = сломано методом."""
    b, c = int(np.sum(~ok0 & ok1)), int(np.sum(ok0 & ~ok1))
    return b, c, float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0


def boot_diff_ci(ok0, ok1, rng):
    idx = rng.integers(0, len(ok0), size=(N_BOOT, len(ok0)))
    d = ok1[idx].mean(1) - ok0[idx].mean(1)
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def h4_lambda(key, method):
    df = pd.read_csv(RESULTS / "h4_lambda.csv")
    row = df[(df.model == key) & (df.lang == "en") & (df.method == method)]
    return float(np.median([float(x) for x in row.iloc[0]["lambdas"].split()]))


def main(keys):
    data, info = load_nevir()
    dev_rows = data["train"] + data["validation"]
    test_rows = data["test"]
    all_rows = dev_rows + test_rows
    texts_q = sorted({r[f] for r in all_rows for f in ("q1", "q2")})
    texts_d = sorted({r[f] for r in all_rows for f in ("doc1", "doc2")})
    qix, dix = {t: i for i, t in enumerate(texts_q)}, {t: i for i, t in enumerate(texts_d)}
    log_path = RESULTS / "nevir_log.json"
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {}
    log["dataset"] = info
    rows, lam_rows = [], []
    for key in keys:
        Q, D, meta = embed(key, texts_q, texts_d)
        dev, test = Split(dev_rows, qix, dix, Q, D), Split(test_rows, qix, dix, Q, D)
        z = np.load(ARTIFACTS / "v_neg" / f"{key}_en.npz")
        fin = list(z["layers"]).index("final")
        v_neg, v_negB, v_edit = z["v_neg"][fin], z["v_negB"][fin], z["v_edit"][fin]
        assert v_neg.shape[0] == Q.shape[1], (key, v_neg.shape, Q.shape)

        res, lam_used = {}, {}
        res["M0_raw"] = correct(test)
        res["M1_projout"] = correct(test.projout(v_neg), None)
        for name, v, h4m in (("M2_feature", v_neg, "M2_feature"), ("M2B_feature", v_negB, "M2B_feature"),
                             ("M2_ctrl_edit", v_edit, None), ("M5_lexical", "lex", None)):
            lam, dev_acc = pick_lambda(dev, v)
            test_acc = np.array([correct(test, v, l)[0].mean() for l in LAMBDAS])
            lam_rows += [dict(model=key, method=name, split=s, lam=l, pairwise_acc=a)
                         for s, accs in (("dev", dev_acc), ("test", test_acc)) for l, a in zip(LAMBDAS, accs)]
            res[name], lam_used[name] = correct(test, v, lam), lam
            if h4m:
                lam_used[name + "_h4"] = h4_lambda(key, h4m)
                res[name + "_h4"] = correct(test, v, lam_used[name + "_h4"])
        rng = np.random.default_rng([SEED, 10])
        rand = []
        for _ in range(N_RANDOM):
            r = unit(rng.standard_normal(Q.shape[1]))
            rand.append(correct(test, r, pick_lambda(dev, r)[0]))
        res["M2_ctrl_random"] = tuple(np.mean([x[i] for x in rand], 0) for i in range(3))

        ok0 = res["M0_raw"][0]
        brng = np.random.default_rng([SEED, 11])
        for meth, (ok, ok1, ok2) in res.items():
            row = dict(model=key, method=meth, n=test.n, lam=lam_used.get(meth, np.nan),
                       pairwise_acc=float(np.mean(ok)), query_acc=float(np.mean(np.r_[ok1, ok2])))
            if meth not in ("M0_raw", "M2_ctrl_random"):
                row["b_fixed"], row["c_broken"], row["p_mcnemar"] = mcnemar(ok0, ok)
                row["diff_ci_lo"], row["diff_ci_hi"] = boot_diff_ci(ok0, ok, brng)
            rows.append(row)
        log[key] = meta
        print(key, {m: f"{np.mean(r[0]):.3f}" for m, r in res.items()}, "λ:", lam_used, flush=True)
        log_path.write_text(json.dumps(log, indent=1, ensure_ascii=False), encoding="utf-8")

    out = pd.DataFrame(rows)
    path = RESULTS / "nevir_methods.csv"
    if path.exists() and set(keys) != {k for k, (_, langs, _) in MODELS.items() if "en" in langs}:
        old = pd.read_csv(path)  # частичный перезапуск: заменить только посчитанные модели
        out = pd.concat([old[~old.model.isin(keys)], out], ignore_index=True)
    out.to_csv(path, index=False)
    lam_df = pd.DataFrame(lam_rows)
    lpath = RESULTS / "nevir_lambda.csv"
    if lpath.exists():
        old = pd.read_csv(lpath)
        lam_df = pd.concat([old[~old.model.isin(keys)], lam_df], ignore_index=True)
    lam_df.to_csv(lpath, index=False)


if __name__ == "__main__":
    main(sys.argv[1:] or [k for k, (_, langs, _) in MODELS.items() if "en" in langs])
