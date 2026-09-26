"""Шаг 7. Сводные таблицы и механические вердикты по критериям из RESEARCH_LOG §4.

Выход: results/verdicts.csv, results/summary_tables.md
"""
import numpy as np
import pandas as pd

from common import MODEL_LABEL, MODELS, RESULTS


def ml_keys(df):
    have = set(zip(df["model"], df["lang"]))
    return [(m, l) for l in ("en", "ru") for m in MODELS if l in MODELS[m][1] and (m, l) in have]


def fmt(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def pfmt(p):
    return "—" if p is None or np.isnan(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def md_table(rows, cols=None):
    if not rows:
        return "_нет данных_"
    cols = cols or list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def main():
    h1 = pd.read_csv(RESULTS / "h1_stability.csv", dtype={"layer": str})
    cf = pd.read_csv(RESULTS / "h1_crossfit.csv", dtype={"layer": str})
    sp = pd.read_csv(RESULTS / "h2_specificity.csv", dtype={"layer": str})
    val = pd.read_csv(RESULTS / "h2_valence.csv", dtype={"layer": str})
    con = pd.read_csv(RESULTS / "h2_construction.csv", dtype={"layer": str})
    h3 = pd.read_csv(RESULTS / "h3_topic.csv", dtype={"layer": str})
    h4 = pd.read_csv(RESULTS / "h4_methods.csv")
    try:
        cl = pd.read_csv(RESULTS / "h5_crosslingual.csv", dtype={"layer": str})
    except pd.errors.EmptyDataError:
        cl = pd.DataFrame(columns=["model", "layer", "kind_en", "kind_ru", "cos", "cos_corrected"])
    cm = pd.read_csv(RESULTS / "h5_crossmodel.csv")
    cmp_ = pd.read_csv(RESULTS / "h5_crossmodel_pairs.csv")

    def g(df, m, l, layer="final", **kw):
        d = df[(df.model == m) & (df.lang == l) & (df.layer == layer)]
        for k, v in kw.items():
            d = d[d[k] == v]
        assert len(d) == 1, (m, l, layer, kw, len(d))
        return d.iloc[0]

    verdicts, t1, t2, t2b, t3, t4, t5 = [], [], [], [], [], [], []
    for m, l in ml_keys(h1):
        name = f"{MODEL_LABEL[m]} [{l.upper()}]"
        # ---------------- H1
        r = g(h1, m, l, kind="neg")
        if r.split_half >= 0.8 and r.split_half > r.split_half_null_p99 and r.snr_half_lo >= 0.7:
            v1 = "да"
        elif r.split_half > r.split_half_null_p99:
            v1 = "частично"
        else:
            v1 = "нет"
        rb = g(h1, m, l, kind="negB")
        t1.append({"модель": name, "split-half": fmt(r.split_half), "SNR-оценка [95% ДИ]":
                   f"{fmt(r.snr_half)} [{fmt(r.snr_half_lo)}, {fmt(r.snr_half_hi)}]",
                   "нуль p99": fmt(r.split_half_null_p99),
                   "наивный бутстрап": f"[{fmt(r.split_half_boot_lo)}, {fmt(r.split_half_boot_hi)}]",
                   "ТЗ 4/5 (нуль)": f"{fmt(r.tz5fold)} ({fmt(r.tz5fold_null_mean)})",
                   "disjoint 1/5": fmt(r.disj5fold),
                   "align": fmt(g(cf, m, l).align_neg),
                   "negB split-half (нуль p99)": f"{fmt(rb.split_half)} ({fmt(rb.split_half_null_p99)})",
                   "H1": v1})
        # ---------------- H2
        s_e, s_r = g(sp, m, l, a="neg", b="edit"), g(sp, m, l, a="neg", b="rp")
        s_a = g(sp, m, l, a="neg", b="ant")
        c = g(cf, m, l)
        vv = g(val, m, l)
        a_ok = (s_e.disattenuated < 0.5 and s_e.gap_lo > 0) and (s_r.gap_lo > 0 and
                                                              (np.isnan(s_r.disattenuated) or s_r.disattenuated < 0.5))
        b_ok = c.nearest_acc >= 0.9
        v2 = "да" if a_ok and b_ok else ("частично" if a_ok or b_ok else "нет")
        not_sentiment = vv.neg_pos_vs_neg > 0 and vv.neg_pos_vs_neg >= vv.neg_pos_vs_neg_null_p05
        t2.append({"модель": name, "↔edit (очищ.)": fmt(s_e.disattenuated), "↔rp (очищ.)": fmt(s_r.disattenuated),
                   "↔ant (очищ.)": fmt(s_a.disattenuated), "ближайшее направл.": fmt(c.nearest_acc),
                   "v_neg: позит.↔негат. утв. (нуль)": f"{fmt(vv.neg_pos_vs_neg)} ({fmt(vv.neg_pos_vs_neg_null_mean)})",
                   "v_ant: позит.↔негат.": fmt(vv.ant_pos_vs_neg),
                   "cos с осью тональности": fmt(vv.cos_sent_neg), "H2 (план)": v2,
                   "не ось тональности?": "да" if not_sentiment else "нет"})
        sbe, sba = g(sp, m, l, a="negB", b="editB"), g(sp, m, l, a="negB", b="antB")
        t2b.append({"модель": name, "negB split-half": fmt(g(h1, m, l, kind="negB").split_half),
                    "↔editB (очищ.)": fmt(sbe.disattenuated), "↔antB (очищ.)": fmt(sba.disattenuated),
                    "ближайшее (B)": fmt(c.nearest_accB), "cos с осью тональности": fmt(vv.cos_sent_negB),
                    "probe AUC: v_neg / v_negB": f"{fmt(c.probe_auc)} / {fmt(c.probe_aucB)}",
                    "STS-B AUC: v_neg / v_negB / v_edit": f"{fmt(c.natural_auc_neg)} / {fmt(c.natural_auc_negB)} / "
                                                          f"{fmt(c.natural_auc_edit)}",
                    "доля пар со сдвигом вдоль оси: v_neg / v_negB": f"{fmt(c.pair_sign)} / {fmt(c.pair_signB)}"})
        # ---------------- H3
        t = g(h3, m, l)
        v3_frac = t.frac_neg <= t.null_aniso
        v3_topic = t.topic_consistency_neg_pct >= 0.05
        v3 = "да" if v3_frac and v3_topic else ("частично" if v3_frac or v3_topic else "нет")
        t3.append({"модель": name, "доля v_neg": fmt(t.frac_neg, 3), "доля v_negB": fmt(t.frac_negB, 3),
                   "нуль изотр.": fmt(t.null_iso, 3), "нуль анизотр.": fmt(t.null_aniso, 3),
                   "доля v_edit": fmt(t.frac_edit, 3), "доля v_rp": fmt(t.frac_rp, 3),
                   "max |cos| с парой тем": fmt(t.max_abs_cos_topic_pair),
                   "консистентность по темам (перцентиль)": f"{fmt(t.topic_consistency_neg)} "
                                                           f"({fmt(t.topic_consistency_neg_pct)})",
                   "H3": v3})
        # ---------------- H4
        d4 = h4[(h4.model == m) & (h4.lang == l)]

        def fr(meth, typ):
            return d4[(d4.method == meth) & (d4.type == typ)].iloc[0]

        base_macro = fr("M0_raw", "MACRO").fail_rate
        ok4 = []
        for meth in ("M1_projout", "M2_feature"):
            r1 = fr(meth, "T1_neg")
            better_t1 = r1.fail_rate < fr("M0_raw", "T1_neg").fail_rate and r1.p_mcnemar < 0.05
            macro_ok = fr(meth, "MACRO").fail_rate <= base_macro
            # строгое прочтение §4: «T1 ценой T3/T4» = значимое ухудшение защитного типа
            guard_ok = all(not (fr(meth, t).fail_rate > fr("M0_raw", t).fail_rate and fr(meth, t).p_mcnemar < 0.05)
                           for t in ("T3_negpara", "T4_neganchor"))
            ok4.append((better_t1, macro_ok, guard_ok))
        if any(a and b and c for a, b, c in ok4):
            v4 = "да"
        elif any(a and b for a, b, _ in ok4):
            v4 = "частично (T1 ценой T3/T4)"
        elif any(a for a, _, _ in ok4):
            v4 = "частично (среднее хуже)"
        else:
            v4 = "нет"
        v4_lenient = "да" if any(a and b for a, b, _ in ok4) else v4
        row = {"модель": name}
        for meth, lab in (("M0_raw", "base"), ("M1_projout", "M1"), ("M1B_projout", "M1B"), ("M2_feature", "M2"),
                          ("M2B_feature", "M2B"), ("M2_feature_T1", "M2(T1)"), ("M3_center", "center"),
                          ("M4_whiten", "whiten"), ("M1_ctrl_edit", "M1 edit"), ("M2_ctrl_edit", "M2 edit"),
                          ("M5_lexical", "regex"), ("M5_lexical_T1", "regex(T1)")):
            row[lab] = f"{fmt(fr(meth, 'T1_neg').fail_rate)} / {fmt(fr(meth, 'MACRO').fail_rate)}"
        row["p(T1): M1 / M2"] = f"{pfmt(fr('M1_projout', 'T1_neg').p_mcnemar)} / {pfmt(fr('M2_feature', 'T1_neg').p_mcnemar)}"
        row["M2: T3 база→M2 (p)"] = (f"{fmt(fr('M0_raw', 'T3_negpara').fail_rate)}→{fmt(fr('M2_feature', 'T3_negpara').fail_rate)}"
                                     f" ({pfmt(fr('M2_feature', 'T3_negpara').p_mcnemar)})")
        row["M2: T4 база→M2 (p)"] = (f"{fmt(fr('M0_raw', 'T4_neganchor').fail_rate)}→{fmt(fr('M2_feature', 'T4_neganchor').fail_rate)}"
                                     f" ({pfmt(fr('M2_feature', 'T4_neganchor').p_mcnemar)})")
        row["H4"] = v4
        t4.append(row)
        verdicts.append(dict(model=m, lang=l, H1=v1, H2=v2, H2_not_sentiment=not_sentiment, H3=v3, H4=v4,
                             H4_lenient=v4_lenient))

    # ---------------- H5: лучший слой
    for m, l in ml_keys(h1):
        d = cf[(cf.model == m) & (cf.lang == l) & (cf.layer != "final")]
        s = sp[(sp.model == m) & (sp.lang == l) & (sp.a == "neg") & (sp.b == "edit") & (sp.layer != "final")]
        vv = val[(val.model == m) & (val.lang == l) & (val.layer != "final")]
        nl = int(d.n_layers.iloc[0])
        best_align = d.loc[d.align_neg.idxmax()]
        best_nat = d.loc[d.natural_auc_negB.idxmax()]
        fin = g(cf, m, l)
        flip = vv[vv.neg_pos_vs_neg < 0]
        t5.append({"модель": f"{MODEL_LABEL[m]} [{l.upper()}]", "слоёв": nl,
                   "align: лучший слой → final": f"{best_align.layer} ({fmt(best_align.align_neg)}) → {fmt(fin.align_neg)}",
                   "STS-B AUC v_negB: лучший → final": f"{best_nat.layer} ({fmt(best_nat.natural_auc_negB)}) → "
                                                       f"{fmt(fin.natural_auc_negB)}",
                   "первый слой, где v_neg(позит.)↔v_neg(негат.) < 0": (str(int(flip.layer.astype(int).min()))
                                                                          if len(flip) else "нет"),
                   "↔edit очищ.: слой 0/1 → final": f"{fmt(s.sort_values('layer', key=lambda x: x.astype(int)).disattenuated.iloc[0])} → "
                                                    f"{fmt(g(sp, m, l, a='neg', b='edit').disattenuated)}"})

    t6 = []
    for m in cl.model.unique():
        d = cl[(cl.model == m) & (cl.layer == "final")]

        def cc(a, b):
            r = d[(d.kind_en == a) & (d.kind_ru == b)].iloc[0]
            return f"{fmt(r.cos)} ({fmt(r.cos_corrected)})"
        t6.append({"модель": MODEL_LABEL[m], "neg↔neg": cc("neg", "neg"), "negB↔negB": cc("negB", "negB"),
                   "edit↔edit": cc("edit", "edit"), "ant↔ant": cc("ant", "ant"), "neg↔edit": cc("neg", "edit"),
                   "neg↔ant": cc("neg", "ant")})

    t7 = []
    for l in ("en", "ru"):
        d = cm[cm.lang == l]
        for k in ("neg", "negB", "edit", "ant"):
            same = d[(d.kind_src == k) & (d.kind_dst == k)].cos
            row = {"язык": l.upper(), "направление": k, "cos(перенесённое, своё) среднее [мин, макс]":
                   f"{fmt(same.mean())} [{fmt(same.min())}, {fmt(same.max())}]"}
            for k2 in ("neg", "edit", "ant"):
                row[f"↔ {k2} целевой модели"] = ("—" if k2 == k else
                                                 fmt(d[(d.kind_src == k) & (d.kind_dst == k2)].cos.mean()))
            pc = cmp_[(cmp_.lang == l) & (cmp_.kind == k)].pair_cos.mean() if k in ("neg", "edit", "ant") else np.nan
            row["перенос отдельных пар (cos)"] = fmt(pc)
            t7.append(row)

    pd.DataFrame(verdicts).to_csv(RESULTS / "verdicts.csv", index=False)
    parts = [
        "# Сводные таблицы (генерируются s09_summary.py)\n",
        "Все значения — финальный выход модели (вектор для поиска), если не сказано иное.\n",
        "## H1. Устойчивость v_neg\n", md_table(t1),
        "\n\n## H2. Специфичность v_neg (по плану)\n", md_table(t2),
        "\n\n## H2b. Сбалансированное по тональности направление v_negB\n", md_table(t2b),
        "\n\n## H3. Тема\n", md_table(t3),
        "\n\n## H4. Доля провалов T1 / среднее по T1–T4\n", md_table(t4),
        "\n\n## H5a. Слои\n", md_table(t5),
        "\n\n## H5b. EN ↔ RU внутри одной модели (final): cos (с поправкой на шум)\n", md_table(t6),
        "\n\n## H5c. Перенос между моделями через ridge-отображение без отрицаний (final)\n", md_table(t7),
        "\n\n## Конструкции (final)\n",
        md_table(con[con.layer == "final"][["model", "lang", "a", "b", "kind", "cos", "null_mean", "pct_rank"]]
                 .round(3).to_dict("records")),
        "\n",
    ]
    (RESULTS / "summary_tables.md").write_text("\n".join(parts), encoding="utf-8")
    print(pd.DataFrame(verdicts).to_string())


if __name__ == "__main__":
    main()
