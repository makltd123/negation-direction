"""Графики отчёта (results/figures/*.png) из CSV шагов 2–6.

Цвет закреплён за моделью во всех графиках (палитра проверена валидатором CVD, 8 слотов по порядку).
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIG, MODEL_LABEL, MODELS, RESULTS

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
COLOR = {m: PALETTE[i] for i, m in enumerate(MODELS)}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
NEUTRAL = "#8a8985"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "axes.titlecolor": INK, "legend.frameon": False, "text.color": INK,
})


def ml_order(df):
    keys = [(m, l) for l in ("en", "ru") for m in MODELS if l in MODELS[m][1]]
    have = set(zip(df["model"], df["lang"]))
    return [k for k in keys if k in have]


SHORT = {"mpnet": "mpnet", "bge": "bge-base", "e5": "e5-base", "me5": "m-e5-base", "mminilm": "m-MiniLM",
         "bgem3": "bge-m3", "rubert": "rubert-tiny2", "user": "USER-base"}


def short(m, l):
    return f"{SHORT[m]}\n[{l.upper()}]"


def depth(df):
    return np.where(df["layer"] == "final", np.nan, pd.to_numeric(df["layer"], errors="coerce") / df["n_layers"])


# ---------------------------------------------------------------- 1. H1: heatmaps фолдов
def fig_h1_heatmaps():
    mats = json.loads((RESULTS / "h1_fold_matrices.json").read_text(encoding="utf-8"))
    h1 = pd.read_csv(RESULTS / "h1_stability.csv")
    keys = [k for k in ml_order(h1) if f"{k[0]}_{k[1]}" in mats]
    fig, axes = plt.subplots(2, len(keys), figsize=(1.65 * len(keys) + 1, 4.3))
    for j, (m, l) in enumerate(keys):
        r = h1[(h1.model == m) & (h1.lang == l) & (h1.layer == "final") & (h1.kind == "neg")].iloc[0]
        for i, (proc, title, stat, null) in enumerate((
                ("tz", "ТЗ: 4/5", r.tz5fold, r.tz5fold_null_mean),
                ("disjoint", "disjoint 1/5", r.disj5fold, r.disj5fold_null_p99))):
            C = np.array(mats[f"{m}_{l}"]["neg"][proc])
            ax = axes[i, j]
            im = ax.imshow(C, vmin=0, vmax=1, cmap="Blues")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_xlabel(f"mean={stat:.2f}\n{'нуль ≈' if proc == 'tz' else 'нуль p99 ='} {null:.2f}",
                          fontsize=8, color=INK2)
            if j == 0:
                ax.set_ylabel(title, fontsize=9)
            if i == 0:
                ax.set_title(short(m, l), fontsize=8)
    fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01, label="cos между v_neg фолдов")
    fig.suptitle("H1. Попарный cos между v_neg из разных фолдов (final). Верх: процедура ТЗ (фолды по 4/5 "
                 "пересекаются, поэтому даже шум даёт ≈0.7); низ: непересекающиеся фолды по 1/5", fontsize=10, x=0.02, ha="left")
    fig.savefig(FIG / "fig1_h1_fold_heatmaps.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 2. H2: self vs controls
def fig_h2_bars():
    sp = pd.read_csv(RESULTS / "h2_specificity.csv")
    h1 = pd.read_csv(RESULTS / "h1_stability.csv")
    sp = sp[sp.layer == "final"]
    keys = ml_order(sp)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.6), sharex=True)
    for ax, (main, ctrls, title) in zip(axes, (
            ("neg", ("edit", "ant", "rp"), "v_neg (как в ТЗ: среднее по всем парам)"),
            ("negB", ("editB", "antB", "rp"), "v_negB (позитивные, негативные и нейтральные утверждения с равным весом)"))):
        series = [("self", f"{main} ↔ {main} (split-half)", "#2a78d6"),
                  (ctrls[0], "↔ наречие (edit)", "#eb6834"),
                  (ctrls[1], "↔ антоним (ant)", "#1baf7a"),
                  (ctrls[2], "↔ случайная разница (rp)", "#eda100")]
        x = np.arange(len(keys))
        w = 0.2
        for i, (key, lab, col) in enumerate(series):
            vals = []
            for m, l in keys:
                d = sp[(sp.model == m) & (sp.lang == l) & (sp.a == main)]
                vals.append(d.self_a.iloc[0] if key == "self" else d[d.b == key].cross.iloc[0])
            ax.bar(x + (i - 1.5) * w, vals, w * 0.9, color=col, label=lab)
        nulls = [h1[(h1.model == m) & (h1.lang == l) & (h1.layer == "final") & (h1.kind == main)]
                 .split_half_null_p99.iloc[0] for m, l in keys]
        ax.scatter(x - 1.5 * w, nulls, marker="_", s=180, color=INK, zorder=3,
                   label="99-й перцентиль нуля (смена знака)")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_ylim(-0.4, 1.05)
        ax.set_ylabel("cos (половина A ↔ половина B)")
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.legend(ncol=5, fontsize=7.5, loc="lower left")
    axes[-1].set_xticks(np.arange(len(keys)), [short(m, l) for m, l in keys], fontsize=7.5)
    fig.suptitle("H2. Устойчивость направления негации против его похожести на контрольные направления "
                 "(final, 200 разбиений утверждений на половины)", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_h2_self_vs_controls.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 3. H5: послойные кривые
def fig_h5_layers():
    h1 = pd.read_csv(RESULTS / "h1_stability.csv")
    cf = pd.read_csv(RESULTS / "h1_crossfit.csv")
    sp = pd.read_csv(RESULTS / "h2_specificity.csv")
    val = pd.read_csv(RESULTS / "h2_valence.csv")
    panels = [
        ("Устойчивость v_neg (split-half)", h1[h1.kind == "neg"], "split_half", (0, 1.02)),
        ("Выравнивание отдельной пары с осью\ncos(d_i, v_neg), кросс-фит", cf, "align_neg", (0, 1.02)),
        ("Похожесть v_neg на v_edit (наречие)\nс поправкой на шум", sp[(sp.a == "neg") & (sp.b == "edit")],
         "disattenuated", (-0.4, 1.05)),
        ("v_neg по позитивным ↔ по негативным\nутверждениям (<0: негация = смена тональности)", val,
         "neg_pos_vs_neg", (-1, 1.02)),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
    for row, lang in enumerate(("en", "ru")):
        for col, (title, df, ycol, ylim) in enumerate(panels):
            ax = axes[row, col]
            d = df[df.lang == lang]
            for m in MODELS:
                dm = d[d.model == m]
                if dm.empty:
                    continue
                layers = dm[dm.layer != "final"].copy()
                layers["x"] = pd.to_numeric(layers.layer) / layers.n_layers
                layers = layers.sort_values("x")
                ax.plot(layers.x, layers[ycol], color=COLOR[m], lw=2, label=MODEL_LABEL[m])
                fin = dm[dm.layer == "final"][ycol].iloc[0]
                ax.scatter([1.06], [fin], color=COLOR[m], s=36, zorder=3, edgecolor=SURF, linewidth=1.5)
            ax.set_ylim(*ylim)
            ax.set_xlim(-0.03, 1.1)
            if row == 0:
                ax.set_title(title, loc="left", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{'Английский' if lang == 'en' else 'Русский'}", fontsize=10)
            if row == 1:
                ax.set_xlabel("глубина слоя (слой / число слоёв); точка справа = final")
        axes[row, 0].legend(fontsize=7.5, loc="lower left")
        axes[row, 3].axhline(0, color=INK2, lw=0.8)
    fig.suptitle("H5. Качество направления негации по слоям энкодера", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_h5_layers.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 4. H4: до/после
METH = [("M0_raw", "baseline (cos)", "#52514e", None), ("M1_projout", "вычесть проекцию v_neg", "#2a78d6", None),
        ("M1B_projout", "вычесть проекцию v_negB", "#2a78d6", "////"),
        ("M2_feature", "признак |Δ проекции на v_neg|", "#eb6834", None),
        ("M2B_feature", "признак |Δ проекции на v_negB|", "#eb6834", "////"),
        ("M5_lexical_T1", "контроль: регулярка «есть ли не/not» (λ по T1)", "#8a8985", "....")]


def fig_h4():
    h4 = pd.read_csv(RESULTS / "h4_methods.csv")
    keys = ml_order(h4)
    types = [("T1_neg", "T1: ловушка «не A»"), ("T2_ant", "T2: ловушка-антоним"),
             ("T3_negpara", "T3: правильный ответ «не B»"), ("T4_neganchor", "T4: якорь «не A»"),
             ("MACRO", "Среднее по T1–T4")]
    fig, axes = plt.subplots(len(types), 1, figsize=(12.5, 13), sharex=True)
    x = np.arange(len(keys))
    w = 0.8 / len(METH)
    for ax, (t, tl) in zip(axes, types):
        for i, (meth, lab, col, hatch) in enumerate(METH):
            vals = [h4[(h4.model == m) & (h4.lang == l) & (h4.method == meth) & (h4.type == t)].fail_rate.iloc[0]
                    for m, l in keys]
            ax.bar(x + (i - (len(METH) - 1) / 2) * w, vals, w * 0.88, color=SURF if hatch else col,
                   edgecolor=col, hatch=hatch, linewidth=0.8, label=lab)
        ax.axhline(0.5, color=INK2, lw=0.8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("доля провалов")
        ax.set_title(tl, loc="left", fontsize=9.5)
    axes[-1].set_xticks(x, [short(m, l) for m, l in keys], fontsize=7.5)
    axes[0].legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.55))
    fig.suptitle("H4. Доля провалов на отложенных тройках: cos(anchor, ловушка) ≥ cos(anchor, правильный); "
                 "меньше = лучше", x=0.01, ha="left", fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_h4_failrates.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 5. H3: доля в подпространстве тем
def fig_h3():
    h3 = pd.read_csv(RESULTS / "h3_topic.csv")
    d = h3[h3.layer == "final"]
    keys = ml_order(d)
    ser = [("frac_neg", "v_neg", "#2a78d6", None), ("frac_negB", "v_negB", "#2a78d6", "////"),
           ("frac_edit", "v_edit", "#eb6834", None), ("frac_ant", "v_ant", "#1baf7a", None),
           ("frac_rp", "v_rp (случайная разница)", "#eda100", None)]
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(keys))
    w = 0.16
    for i, (c, lab, col, hatch) in enumerate(ser):
        vals = [d[(d.model == m) & (d.lang == l)][c].iloc[0] for m, l in keys]
        ax.bar(x + (i - 2) * w, vals, w * 0.9, color=SURF if hatch else col, edgecolor=col, hatch=hatch,
               linewidth=0.8, label=lab)
    an = [d[(d.model == m) & (d.lang == l)].null_aniso.iloc[0] for m, l in keys]
    iso = [d[(d.model == m) & (d.lang == l)].null_iso.iloc[0] for m, l in keys]
    ax.scatter(x, an, marker="_", s=600, color=INK, zorder=3, label="нуль «как данные» (анизотропный)")
    ax.scatter(x, iso, marker="_", s=600, color=NEUTRAL, zorder=3, label="нуль изотропный (7/d)")
    ax.set_xticks(x, [short(m, l) for m, l in keys], fontsize=7.5)
    ax.set_ylabel("доля ‖v‖² в подпространстве тем")
    ax.set_title("H3. Какая доля направления лежит в 7-мерном подпространстве тем (final)", loc="left")
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.savefig(FIG / "fig5_h3_topic.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 6. H5: EN <-> RU в одной модели
def fig_h5_crossling():
    cl = pd.read_csv(RESULTS / "h5_crosslingual.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    combos = [("neg", "neg", "v_neg(EN) ↔ v_neg(RU)", 2.2, "-", INK),
              ("negB", "negB", "v_negB(EN) ↔ v_negB(RU)", 2.2, "-", "#2a78d6"),
              ("edit", "edit", "v_edit(EN) ↔ v_edit(RU)", 1.4, "-", NEUTRAL),
              ("neg", "edit", "v_neg(EN) ↔ v_edit(RU)", 1.4, ":", NEUTRAL),
              ("ant", "ant", "v_ant(EN) ↔ v_ant(RU)", 1.4, "--", NEUTRAL)]
    for ax, m in zip(axes, [m for m in MODELS if set(MODELS[m][1]) == {"en", "ru"}]):
        d = cl[cl.model == m]
        for k1, k2, lab, lw, ls, col in combos:
            dd = d[(d.kind_en == k1) & (d.kind_ru == k2) & (d.layer != "final")].copy()
            dd["x"] = pd.to_numeric(dd.layer) / dd.n_layers
            ax.plot(dd.x, dd.cos, lw=lw, ls=ls, color=col, label=lab)
        ax.axhline(0, color=GRID, lw=1)
        ax.set_title(MODEL_LABEL[m], loc="left", fontsize=9)
        ax.set_xlabel("глубина слоя")
    axes[0].set_ylabel("cos направлений EN и RU")
    axes[0].legend(fontsize=7.5)
    fig.suptitle("H5. Совпадает ли направление негации для английского и русского внутри одной модели",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_h5_crosslingual.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 7. детектор вне шаблонов
def fig_probe_natural():
    cf = pd.read_csv(RESULTS / "h1_crossfit.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True, sharey=True)
    for row, lang in enumerate(("en", "ru")):
        for col, (ycol, title) in enumerate((("natural_auc_neg", "x·v_neg"), ("natural_auc_negB", "x·v_negB"))):
            ax = axes[row, col]
            d = cf[cf.lang == lang]
            for m in MODELS:
                dm = d[d.model == m]
                if dm.empty:
                    continue
                layers = dm[dm.layer != "final"].copy()
                layers["x"] = pd.to_numeric(layers.layer) / layers.n_layers
                layers = layers.sort_values("x")
                ax.plot(layers.x, layers[ycol], color=COLOR[m], lw=2, label=MODEL_LABEL[m])
                ax.scatter([1.06], [dm[dm.layer == "final"][ycol].iloc[0]], color=COLOR[m], s=36, zorder=3,
                           edgecolor=SURF, linewidth=1.5)
                ax.scatter([1.06], [dm[dm.layer == "final"]["natural_auc_edit"].iloc[0]], color=COLOR[m], s=26,
                           marker="x", zorder=3)
            ax.axhline(0.5, color=INK2, lw=0.8)
            ax.set_ylim(0.3, 1.01)
            if row == 0:
                ax.set_title(f"AUC детектора {title}", loc="left", fontsize=9.5)
            if col == 0:
                ax.set_ylabel("Английский" if lang == "en" else "Русский")
                ax.legend(fontsize=7, loc="lower left")
            if row == 1:
                ax.set_xlabel("глубина слоя; справа: ● final, × контроль x·v_edit на final")
    fig.suptitle("Читается ли отрицание в живых фразах (STS-B, ~20% с отрицанием): AUC проекции на направление, "
                 "найденное на шаблонных парах", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig7_probe_natural.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    for f in (fig_h1_heatmaps, fig_h2_bars, fig_h5_layers, fig_h4, fig_h3, fig_h5_crossling, fig_probe_natural):
        try:
            f()
            print("ok", f.__name__)
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
