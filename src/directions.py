"""DiffMean-направления и метрики их качества.

Все направления считаются через суммы разностей по «группам» (группа = базовое утверждение,
его 5 вариантов с разными рамками сильно коррелированы, поэтому делить на фолды можно только по
группам). Направление по набору групп с весами w: unit(w @ S), где S[g] = сумма разностей группы g.

Виды направлений: neg (отрицание), edit (контроль 1: наречие), ant (контроль 3: антоним),
rp (контроль 2: случайная разница) и *сбалансированные по тональности* negB / editB / antB:
то же, но позитивные, негативные и нейтральные утверждения входят с равным суммарным весом
(добавлено после того, как выяснилось, что v_neg зависит от тональности утверждения, см. RESEARCH_LOG §6).
"""
import numpy as np
from sklearn.metrics import roc_auc_score

from common import unit

KINDS = ("neg", "edit", "ant", "rp", "negB", "editB", "antB")
BASE_KINDS = ("neg", "edit", "ant", "rp")


def is_bal(kind):
    return kind.endswith("B")


def base_of(kind):
    return kind[:-1] if is_bal(kind) else kind


def balance(W, cls):
    """Перевзвешивает строки W (R, G) или (G,) так, чтобы каждый класс групп имел суммарный вес 1."""
    W = np.atleast_2d(np.asarray(W, dtype=float)).copy()
    for c in np.unique(cls):
        m = cls == c
        tot = W[:, m].sum(1, keepdims=True)
        W[:, m] = np.where(tot > 0, W[:, m] / np.where(tot > 0, tot, 1), 0)
    return W


class PairSet:
    """Разности пар (L2-нормированные представления) для одной модели/языка/слоя."""

    def __init__(self, emb, pairs, layer):
        X = emb.rep(layer)
        ix = emb.idx
        self.Xs = {k: X[[ix[r[k]] for r in pairs]] for k in ("pos", "neg", "edit", "ant")}
        pos = self.Xs["pos"]
        self.D = {k: self.Xs[k] - pos for k in ("neg", "edit", "ant")}
        sign = np.array([r["rp_sign"] for r in pairs], dtype=float)[:, None]
        self.D["rp"] = sign * (pos[[r["rp_partner"] for r in pairs]] - pos)
        self.groups = np.array([r["base_id"] for r in pairs])
        self.G = int(self.groups.max()) + 1
        self.S = {k: group_sums(D, self.groups, self.G) for k, D in self.D.items()}
        first = {}
        for r in pairs:
            first.setdefault(r["base_id"], r)
        self.g_topic = np.array([first[g]["topic"] for g in range(self.G)])
        self.g_valence = np.array([first[g]["valence"] for g in range(self.G)])
        self.g_constr = np.array([first[g]["construction"] for g in range(self.G)])

    def direction(self, kind, w=None):
        w = np.ones(self.G) if w is None else np.asarray(w, dtype=float)
        return unit(self.W(kind, w)[0] @ self.S[base_of(kind)])

    def W(self, kind, W):
        """Веса групп для вида направления (для сбалансированных — нормировка по классам тональности)."""
        return balance(W, self.g_valence) if is_bal(kind) else np.atleast_2d(W)

    def Sk(self, kind):
        return self.S[base_of(kind)]


def group_sums(D, groups, G):
    S = np.zeros((G, D.shape[1]), dtype=np.float64)
    np.add.at(S, groups, D)
    return S


def group_folds(G, k, rng):
    fold = np.empty(G, dtype=int)
    fold[rng.permutation(G)] = np.arange(G) % k
    return fold


def half_weights(G, R, rng, counts=None):
    """R случайных разбиений групп на две непересекающиеся половины -> (WA, WB) формы (R, G)."""
    counts = np.ones(G) if counts is None else counts
    present = np.flatnonzero(counts > 0)
    WA = np.zeros((R, G))
    for r in range(R):
        p = rng.permutation(present)
        WA[r, p[: len(p) // 2]] = 1
    WB = (counts > 0) - WA
    return WA * counts, WB * counts


def half_weights_strat(cls, R, rng):
    """Как half_weights, но каждая половина получает половину групп КАЖДОГО класса."""
    G = len(cls)
    WA = np.zeros((R, G))
    for r in range(R):
        for c in np.unique(cls):
            idx = rng.permutation(np.flatnonzero(cls == c))
            k = len(idx) // 2 + (rng.random() < 0.5 if len(idx) % 2 else 0)
            WA[r, idx[:k]] = 1
    return WA, 1 - WA


def cos_rows(A, B):
    return np.sum(unit(A) * unit(B), axis=-1)


# ---------------------------------------------------------------- H1: устойчивость
def split_half(S, WA, WB):
    """Средний cos между направлениями по непересекающимся половинам групп."""
    return float(np.mean(cos_rows(WA @ S, WB @ S)))


def cross_half(S1, S2, WA, WB):
    """Похожесть направления 1 на направление 2 при тех же разбиениях (симметризовано)."""
    return float(np.mean(0.5 * (cos_rows(WA @ S1, WB @ S2) + cos_rows(WA @ S2, WB @ S1))))


def fold_matrix(S, fold, k=5, disjoint=False, cls=None):
    """Процедура из ТЗ: направление по 4/5 групп (или по 1/5, disjoint=True) -> матрица cos k x k."""
    W = np.stack([(fold == i) if disjoint else (fold != i) for i in range(k)]).astype(float)
    if cls is not None:
        W = balance(W, cls)
    V = unit(W @ S)
    return V @ V.T


def offdiag_mean(C):
    k = C.shape[0]
    return float((C.sum() - np.trace(C)) / (k * k - k))


def signflip(S, rng):
    """Нулевая гипотеза «общего направления нет»: случайный знак у каждой группы."""
    return S * rng.choice([-1.0, 1.0], size=(S.shape[0], 1))


def snr_reliability(S, counts=None):
    """Аналитическая оценка устойчивости направления через моменты групповых средних.

    M_g — средняя разность группы g. Модель: M_g = mu + шум с tr(Cov) = b.
    a = ||mean M||^2 - b/G — несмещённая оценка ||mu||^2.
    Ожидаемый cos между направлениями по двум непересекающимся половинам (G/2 групп):
        S_half ≈ a / (a + 2b/G);   между двумя полными оценками: a / (a + b/G).
    В отличие от split-half, эти моменты корректно бутстрапятся (не нужно делить выборку).
    """
    counts = np.ones(S.shape[0]) if counts is None else counts
    M = S / 5.0  # 5 рамок на утверждение
    G = counts.sum()
    mean = (counts[:, None] * M).sum(0) / G
    b = float((counts[:, None] * (M - mean) ** 2).sum() / (G - 1))
    a = float(mean @ mean - b / G)
    a = max(a, 0.0)
    return a / (a + 2 * b / G), a, b


def snr_bootstrap_ci(S, rng, n_boot=300, level=95):
    """ДИ для snr_reliability: basic bootstrap с «истинным» значением бутстрап-мира.

    В бутстрап-мире истина — эмпирическое распределение групп: ||mean||^2 (plug-in) и дисперсия
    b*(G-1)/G. Оценка a* несмещённа для plug-in ||mean||^2, а не для a, поэтому наивный
    перцентильный интервал смещён вверх при слабом сигнале. Интервал: theta_hat - квантили(theta* - theta_bw).
    """
    G = S.shape[0]
    th, a, b = snr_reliability(S)
    a_bw, b_bw = a + b / G, b * (G - 1) / G
    th_bw = a_bw / (a_bw + 2 * b_bw / G)
    diffs = []
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, G, G), minlength=G).astype(float)
        diffs.append(snr_reliability(S, counts)[0] - th_bw)
    lo, hi = np.percentile(diffs, [(100 - level) / 2, 100 - (100 - level) / 2])
    return th, float(np.clip(th - hi, -1, 1)), float(np.clip(th - lo, -1, 1))


def spearman_brown(s_half):
    """Ожидаемый cos между двумя независимыми оценками направления по ПОЛНЫМ данным."""
    return 2 * s_half / (1 + s_half) if s_half > 0 else float("nan")


# ---------------------------------------------------------------- per-pair и классификация
def crossfit(ps: PairSet, fold, k=5):
    """Кросс-фит по группам: для строк фолда i направления считаются по остальным фолдам.

    Возвращает dict:
      align[kind]   — cos(d_i, v_kind^{-i}) для каждой пары (насколько отдельная пара идёт вдоль оси)
      nearest_acc   — доля held-out разностей neg/edit/ant, ближайшее направление которых «своё»
      probe_auc     — AUC скалярной проекции x·v_neg: отрицательные предложения против pos/edit/ant
      probe_auc_pos — то же, только против pos
      pair_sign     — доля пар с (x_neg - x_pos)·v_neg > 0
      (*_B — то же для сбалансированных направлений negB/editB/antB)
    """
    rowfold = fold[ps.groups]
    align = {kd: np.empty(len(ps.groups)) for kd in KINDS}
    out = {}
    for suf in ("", "B"):
        correct, total = 0, 0
        scores, labels, scores_pos, labels_pos, signs = [], [], [], [], []
        for i in range(k):
            te = rowfold == i
            w = (fold != i).astype(float)
            V = {kd: ps.direction(kd, w) for kd in KINDS}
            if suf == "":
                for kd in KINDS:
                    align[kd][te] = cos_rows(ps.D[base_of(kd)][te], V[kd][None, :])
            M = np.stack([V["neg" + suf], V["edit" + suf], V["ant" + suf]])
            for c, kd in enumerate(("neg", "edit", "ant")):
                pred = np.argmax(unit(ps.D[kd][te]) @ M.T, axis=1)
                correct += int(np.sum(pred == c))
                total += int(te.sum())
            v = V["neg" + suf]
            for kd in ("pos", "edit", "ant", "neg"):
                s = ps.Xs[kd][te] @ v
                scores.append(s)
                labels.append(np.full(len(s), kd == "neg"))
                if kd in ("pos", "neg"):
                    scores_pos.append(s)
                    labels_pos.append(np.full(len(s), kd == "neg"))
            signs.append(ps.D["neg"][te] @ v > 0)
        out.update({
            "nearest_acc" + suf: correct / total,
            "probe_auc" + suf: float(roc_auc_score(np.concatenate(labels), np.concatenate(scores))),
            "probe_auc_pos" + suf: float(roc_auc_score(np.concatenate(labels_pos), np.concatenate(scores_pos))),
            "pair_sign" + suf: float(np.mean(np.concatenate(signs))),
        })
    out["align"] = {kd: float(np.mean(a)) for kd, a in align.items()}
    out["align_sq"] = {kd: float(np.mean(a ** 2)) for kd, a in align.items()}
    return out


def subset_consistency(ps: PairSet, kind, labels):
    """Средний попарный cos между направлениями, посчитанными по разным подмножествам групп."""
    uniq = sorted(set(labels))
    V = np.stack([ps.direction(kind, (labels == u).astype(float)) for u in uniq])
    C = V @ V.T
    return offdiag_mean(C), C, uniq


def random_partition_consistency(ps: PairSet, kind, labels, n, rng):
    """Нулевое распределение для subset_consistency: те же размеры подмножеств, группы перемешаны."""
    out = []
    for _ in range(n):
        out.append(subset_consistency(ps, kind, rng.permutation(labels))[0])
    return np.array(out)
