"""Шаг 0. Сборка всех наборов данных из src/lexicon.py (+ STS-B для «естественного» корпуса).

Выход (data/):
  pairs_{lang}.jsonl    — основной набор: 48 утверждений x 5 рамок = 240 строк; в каждой
                          pos / neg (отрицание) / edit (контроль 1: наречие) / ant (контроль 3: антоним)
                          и rp_partner (контроль 2: утверждение другой темы с той же рамкой)
  topics_{lang}.jsonl   — корпус для топик-направлений (контроль 4 / H3)
  h4_{lang}.jsonl       — отложенные тройки (anchor, positive, negative) для H4, 4 типа
  natural_{lang}.jsonl  — предложения STS-B (train) с флагом наличия отрицания
  dataset_stats.json    — размеры и проверки (пересечения, отсутствие отрицаний где не надо)
"""
import json
import random
import re
from pathlib import Path

from lexicon import (ADVERBS_EN, ADVERBS_RU, BASES_EN, BASES_RU, FRAMES_EN, FRAMES_RU,
                     H4_EN, H4_RU, TOPIC_CORPUS_EN, TOPIC_CORPUS_RU, TOPICS)

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

NEG_RE = {
    "en": re.compile(r"\b(not|no|never|nobody|nothing|none|neither|nor|cannot)\b|n't\b", re.I),
    "ru": re.compile(r"(?<![\w-])(не|нет|ни|никогда|ничего|никто|нигде|никак|ничто)(?![\w-])", re.I),
}
EN_AUX = re.compile(r"\^(\w+)")
EN_DO = re.compile(r"\{([^|}]+)\|([^|}]+)\|([^|}]+)\}")
RU_MARK = re.compile(r"\^(\S+)")


def realize(template: str, lang: str, mode: str, adv: str | None = None) -> str:
    """mode: pos | neg | edit. Ровно одна метка на шаблон."""
    if lang == "en":
        m_aux, m_do = EN_AUX.search(template), EN_DO.search(template)
        assert (m_aux is None) != (m_do is None), template
        if m_aux:
            aux = m_aux.group(1)
            rep = {"pos": aux, "neg": f"{aux} not", "edit": f"{aux} {adv}"}[mode]
            return template[:m_aux.start()] + rep + template[m_aux.end():]
        surface, lemma, do = m_do.groups()
        rep = {"pos": surface, "neg": f"{do} not {lemma}", "edit": f"{adv} {surface}"}[mode]
        return template[:m_do.start()] + rep + template[m_do.end():]
    m = RU_MARK.search(template)
    assert m is not None and len(RU_MARK.findall(template)) == 1, template
    w = m.group(1)
    rep = {"pos": w, "neg": f"не {w}", "edit": f"{adv} {w}"}[mode]
    return template[:m.start()] + rep + template[m.end():]


def construction(template: str, lang: str) -> str:
    if lang == "ru":
        w = RU_MARK.search(template).group(1).lower()
        return "copula" if w in {"был", "была", "было", "были"} else "verb"
    m_aux = EN_AUX.search(template)
    if m_aux:
        a = m_aux.group(1)
        return {"is": "copula", "are": "copula", "was": "copula", "were": "copula"}.get(a, "modal/perfect")
    return "do-support"


def strip_marks(s: str, lang: str) -> str:
    return realize(s, lang, "pos") if ("^" in s or "{" in s) else s


def add_frame(frame: str, sent: str) -> str:
    if not frame:
        return sent
    first = sent.split()[0]
    if first != "I":
        sent = sent[0].lower() + sent[1:]
    return frame + sent


def build_pairs(lang: str, rng: random.Random):
    bases = BASES_EN if lang == "en" else BASES_RU
    frames = FRAMES_EN if lang == "en" else FRAMES_RU
    advs = ADVERBS_EN if lang == "en" else ADVERBS_RU
    rows = []
    for b, (topic, val, tpl, ant) in enumerate(bases):
        for f, frame in enumerate(frames):
            adv = rng.choice(advs)
            rows.append(dict(
                pair_id=len(rows), base_id=b, topic=topic, valence=val,
                construction=construction(tpl, lang), frame_id=f,
                pos=add_frame(frame, realize(tpl, lang, "pos")),
                neg=add_frame(frame, realize(tpl, lang, "neg")),
                edit=add_frame(frame, realize(tpl, lang, "edit", adv)), edit_adv=adv,
                ant=add_frame(frame, strip_marks(ant, lang)),
            ))
    # контроль 2: партнёр — утверждение ДРУГОЙ темы с той же рамкой (случайный, с возвращением)
    # ориентация разности (self->partner или partner->self) случайна: иначе половины выборки
    # анти-коррелированы тривиально (партнёры половины A — это исходные фразы половины B)
    sign_rng = random.Random(SEED + 100)  # отдельный поток: не сдвигает остальные случайные выборки
    for r in rows:
        cand = [q for q in rows if q["topic"] != r["topic"] and q["frame_id"] == r["frame_id"]]
        r["rp_partner"] = rng.choice(cand)["pair_id"]
        r["rp_sign"] = sign_rng.choice([1, -1])
    return rows


def build_h4(lang: str):
    items = H4_EN if lang == "en" else H4_RU
    rows, triples = [], []
    for i, (kind, anchor_t, para, ant_t) in enumerate(items):
        anchor = realize(anchor_t, lang, "pos")
        neg = realize(anchor_t, lang, "neg")
        ant = realize(ant_t, lang, "pos")
        neg_ant = realize(ant_t, lang, "neg")
        rows.append(dict(item_id=i, kind=kind, anchor=anchor, para=para, neg=neg, ant=ant, neg_ant=neg_ant))
        # T1: отрицание как «ловушка» (главный тип из ТЗ), T2: антоним как ловушка
        triples.append(dict(item_id=i, kind=kind, type="T1_neg", anchor=anchor, positive=para, negative=neg))
        triples.append(dict(item_id=i, kind=kind, type="T2_ant", anchor=anchor, positive=para, negative=ant))
        if kind == "comp":
            # T3: правильный ответ сам содержит отрицание ("not closed" == "open")
            triples.append(dict(item_id=i, kind=kind, type="T3_negpara", anchor=anchor, positive=neg_ant, negative=ant))
            # T4: якорь с отрицанием; правильный ответ — антоним без отрицания
            triples.append(dict(item_id=i, kind=kind, type="T4_neganchor", anchor=neg, positive=ant, negative=anchor))
    return rows, triples


def build_natural(lang: str, rng: random.Random, n_total: int = 3000):
    from datasets import load_dataset
    ds = load_dataset("PhilipMay/stsb_multi_mt", lang, split="train")
    sents = sorted({s.strip() for col in ("sentence1", "sentence2") for s in ds[col] if s and s.strip()})
    neg = [s for s in sents if NEG_RE[lang].search(s)]
    non = [s for s in sents if not NEG_RE[lang].search(s)]
    rng.shuffle(non)
    non = non[: max(0, n_total - len(neg))]
    out = [dict(text=s, has_neg=True) for s in neg] + [dict(text=s, has_neg=False) for s in non]
    rng.shuffle(out)
    return out


def words(s: str) -> set[str]:
    return set(re.findall(r"\w+", s.lower()))


def write_jsonl(path: Path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    DATA.mkdir(exist_ok=True)
    stats = {}
    for lang in ("en", "ru"):
        rng = random.Random(SEED + (0 if lang == "en" else 1))
        pairs = build_pairs(lang, rng)
        topics = [dict(topic=t, text=s) for t in TOPICS
                  for s in (TOPIC_CORPUS_EN if lang == "en" else TOPIC_CORPUS_RU)[t]]
        h4_items, h4_triples = build_h4(lang)
        natural = build_natural(lang, rng)

        # --- проверки ---
        neg_re = NEG_RE[lang]
        bad = []
        for r in pairs:
            for k in ("pos", "edit", "ant"):
                if neg_re.search(r[k]):
                    bad.append((k, r[k]))
            if not neg_re.search(r["neg"]):
                bad.append(("neg-missing", r["neg"]))
        for r in topics:
            if neg_re.search(r["text"]):
                bad.append(("topic", r["text"]))
        for r in h4_items:
            for k in ("anchor", "para", "ant"):
                if neg_re.search(r[k]):
                    bad.append((f"h4-{k}", r[k]))
        assert not bad, bad

        extraction_sents = {r[k] for r in pairs for k in ("pos", "neg", "edit", "ant")}
        h4_sents = {r[k] for r in h4_items for k in ("anchor", "para", "neg", "ant", "neg_ant")}
        assert not (extraction_sents & h4_sents), extraction_sents & h4_sents
        ext_words = set().union(*(words(s) for s in extraction_sents))
        # доля «содержательных» слов якорей H4, встречающихся в наборе для извлечения
        stop = {"the", "a", "an", "is", "are", "was", "were", "not", "to", "of", "in", "on", "at", "this",
                "и", "в", "на", "не", "с", "к", "по", "до", "за", "от", "из", "был", "была", "было", "были"}
        h4_words = set().union(*(words(r["anchor"]) for r in h4_items)) - stop
        overlap = sorted(h4_words & ext_words)

        write_jsonl(DATA / f"pairs_{lang}.jsonl", pairs)
        write_jsonl(DATA / f"topics_{lang}.jsonl", topics)
        write_jsonl(DATA / f"h4_{lang}.jsonl", h4_items)
        write_jsonl(DATA / f"h4_triples_{lang}.jsonl", h4_triples)
        write_jsonl(DATA / f"natural_{lang}.jsonl", natural)

        def count(key, rows):
            out = {}
            for r in rows:
                out[r[key]] = out.get(r[key], 0) + 1
            return out

        stats[lang] = dict(
            n_bases=len({r["base_id"] for r in pairs}), n_pairs=len(pairs),
            valence=count("valence", pairs[::5]), construction=count("construction", pairs[::5]),
            topics=count("topic", pairs[::5]), edit_adverbs=count("edit_adv", pairs),
            n_topic_sents=len(topics), n_h4_items=len(h4_items),
            h4_items_by_kind=count("kind", h4_items), h4_triples_by_type=count("type", h4_triples),
            n_natural=len(natural), n_natural_neg=sum(r["has_neg"] for r in natural),
            h4_anchor_content_words=len(h4_words), h4_words_seen_in_extraction=len(overlap),
            h4_words_seen_in_extraction_list=overlap,
        )
        print(lang, json.dumps({k: v for k, v in stats[lang].items() if not k.endswith("_list")},
                               ensure_ascii=False))
        for r in pairs[:2] + pairs[5:7] + pairs[60:61]:
            print("  ", r["pos"], "|", r["neg"], "|", r["edit"], "|", r["ant"])
    with open(DATA / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
