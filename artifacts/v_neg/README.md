# Найденные направления (DiffMean)

Один файл `{model}_{lang}.npz` на модель и язык (ключи моделей — `src/common.py`, `MODELS`).

| массив | форма | что это |
|---|---|---|
| `layers` | (K,) | подписи строк: номер слоя энкодера (`0` = выход эмбеддинг-слоя) и `final` |
| `v_neg` | (K, d) | направление негации: unit(mean(x(отрицание) − x(утверждение))) по всем 240 парам |
| `v_negB` | (K, d) | то же, но позитивные, негативные и нейтральные утверждения с равным весом (рекомендуется, если нужна «негация без тональности») |
| `v_edit`, `v_editB` | (K, d) | контроль: вставка нейтрального наречия в то же место |
| `v_ant`, `v_antB` | (K, d) | контроль: замена на антоним |
| `v_rp` | (K, d) | контроль: разность случайных утверждений разных тем (направления не имеет) |

Все векторы единичной длины и посчитаны на **L2-нормированных** представлениях: представление
слоя = пулинг модели (mean или CLS) от hidden_states, `final` = выход `SentenceTransformer.encode`.
Для e5 / multilingual-e5 / USER-base тексты подавались с префиксом `query: `.
У моделей с CLS-пулингом (bge, bge-m3, rubert-tiny2) слой 0 выброшен: [CLS] до первого слоя
одинаков для всех предложений.

Пример — признак «разница проекций» из H4 (метод M2):

```python
import numpy as np
from sentence_transformers import SentenceTransformer

z = np.load("artifacts/v_neg/mpnet_en.npz")
v = z["v_neg"][list(z["layers"]).index("final")]
m = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
a, c = m.encode(["The store is open.", "The store is not open."], normalize_embeddings=True)
lam = 2.0  # калибровать на своих данных; в H4 подбирался кросс-валидацией (results/h4_lambda.csv)
score = a @ c - lam * abs(a @ v - c @ v)
```

Направления получены на 48 шаблонных утверждениях на язык (× 5 рамок) и проверялись на
отложенных данных, но это не «универсальная ось негации». Ограничения описаны в `FINAL_REPORT.md`.
