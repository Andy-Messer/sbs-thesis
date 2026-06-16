# План экспериментов: LLM-as-a-Judge

## Статус на 2026-06-15

### Завершённые

| ID | Эксперимент | Данные | Результат |
|---|---|---|---|
| T1 | Faithful judge (N=100) | votes_faithful.jsonl (1200) | 52% согласия, SbS=-16%, TOST ✗ |
| T1 | Improve judge (N=100) | votes_improve.jsonl (300) | best: consensus l=4, \|Δ\|=0 |
| T2 | Multimodal variance | variance.json | σ₁=1.592, σ/√n подтверждена |
| T3 | Текст главы 5 | 05-results.tex | числа подставлены, графики раскомментированы |

### Частично выполнено

| ID | Эксперимент | Статус | Осталось |
|---|---|---|---|
| T4 | Massive val set v4 | massive_val_v4.json: 1460 запросов с ассессорами | извлечение ответов: 200/1460 |

---

## Шаг 1: Massive Val Set (N=1460)

**Цель:** Снизить MDE с 20% до 3-5% за счёт увеличения выборки.

**Данные:**
- v4 баскет: 1460 запросов с 10 ассессорами каждый
- Вердикты: draw=37%, gpt_win=33%, ydx_win=29% (сбалансировано)
- Файлы: `code/baskets/v4/massive_val_v4.json`, `massive_val_cache.json`

**Задачи:**
1. Достроить извлечение ответов: 200 → 1460 (`extract_v4_answers.py`, ~25 мин)
2. Faithful judge на N=1460: 4 модели × 3 промпта = 17520 вызовов (~12 ч)
3. Анализ: bootstrap CI, TOST, MDE = f(N, p̂)

**MDE расчёт:**
```
N=1460, p̂=0.5, α=0.05, β=0.2
MDE ≈ (1.96 + 0.84) × √(0.5×0.5/1460) ≈ 3.3%
```

---

## Шаг 2: Ablation Study

**Цель:** Доказать, что multi-prompt/consensus работают только ПОДАВ смещение.

**4 замера на massive val set (N=1460):**

| Замер | Модель | Промпт | Агрегация | Ожидание |
|---|---|---|---|---|
| A: Baseline | qwen2.5:7b | Winner | — | ~30% |
| B: Anti-bias | qwen2.5:7b | Anti-bias (чек-лист + штраф за длину) | — | ~50% |
| C: +Multi-prompt | qwen2.5:7b | 3 anti-bias варианта | k=3 | ~65% |
| D: +Consensus | qwen2.5:7b + qwen3:1.7b + qwen2.5:3b | 3×3 | l=3, m=3 | ~75-80% |

**Скрипт:** `code/experiments/ablation_study.py`

**Вывод для защиты:** Консенсус и multi-prompt работают как второй этап, поверх откалиброванной модели. Без калибровки агрегация вредна.

---

## Шаг 3: Capacity Test

**Цель:** Доказать, что бесплатные модели упираются в «потолок рассуждений».

**Дизайн:** A/B тест на massive val set:

| Модель | Тип | Стоимость |
|---|---|---|
| qwen2.5vl:7b | Open-source | ~0 |
| GPT-4o | Proprietary | ~$0.005/1K tokens |
| Claude 3.5 Sonnet | Proprietary | ~$0.003/1K tokens |

**Что сравниваем:**
- Coverage (% валидных вердиктов)
- Agreement с асессорами
- F1, precision, recall
- Latency, cost per verdict

**Скрипты:**
- `code/experiments/proprietary_judge.py` (OpenAI/Anthropic API)
- `code/experiments/capacity_test.py` (сравнение)

**Ожидаемый вывод:** 1 вызов GPT-4o (1×) бьёт consensus из 12 вызовов qwen2.5:7b (12×) по F1.

---

## Шаг 4: Dynamic Few-Shot (RAG for Judge)

**Цель:** «Дотянуть» бесплатную модель через retrieval augmented generation.

**Архитектура:**
1. Эмбеддинги Golden Set → векторная БД (FAISS / ChromaDB)
2. На входе нового запроса: retrieval 3 ближайших примеров
3. Few-shot промпт: «Вот как асессоры разметили похожие случаи: ...»

**Скрипты:**
- `code/experiments/build_embeddings.py` (эмбеддинги запросов)
- `code/experiments/dynamic_fewshot_judge.py` (retrieval + инжект + judge)

**Сравнение:** static few-shot (из train) vs dynamic few-shot (RAG) vs baseline.

---

## Порядок выполнения

```
Шаг 1 (данные) → Шаг 2 (ablation) → Шаг 3 (capacity) → Шаг 4 (RAG)
     ↓                  ↓                    ↓                  ↓
  extract →         ablation.py       proprietary_judge.py  dynamic_fewshot.py
  faithful.py
  analyze.py
```

**Шаг 1 — критический путь.** Все остальные шаги зависят от massive val set.

---

## Текущие ограничения

1. **N=50 на test** — недостаточно для TOST (MDE=20%)
2. **Без API ключей** — Capacity Test (Шаг 3) требует OpenAI/Anthropic API
3. **Только текстовый judge** — мультимодальный judge (скриншоты) не масштабирован
4. **Verbosity bias не устранён** — агрегация усиливает, а не гасит смещение

---

## Файлы экспериментов

```
code/experiments/
├── staged_results/
│   ├── votes_faithful.jsonl      # 1200 голосов (faithful judge)
│   ├── votes_improve.jsonl       # 300 голосов (improve judge)
│   ├── analysis_faithful.json    # итоговый анализ faithful
│   ├── improve.json              # анализ итеративных улучшений
│   ├── variance.json             # снижение дисперсии
│   └── cache_v2.json             # извлечённые ответы (100)
├── faithful_judge_run.py         # честный judge (ассессорский протокол)
├── improve_judge.py              # итеративные улучшения
├── analyze_faithful.py           # анализ faithful
├── analyze_improve.py            # анализ improve + графики
├── multimodal_variance.py        # снижение дисперсии
├── make_result_charts.py         # графики главы 5
└── staged_sbs_v2.py              # извлечение ответов (Playwright)

code/baskets/v4/
├── massive_val_v4.json           # 1460 запросов с ассессорами
├── massive_val_cache.json        # извлечённые ответы (200/1460)
├── build_massive_val_set.py      # мерж пулов → massive_val_v4.json
└── extract_v4_answers.py         # Playwright извлечение

figures/
├── improve-ladder.jpeg           # лестница улучшений
├── improve-delta.jpeg            # |Δ SbS-NPS| по итерациям
├── improve-percrit.jpeg          # покритериальное согласие
├── variance-reduction.jpeg       # σ/√n
├── results-sbs-stages.jpeg       # SbS-NPS по схемам
├── results-confusion.jpeg        # матрица согласованности
├── results-nps-density.jpeg      # bootstrap-плотности
├── results-pareto.jpeg           # согласие vs стоимость
└── pilot-multimodal-load.jpeg    # загрузка CPU/RAM
```
