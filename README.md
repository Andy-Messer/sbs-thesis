# sbs-thesis

Магистерская диссертация. Вёрстка — LaTeX по шаблону кафедры космической
физики МФТИ (`mipt-thesis-ms`), язык — русский.

## Сборка

Нужен дистрибутив LaTeX (TeX Live / MiKTeX) с `pdflatex` и `biber`.

```sh
make            # pdflatex -> biber -> pdflatex x2, на выходе main.pdf
make clean      # удалить промежуточные артефакты
make watch      # авто-пересборка через latexmk (если установлен)
```

## Структура

- `main.tex` — корневой файл (преамбула + `\input` разделов).
- `preamble.tex` — собственные макросы и обозначения.
- `sections/` — разделы, по файлу на главу.
- `bibliography.bib` — библиография (biblatex/biber).
- `figures/` — рисунки.
- `code/` — код экспериментов (Python).
- `CONVENTIONS.md` — правила написания.

Шаблонные файлы `mipt-thesis-*.cls/.sty/.bbx` взяты из
<https://github.com/yl3dy/mipt-thesis> (MIT, см. `LICENSE-mipt-thesis`).
