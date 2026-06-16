"""Конвертер LaTeX-главы в плоские блоки для виджета инлайн-комментариев.

Читает один sections/0X-*.tex и печатает JSON-список блоков вида
{id, line, kind, text}, где line -- номер строки начала блока в файле
(чтобы комментарий из виджета мапился ровно в file:line), kind --
'h1'|'h2'|'h3'|'p'|'li'|'eq'|'todo'.
LaTeX-разметка упрощается до читабельного текста.
"""
import sys
import re
import json
from pathlib import Path


def strip_inline(s: str) -> str:
    s = re.sub(r"\\(textbf|emph|textit|texttt|mathrm|mathit)\{([^{}]*)\}", r"\2", s)
    s = re.sub(r"\\eqref\{[^}]*\}", "(ф-ла)", s)
    s = re.sub(r"\\ref\{[^}]*\}", "§", s)
    s = re.sub(r"\\cite\{[^}]*\}", "[ист.]", s)
    s = re.sub(r"\\label\{[^}]*\}", "", s)
    s = re.sub(r"\\(begin|end)\{[^}]*\}", "", s)
    s = re.sub(r"\\item\b", "", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", "", s)        # прочие команды
    s = s.replace("\\\\", " ").replace("~", " ")
    s = s.replace("{", "").replace("}", "")
    s = s.replace("--", "—").replace("\\%", "%").replace("\\&", "&")
    s = s.replace("<<", "«").replace(">>", "»")
    s = re.sub(r"\$[^$]*\$", lambda m: "·", s) if False else s
    s = re.sub(r"\s+", " ", s).strip()
    return s


def convert(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks = []
    buf = []
    buf_line = 0
    buf_kind = "p"
    in_eq = False
    eq_line = 0

    def flush():
        nonlocal buf, buf_line, buf_kind
        if buf:
            txt = strip_inline(" ".join(buf))
            if txt:
                blocks.append({"line": buf_line, "kind": buf_kind, "text": txt})
        buf = []
        buf_kind = "p"

    for i, raw in enumerate(lines, start=1):
        line = raw.rstrip()
        st = line.strip()
        if in_eq:
            if re.match(r"\\end\{(equation|align|equation\*|align\*)\}", st):
                in_eq = False
                blocks.append({"line": eq_line, "kind": "eq",
                               "text": "[формула — правится в .tex]"})
            continue
        if re.match(r"\\begin\{(equation|align|equation\*|align\*)\}", st):
            flush(); in_eq = True; eq_line = i; continue
        if st.startswith("% TODO") or st.startswith("%TODO"):
            flush()
            blocks.append({"line": i, "kind": "todo",
                           "text": strip_inline(st.lstrip("% "))})
            continue
        if st.startswith("%"):
            continue
        m = re.match(r"\\chapter\{(.+)\}", st)
        if m:
            flush(); blocks.append({"line": i, "kind": "h1", "text": strip_inline(m.group(1))}); continue
        m = re.match(r"\\section\{(.+)\}", st)
        if m:
            flush(); blocks.append({"line": i, "kind": "h2", "text": strip_inline(m.group(1))}); continue
        m = re.match(r"\\subsection\{(.+)\}", st)
        if m:
            flush(); blocks.append({"line": i, "kind": "h3", "text": strip_inline(m.group(1))}); continue
        if re.match(r"\\(label|begin\{itemize\}|end\{itemize\}|begin\{enumerate\}|end\{enumerate\}|centering|begin\{table\}|end\{table\}|begin\{tabular\}|end\{tabular\}|begin\{figure\}|end\{figure\}|toprule|midrule|bottomrule|caption|includegraphics)", st):
            # таблицы/рисунки/окружения — помечаем один раз, тело пропускаем грубо
            if st.startswith("\\caption"):
                flush()
                cap = re.match(r"\\caption\{(.+)\}", st)
                if cap:
                    blocks.append({"line": i, "kind": "cap", "text": "Подпись: " + strip_inline(cap.group(1))})
            continue
        if st.startswith("\\item"):
            flush()
            buf_line = i
            buf_kind = "li"
            buf.append(st)
            continue
        if st == "":
            flush(); continue
        # строка таблицы / служебное — пропустим если похоже на &...\\
        if "&" in st and st.endswith("\\\\"):
            continue
        if not buf:
            buf_line = i
        buf.append(st)
    flush()

    # выкинуть пустые li
    blocks = [b for b in blocks if b["text"]]
    for j, b in enumerate(blocks):
        b["id"] = j
    return blocks


if __name__ == "__main__":
    p = Path(sys.argv[1])
    print(json.dumps(convert(p), ensure_ascii=False))
