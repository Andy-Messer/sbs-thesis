# Сборка магистерской диссертации (шаблон mipt-thesis-ms, pdfLaTeX + biber).
# Класс шаблона рассчитан на pdflatex (T2A/inputenc), не XeLaTeX.

MAIN = main
LATEX = pdflatex -interaction=nonstopmode -halt-on-error
BIBER = biber

.PHONY: all clean distclean watch

all: $(MAIN).pdf

# Полная сборка с библиографией: latex -> biber -> latex -> latex
$(MAIN).pdf: $(MAIN).tex preamble.tex bibliography.bib $(wildcard sections/*.tex)
	$(LATEX) $(MAIN)
	$(BIBER) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

# Если установлен latexmk — удобнее так (сам считает число проходов):
#   latexmk -pdf -interaction=nonstopmode main.tex
watch:
	latexmk -pdf -pvc -interaction=nonstopmode $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).toc \
	      $(MAIN).bbl $(MAIN).bcf $(MAIN).blg $(MAIN).run.xml \
	      $(MAIN).lof $(MAIN).lot sections/*.aux

distclean: clean
	rm -f $(MAIN).pdf
