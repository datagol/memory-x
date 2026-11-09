# Build the Memory-X technical report (xelatex; Manrope is bundled in fonts/ under the SIL OFL).
TEX := memory-x-technical-report

all: $(TEX).pdf

figures:
	cd ../../.. && uv run --no-project --with matplotlib --with "psycopg[binary]" --with numpy python docs/paper/tech-report/make_figures.py

$(TEX).pdf: $(TEX).tex $(wildcard *.tex) $(wildcard figures/*.pdf)
	xelatex -interaction=nonstopmode -halt-on-error $(TEX).tex >/dev/null
	xelatex -interaction=nonstopmode -halt-on-error $(TEX).tex >/dev/null
	@grep -E "Warning.*(undefined|Overfull)" $(TEX).log || true

clean:
	rm -f $(TEX).aux $(TEX).log $(TEX).out $(TEX).toc

.PHONY: all figures clean
