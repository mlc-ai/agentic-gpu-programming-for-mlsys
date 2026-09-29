PYTHON ?= .venv/bin/python
PORT ?= 8080
BIND ?= 127.0.0.1

.PHONY: html serve
html:
	$(PYTHON) -m sphinx -E -a -n -W --keep-going -b html -d _build/doctrees . _build/html

serve: html
	$(PYTHON) -m http.server --bind $(BIND) --directory _build/html $(PORT)
