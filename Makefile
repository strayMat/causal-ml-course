.PHONY: install
install: ## Install the virtual environment
	@echo "🚀 Creating virtual environment"
	@uv sync

.PHONY: docs-clean
docs-clean: ## Clean the documentation
	rm -rf docs/_build

.PHONY: docs-html
docs-html: ## Build and serve the documentation
	uv run jupytext notebooks/*.py --to ipynb 
	mkdir -p docs/source/_static/slides
	cp slides/biblio.bib slides/*.pdf docs/source/_static/slides/
	@uv run sphinx-build -b html docs/source docs/_build
