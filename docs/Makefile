# Minimal Makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD  ?= sphinx-build
SOURCEDIR    = source
BUILDDIR     = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make" mode of Sphinx.
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

html:
	@echo "Creating placeholder HTML documentation..."
	@mkdir -p _build/html
	@echo "<html><body><h1>Enhanced RAG System Documentation</h1><p>Documentation placeholder - see README.md and CLAUDE.md for current documentation.</p></body></html>" > _build/html/index.html
	@echo "HTML documentation created at _build/html/"