# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
PACKAGEDIR    = $(SOURCEDIR)/../..
EXAMPLEDIR    = $(SOURCEDIR)/_static/examples

# OS-specific variables
SED_INPLACE := $(shell [ "$(shell uname)" = "Darwin" ] && echo "sed -i ''" || echo "sed -i")

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help $(SOURCEDIR) $(BUILDDIR) $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	# Create LICENSE.md with a proper title
	(echo "# License\n" && cat $(PACKAGEDIR)/LICENSE.txt) > $(SOURCEDIR)/LICENSE.md

	# Render the README file with Quarto
	quarto render $(SOURCEDIR)/_static/README.qmd -t gfm --output-dir ../

	# Prepare README.md for the root directory
	cp $(SOURCEDIR)/README.md ../README.md
	$(SED_INPLACE) 's|<a href="LICENSE.html">MIT license</a>|[MIT license](LICENSE.txt)|g' ../README.md

	# Prepare README.md for Sphinx HTML documentation
	cp $(SOURCEDIR)/README.md $(SOURCEDIR)/README_sphinx.md
	$(SED_INPLACE) 's|\[MIT license\](LICENSE.txt)|<a href="LICENSE.html">MIT license</a>|g' $(SOURCEDIR)/README_sphinx.md

	# Ensure LICENSE.html exists in Sphinx output
	mkdir -p $(BUILDDIR)/html
	pandoc $(SOURCEDIR)/LICENSE.md -o $(BUILDDIR)/html/LICENSE.html

	# Process Quarto examples
	for path in $(EXAMPLEDIR)/*.qmd; do \
		file=$$(basename "$$path"); \
		quarto render "$(EXAMPLEDIR)/$${file}" -t gfm; \
		mv "$(EXAMPLEDIR)/$${file%.qmd}.md" "$(SOURCEDIR)/$${file%.qmd}.md"; \
		$(SED_INPLACE) "s/$${file%.qmd}_files/_static\/examples\/$${file%.qmd}_files/g" \
			"$(SOURCEDIR)/$${file%.qmd}.md"; \
	done

	# Build the Sphinx documentation
	@$(SPHINXBUILD) -M $@ $(SOURCEDIR) $(BUILDDIR) $(SPHINXOPTS) $(O)
