# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Translation targets
.PHONY: translate-update translate-build

# All languages with built-in Sphinx search support + Chinese and Hindi
# List: English, Arabic, Danish, Dutch, Finnish, French, German, Hungarian, 
# Italian, Japanese, Norwegian, Portuguese, Romanian, Russian, Spanish, 
# Swedish, Turkish, Chinese, Hindi.
ALL_LANGS = en ar da nl fi fr de hu it ja no pt ro ru es sv tr zh hi

LANGS ?= $(ALL_LANGS)

translate-update:
	$(SPHINXBUILD) -M gettext "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	sphinx-intl update -p "$(BUILDDIR)/gettext" -l $(LANGS)

translate-build:
	@for lang in $(LANGS); do \
		$(SPHINXBUILD) -b html -D language=$$lang "$(SOURCEDIR)" "$(BUILDDIR)/html/$$lang" $(SPHINXOPTS) $(O); \
	done
