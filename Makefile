include .env
export

.PHONY: help report

PKG=fake_faces

TRAIN_PATH=$(FAKE_FACES_DIR)/real-vs-fake/train
TEST_PATH=$(FAKE_FACES_DIR)/real-vs-fake/test
VALID_PATH=$(FAKE_FACES_DIR)/real-vs-fake/valid

#################################################################################
# Project Commands                                                              #
#################################################################################

## Compile the project report PDF
report: report.pdf

report.pdf: report.tex report.bib
	pdflatex -interaction=nonstopmode $<
	bibtex report.aux
	pdflatex -interaction=nonstopmode $<
	pdflatex -interaction=nonstopmode $<

## align images
align:
	fake-faces align-all $(TRAIN_PATH) $(FAKE_FACES_DIR)/aligned/train/ --num_threads 4
	fake-faces align-all $(VALID_PATH) $(FAKE_FACES_DIR)/aligned/valid/ --num_threads 4
	fake-faces align-all $(TEST_PATH) $(FAKE_FACES_DIR)/aligned/test/ --num_threads 4

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
#	* save line in hold space
#	* purge line
#	* Loop:
#		* append newline + line to hold space
#		* go to next line
#		* if line starts with doc comment, strip comment character off and loop
#	* remove target prerequisites
#	* append hold space (+ newline) to line
#	* replace newline plus comments by `---`
#	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
