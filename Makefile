# TileGen one-command pipeline.
#
#   make all BUDGET=6   -> setup, matrices, C++ build, tests, verify, tune, report
#
# The default BUDGET is small so `make all` finishes quickly on the CPU backend.

PYTHON      ?= python3
CMAKE       ?= $(shell if command -v cmake >/dev/null 2>&1; then command -v cmake; \
                        elif [ -x /opt/homebrew/bin/cmake ]; then echo /opt/homebrew/bin/cmake; \
                        else echo cmake; fi)
BUDGET      ?= 6
BUILD_DIR   ?= build
CPP_BUILD   ?= $(BUILD_DIR)/cpp
RESULTS     ?= results
MATRIX_DIR  ?= data/matrices

# Homebrew installs cmake on macOS outside the default PATH.
export PATH := /opt/homebrew/bin:$(PATH)

.PHONY: all setup matrices build test test-python test-cpp verify tune report clean help

all: setup matrices build test verify tune report
	@echo "== TileGen pipeline complete =="

help:
	@echo "targets: setup matrices build test verify tune report all clean"

setup:
	$(PYTHON) -m pip install -e ".[dev]" \
	  || $(PYTHON) -m pip install ".[dev]" \
	  || echo "[warn] pip install skipped; using the in-tree tilegen package"

matrices:
	$(PYTHON) -m tilegen.matrices --out $(MATRIX_DIR)

build:
	$(CMAKE) -S cpp -B $(CPP_BUILD) -DCMAKE_BUILD_TYPE=Release
	$(CMAKE) --build $(CPP_BUILD) -j

test: test-cpp test-python

test-cpp:
	$(CMAKE) -S cpp -B $(CPP_BUILD) -DCMAKE_BUILD_TYPE=Release >/dev/null
	$(CMAKE) --build $(CPP_BUILD) -j --target tilegen_tests
	./$(CPP_BUILD)/tilegen_tests

test-python:
	$(PYTHON) -m pytest -q

verify:
	$(PYTHON) scripts/verify_all.py

tune:
	$(PYTHON) scripts/autotune.py --budget $(BUDGET) --out $(RESULTS)

report:
	$(PYTHON) scripts/analyze.py

clean:
	rm -rf $(BUILD_DIR) $(RESULTS)
	rm -f $(MATRIX_DIR)/*.npz $(MATRIX_DIR)/*.bin
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
