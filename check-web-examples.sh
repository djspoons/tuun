#!/bin/bash
set -e

# Checks all tuun-synth expressions in .md and .html files.
# Usage: ./check-web.sh [root]

ROOT=${1-docs}

FILES=$(find ${ROOT} \
  -path "${ROOT}/vendor" -prune -o \
  -path "${ROOT}/_site" -prune -o \
  \( -name '*.md' -o -name '*.html' \) -print | sort)

cargo run --bin web_checker -- $FILES
