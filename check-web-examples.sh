#!/bin/bash
set -e

# Checks all tuun-synth expressions in .md and .html files.
# Usage: ./check-web.sh

FILES=$(find docs web \
  -path 'docs/vendor' -prune -o \
  -path 'docs/_site' -prune -o \
  \( -name '*.md' -o -name '*.html' \) -print | sort)

cargo run --bin web_checker -- $FILES
