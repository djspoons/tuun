#!/bin/bash
set -e

# Usage: ./build-wasm.sh [context_file]
#
# Builds the Tuun synthesizer for WebAssembly.
# Optionally specify a custom .tuun context file to embed.
# If not specified, uses context.tuun from the repository root.
#
# The context file is embedded at compile time and provides
# all the function definitions (like $, Qw, Hw, etc.) that
# will be available in the web version.

CONTEXT_FILE="${1:-context.tuun}"

echo "🔧 Building Tuun for WebAssembly..."
echo "📄 Using context file: $CONTEXT_FILE"
echo

# Check if context file exists
if [ ! -f "$CONTEXT_FILE" ]; then
    echo "❌ Context file not found: $CONTEXT_FILE"
    echo "Usage: $0 [context_file.tuun]"
    exit 1
fi

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "❌ wasm-pack is not installed"
    echo "Install it with: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh"
    exit 1
fi

# Install wasm32 target if needed
echo "📦 Ensuring wasm32-unknown-unknown target is installed..."
rustup target add wasm32-unknown-unknown

echo
echo "🔨 Building with wasm-pack..."
echo "   Context: $CONTEXT_FILE will be embedded in the WASM binary"

# If a custom context file is specified, create a temporary symbolic link
# so that include_str! can find it at the expected location
if [ "$CONTEXT_FILE" != "context.tuun" ]; then
    echo "   Creating temporary link for custom context file..."
    mv context.tuun context.tuun.bak 2>/dev/null || true
    ln -sf "$(pwd)/$CONTEXT_FILE" context.tuun
    trap 'rm -f context.tuun; mv context.tuun.bak context.tuun 2>/dev/null || true' EXIT
fi

wasm-pack build --target web --out-dir web/pkg -- --no-default-features --features wasm

echo
echo "✅ Build complete! Output in web/pkg/"
echo "   Embedded context: $CONTEXT_FILE"
echo
echo "To test locally:"
echo "  cd web && python3 -m http.server 8080"
echo "  Then open http://localhost:8080 in your browser"
