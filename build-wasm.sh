#!/bin/bash
set -e

# Usage: ./build-wasm.sh
#
# Builds the Tuun synthesizer for WebAssembly.

echo "🔧 Building Tuun for WebAssembly..."

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

wasm-pack build --target web --out-dir web/pkg -- --no-default-features --features wasm

echo
echo "✅ Build complete! Output in web/pkg/"
echo
echo "To test locally:"
echo "  (cd web && python3 -m http.server 8080)"
echo "  Then open http://localhost:8080 in your browser"
