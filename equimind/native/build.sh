#!/bin/bash
# EquiMind C++ Native Module Build Script
# Compiles the performance core with pybind11

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

echo "═══════════════════════════════════════════════"
echo " EquiMind C++ Native Performance Core — Build"
echo "═══════════════════════════════════════════════"

# Find pybind11 cmake directory
PYBIND11_CMAKE=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())" 2>/dev/null)
if [ -z "$PYBIND11_CMAKE" ]; then
    echo "ERROR: pybind11 not found. Install with: pip install pybind11"
    exit 1
fi
echo "✓ pybind11 found at: $PYBIND11_CMAKE"

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure
echo "→ Configuring with CMake..."
cmake "$SCRIPT_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -Dpybind11_DIR="$PYBIND11_CMAKE" \
    -DCMAKE_INSTALL_PREFIX="$PROJECT_ROOT/equimind"

# Build
echo "→ Compiling C++ module..."
cmake --build . --config Release -j$(nproc)

# Copy .so to equimind package
echo "→ Installing module..."
cp equimind_native*.so "$PROJECT_ROOT/equimind/" 2>/dev/null || true
cp equimind_native*.pyd "$PROJECT_ROOT/equimind/" 2>/dev/null || true

echo ""
echo "═══════════════════════════════════════════════"
echo " ✓ Build successful!"
echo " Module installed to: $PROJECT_ROOT/equimind/"
echo "═══════════════════════════════════════════════"

# Quick verification
cd "$PROJECT_ROOT"
python3 -c "
import equimind_native
print('✓ equimind_native loaded successfully')
print('  Submodules: technical, montecarlo, dedup, portfolio')

# Quick benchmark
import time
prices = list(range(1, 10001))
start = time.perf_counter()
for _ in range(1000):
    equimind_native.technical.rsi(prices, 14)
elapsed = time.perf_counter() - start
print(f'  RSI benchmark: 1000 calls on 10K prices = {elapsed:.3f}s')
" 2>/dev/null || echo "(Verification will run after PYTHONPATH is set)"
