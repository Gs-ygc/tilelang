#!/bin/bash
# Development Build and Test Script for TileLang RISCV AME Backend
# Usage: ./dev_build_test.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}TileLang RISCV AME Dev Build${NC}"
echo -e "${GREEN}================================${NC}"

# 1. Setup environment
echo -e "\n${YELLOW}[1/6] Setting up environment...${NC}"
cd /home/gs2ygc/code/xsai-env
source env.sh

# Verify environment
if [ -z "$XS_PROJECT_ROOT" ]; then
    echo -e "${RED}Error: XS_PROJECT_ROOT not set${NC}"
    exit 1
fi

echo "✓ XS_PROJECT_ROOT: $XS_PROJECT_ROOT"

# Check LLVM-AME
if [ -f "$XS_PROJECT_ROOT/local/llvm/bin/clang" ]; then
    echo "✓ LLVM-AME found at: $XS_PROJECT_ROOT/local/llvm"
    export LLVM_AME_PATH=$XS_PROJECT_ROOT/local/llvm
else
    echo -e "${YELLOW}⚠ LLVM-AME not found. Building it first...${NC}"
    cd $XS_PROJECT_ROOT
    make llvm
    export LLVM_AME_PATH=$XS_PROJECT_ROOT/local/llvm
fi

# 2. Clean previous build (optional)
echo -e "\n${YELLOW}[2/6] Cleaning previous build...${NC}"
cd $XS_PROJECT_ROOT/DSL/tilelang
if [ -d "build" ]; then
    echo "Removing old build directory..."
    rm -rf build
fi

# 3. Configure with CMake
echo -e "\n${YELLOW}[3/6] Configuring with CMake...${NC}"
mkdir -p build
cd build
cmake .. \
    -DUSE_CUDA=OFF \
    -DUSE_RISCV_AME=ON \
    -DCMAKE_BUILD_TYPE=Debug \
    -G Ninja \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ CMake configuration successful${NC}"
else
    echo -e "${RED}✗ CMake configuration failed${NC}"
    exit 1
fi

# 4. Build
echo -e "\n${YELLOW}[4/6] Building TileLang...${NC}"
ninja -j$(nproc)

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Build successful${NC}"
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi

# 5. Setup PYTHONPATH for development
echo -e "\n${YELLOW}[5/6] Setting up PYTHONPATH...${NC}"
cd ..
export PYTHONPATH=$PWD:$PYTHONPATH
echo "✓ PYTHONPATH set to: $PYTHONPATH"

# 6. Run tests
echo -e "\n${YELLOW}[6/6] Running tests...${NC}"

echo -e "\n${YELLOW}Test 1: Import TileLang${NC}"
python3 -c "import tilelang; print('✓ TileLang version:', tilelang.__version__)"

echo -e "\n${YELLOW}Test 2: Check RISCV AME target${NC}"
python3 -c "
from tilelang.utils.target import SUPPORTED_TARGETS
if 'riscv_ame' in SUPPORTED_TARGETS:
    print('✓ RISCV AME target is supported')
    print('  Description:', SUPPORTED_TARGETS['riscv_ame'])
else:
    print('✗ RISCV AME target not found')
    exit(1)
"

echo -e "\n${YELLOW}Test 3: LLVM-AME wrapper${NC}"
python3 testing/python/test_riscv_ame_backend.py::test_llvm_ame_compiler_wrapper || true

echo -e "\n${YELLOW}Test 4: Code generation${NC}"
python3 testing/python/test_riscv_ame_backend.py::test_riscv_ame_simple_kernel_codegen || true

echo -e "\n${YELLOW}Test 5: Run example${NC}"
python3 examples/riscv_ame/example_matmul.py || true

# Summary
echo -e "\n${GREEN}================================${NC}"
echo -e "${GREEN}Build and Test Complete!${NC}"
echo -e "${GREEN}================================${NC}"
echo -e "\nDevelopment environment ready. You can now:"
echo -e "  1. Edit code in: $XS_PROJECT_ROOT/tilelang"
echo -e "  2. Rebuild with: cd build && ninja"
echo -e "  3. Test with: PYTHONPATH=$XS_PROJECT_ROOT/tilelang:\$PYTHONPATH python3 <test_file>"
echo -e "\n${YELLOW}Quick test command:${NC}"
echo -e "  cd $XS_PROJECT_ROOT/tilelang"
echo -e "  PYTHONPATH=\$PWD:\$PYTHONPATH python3 examples/riscv_ame/example_matmul.py"
