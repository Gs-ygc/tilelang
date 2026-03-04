#!/bin/bash
# Quick rebuild and test script for development iteration
# Usage: ./quick_test.sh

set -e

cd /home/gs2ygc/code/xsai-env/tilelang

# Quick rebuild
echo "🔨 Rebuilding..."
cd build
ninja

# Set PYTHONPATH
cd ..
export PYTHONPATH=$PWD:$PYTHONPATH

# Quick test
echo -e "\n🧪 Running quick tests..."
python3 -c "
import tilelang
from tilelang.utils.target import SUPPORTED_TARGETS
print('✓ TileLang version:', tilelang.__version__)
print('✓ RISCV AME supported:', 'riscv_ame' in SUPPORTED_TARGETS)
"

echo -e "\n✅ Quick test passed!"
echo "Run full tests with: python3 testing/python/test_riscv_ame_backend.py"
