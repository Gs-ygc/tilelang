#!/usr/bin/env python3
"""
Test to see if pattern matching is working in AME code generation
"""

import os
import tempfile
os.environ["CXX"] = "/home/gs2ygc/code/xsai-env/local/llvm/bin/clang++"
os.environ["CC"] = "/home/gs2ygc/code/xsai-env/local/llvm/bin/clang"

import tilelang
import tilelang.language as T

def matmul_ame_func(M, N, K, dtype=T.float32):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, 1, threads=1) as (bx, by):
            A_tile = T.alloc_local((M, K), dtype)
            B_tile = T.alloc_local((K, N), dtype)
            C_tile = T.alloc_local((M, N), dtype)
            T.clear(C_tile)
            T.copy(A[0, 0], A_tile)
            T.copy(B[0, 0], B_tile)
            T.gemm(A_tile, B_tile, C_tile)
            T.copy(C_tile, C[0, 0])
    return kernel

try:
    # Get the TIR without compiling yet
    kernel = matmul_ame_func(128, 128, 128, dtype=T.int8)
    
    # Now compile with riscv_ame target to generate C code
    from tilelang.engine import lower
    from tilelang.target import Target
    
    target = Target(kind="riscv_ame", bits=64, lanes=1)
    
    # Lower to TIR
    print("Lowering to TIR...")
    c_code, _ = lower(kernel, target)
    
    print("Generated C Code:")
    print("=" * 80)
    lines = c_code.split('\n')
    
    # Look for AME pattern matching comments
    found_pattern = False
    for i, line in enumerate(lines):
        if "AME GEMM" in line or "Matrix multiply" in line:
            print(f"Found at line {i}: {line}")
            found_pattern = True
            # Print surrounding lines
            start = max(0, i - 5)
            end = min(len(lines), i + 20)
            print("\nContext:")
            for j in range(start, end):
                marker = ">>>" if j == i else "   "
                print(f"{marker} {j:3d}: {lines[j]}")
            print()
    
    if not found_pattern:
        print("Pattern matching comments NOT found. Showing kernel function:")
        for i, line in enumerate(lines):
            if "kernel" in line.lower():
                start = i
                end = min(len(lines), i + 100)
                for j in range(start, end):
                    print(f"{j:3d}: {lines[j]}")
                break
                
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
