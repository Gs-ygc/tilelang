#!/usr/bin/env python
"""Debug code generation to see where variables are declared."""

import tilelang
import tilelang.language as T
from tilelang.jit import JIT

@T.prim_func
def matmul_ame(
    A: T.Buffer((128, 128), "float32"),
    B: T.Buffer((128, 128), "float32"),
    C: T.Buffer((128, 128), "float32"),
):
    T.func_attr({"global_symbol": "main"})
    with T.Kernel(1, 1, threads=1) as (bx, by):
        A_tile = T.alloc_local((128, 128), "float32")
        B_tile = T.alloc_local((128, 128), "float32")
        C_tile = T.alloc_local((128, 128), "float32")
        T.copy(A[0, 0], A_tile)
        T.copy(B[0, 0], B_tile)
        T.gemm(A_tile, B_tile, C_tile)
        T.copy(C_tile, C[0, 0])

print("Compiling...")
target = tilelang.tvm.target.Target("riscv_ame")

try:
    jit = JIT(
        matmul_ame,
        target=target,
        out_idx=[2],
        name="matmul_debug",
    )
    print("✅ Compilation succeeded")
except Exception as e:
    print(f"❌ Compilation failed: {e}")
    
    # Try to get the generated code
    import traceback
    traceback.print_exc()
