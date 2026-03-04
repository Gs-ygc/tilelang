#!/usr/bin/env python
import tilelang
import tilelang.language as T

def simple_matmul(M, N, K, dtype=T.float32):
    @T.prim_func
    def matmul_ame(
        A: T.Buffer((M, K), dtype),
        B: T.Buffer((K, N), dtype),
        C: T.Buffer((M, N), dtype),
    ):
        # Use T.gemm for matrix multiplication
        T.gemm(A, B, C)
    
    return matmul_ame

# Create function
matmul = simple_matmul(128, 128, 128, T.float32)

# Print TIR
print("=" * 60)
print("TIR (Before lowering):")
print("=" * 60)
print(matmul.script())
print()

# Try to get lowered version
import tilelang.engine.lower as lower_mod
target = tilelang.tvm.target.Target("riscv_ame")
print("=" * 60)
print(f"Target: {target}")
print("=" * 60)

# Try to lower
try:
    mod = tilelang.tvm.IRModule.from_expr(matmul)
    print("\nIRModule:")
    print(mod)
    
    # Apply TIR passes
    with target:
        mod = tilelang.tvm.tir.transform.LowerOpaqueBlock()(mod)
        print("\nAfter LowerOpaqueBlock:")
        print(mod)
except Exception as e:
    print(f"Lower failed: {e}")
    import traceback
    traceback.print_exc()
