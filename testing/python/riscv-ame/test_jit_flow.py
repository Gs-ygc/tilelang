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
        T.gemm(A, B, C)
    
    return matmul_ame

# Create function
matmul = simple_matmul(128, 128, 128, T.float32)

print("=" * 60)
print("Initial TIR:")
print("=" * 60)
print(matmul.script())

# Now JIT compile and see what code is generated
target = tilelang.tvm.target.Target("riscv_ame")

@T.prim_func
def main(
    A: T.Buffer((128, 128), "float32"),
    B: T.Buffer((128, 128), "float32"),
    C: T.Buffer((128, 128), "float32"),
):
    T.func_attr({"global_symbol": "main"})
    T.gemm(A, B, C)

from tilelang.jit.kernel import JITKernel

print("\n" + "=" * 60)
print("Compiling with JIT...")
print("=" * 60)

try:
    jit = JITKernel(
        main,
        target=target,
        out_idx=[2],
        name="matmul_ame"
    )
    print("\n✅ JIT编译成功!")
    
except Exception as e:
    print(f"\n❌ JIT编译失败: {e}")
    import traceback
    traceback.print_exc()
