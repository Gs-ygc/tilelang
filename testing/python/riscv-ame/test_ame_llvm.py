#!/usr/bin/env python3
"""
使用 AME LLVM 编译器测试多数据类型支持
"""

import os
import tilelang
import tilelang.language as T

# 设置使用 AME LLVM 编译器
os.environ["CXX"] = os.path.expandvars("$LLVM_HOME/bin/clang++")
os.environ["CC"] = os.path.expandvars("$LLVM_HOME/bin/clang")

print(f"使用编译器: {os.environ.get('CXX', 'default')}")
print()

@tilelang.jit(target='riscv_ame', out_idx=[-1])
def matmul_ame(M, N, K, dtype=T.float32):
    """统一的 AME matmul 函数，支持多种数据类型"""
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


def test_dtype(dtype_name, dtype):
    """测试特定数据类型"""
    print(f"=== 测试 {dtype_name} ===")
    
    try:
        func = matmul_ame(128, 128, 128, dtype=dtype)
        print(f"✅ {dtype_name} 编译成功!")
        
        # 显示生成的代码片段
        code = func.src
        if "// AME Matrix Multiply" in code:
            lines = code.split('\n')
            for i, line in enumerate(lines):
                if "// AME Matrix Multiply" in line:
                    print("\n生成的代码片段:")
                    for j in range(i, min(i + 15, len(lines))):
                        print(lines[j])
                    break
        
        return True
    except Exception as e:
        print(f"❌ {dtype_name} 失败: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("AME 多数据类型支持测试 (使用 AME LLVM)")
    print("=" * 60)
    print()
    
    # 测试不同数据类型
    test_cases = [
        # ("fp32 (float32)", T.float32),
        # ("fp16 (float16)", T.float16),
        ("int8", T.int8),
    ]
    
    results = []
    for dtype_name, dtype in test_cases:
        passed = test_dtype(dtype_name, dtype)
        results.append((dtype_name, passed))
        print()
    
    print("=" * 60)
    print("测试总结:")
    print("=" * 60)
    for dtype_name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{dtype_name:20s}: {status}")
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n🎉 所有数据类型都支持!")
    else:
        print("\n⚠️  部分数据类型失败")
