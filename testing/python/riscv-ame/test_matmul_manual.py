#!/usr/bin/env python3
"""
Test RISCV AME code generation without T.gemm (manual loop)
This tests the basic compilation pipeline for CPU target
"""
import tilelang
import tilelang.language as T

@tilelang.jit(target='riscv_ame', out_idx=[-1])
def simple_matmul_loop(M, N, K, dtype=T.float32):
    @T.prim_func
    def matmul_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # Use is_cpu=True for CPU targets (no warp concept)
        with T.Kernel(1, 1, is_cpu=True) as (bx, by):
            # Simple triple loop matmul (no T.gemm intrinsic)
            for i in T.serial(M):
                for j in T.serial(N):
                    acc = T.float32(0)
                    for ki in T.serial(K):
                        acc += A[i, ki] * B[ki, j]
                    C[i, j] = acc
    
    return matmul_kernel

if __name__ == "__main__":
    print('Testing RISCV AME code generation (manual loop)...\n')
    
    # Small matrix
    M, N, K = 4, 4, 4
    func = simple_matmul_loop.func(M, N, K)
    
    # Lower to get artifact
    target = "riscv_ame"
    try:
        artifact = tilelang.lower(func, target=target)
        
        print('✅ Lower succeeded!\n')
        
        # Print the generated source
        if hasattr(artifact, 'kernel_source') and artifact.kernel_source:
            print('=== Generated Kernel Source ===')
            print(artifact.kernel_source)
        
        print('\n✅ Code generation successful!')
        print('\n💡 Note: T.gemm intrinsic requires special handling for CPU targets.')
        print('   For AME instructions, we need to either:')
        print('   1. Implement a CPU-specific Gemm class (like GemmCPU)')
        print('   2. Use pattern matching to detect matmul loops and replace with AME instructions')
        print('   3. Use explicit AME intrinsic calls in the kernel')
        
    except Exception as e:
        print(f'❌ Failed: {e}')
        import traceback
        traceback.print_exc()
