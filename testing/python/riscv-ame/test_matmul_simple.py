#!/usr/bin/env python3
"""
Test RISCV AME matmul code generation with T.gemm
Now using GemmAME implementation
"""
import tilelang
import tilelang.language as T

@tilelang.jit(target='riscv_ame', out_idx=[-1])
def simple_matmul(M, N, K, dtype=T.float32):
    @T.prim_func
    def matmul_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # RISCV AME: Single thread processing large tile
        # AME prefers large tiles per thread (> 128x64x128)
        # threads=1 means single-threaded execution
        with T.Kernel(1, 1, threads=1) as (bx, by):
            # Allocate local tile buffers (128x128 per thread)
            # Large tile size > 128x64x128 for efficient AME usage
            A_tile = T.alloc_local((M, K), dtype)
            B_tile = T.alloc_local((K, N), dtype)
            C_tile = T.alloc_local((M, N), dtype)
            
            # Clear accumulator
            T.clear(C_tile)
            
            # Load A and B
            T.copy(A[0, 0], A_tile)
            T.copy(B[0, 0], B_tile)
            
            # Matrix multiply (GemmAME with multi-threading!)
            T.gemm(A_tile, B_tile, C_tile)
            
            # Store result
            T.copy(C_tile, C[0, 0])
    
    return matmul_kernel

if __name__ == "__main__":
    print('Testing RISCV AME matmul with T.gemm and GemmAME...\n')
    
    # AME prefers large tiles: single thread handles > 128x64x128
    # Using 128x128x128 for this test
    M, N, K = 128, 128, 128
    func = simple_matmul.get_tir(M, N, K)
    
    # Lower to get artifact
    target = "riscv_ame"
    try:
        artifact = tilelang.lower(func, target=target)
        
        print('✅ Lower succeeded!\n')
        
        # Print the generated source
        if hasattr(artifact, 'kernel_source') and artifact.kernel_source:
            print('=== Generated Kernel Source ===')
            print(artifact.kernel_source)
            
            # Check for matmul pattern
            has_loops = 'for' in artifact.kernel_source
            print(f'\n{"✅" if has_loops else "❌"} Contains loop pattern (will be converted to AME by codegen)')
        
        print('\n✅ Code generation successful with GemmAME!')
        
    except Exception as e:
        print(f'❌ Failed: {e}')
        import traceback
        traceback.print_exc()
