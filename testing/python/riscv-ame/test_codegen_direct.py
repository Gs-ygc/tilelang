#!/usr/bin/env python3
"""
Test RISCV AME code generation directly
"""
import tilelang
import tilelang.language as T

@tilelang.jit(target='riscv_ame', out_idx=[-1])
def simple_add(N, dtype=T.float32):
    @T.prim_func
    def add_kernel(A: T.Tensor((N,), dtype), B: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)):
        with T.Kernel(1, threads=1) as _:
            for i in T.serial(N):
                C[i] = A[i] + B[i]
    return add_kernel

if __name__ == "__main__":
    print('Testing RISCV AME code generation...\n')
    
    # Get function
    N = 256
    func = simple_add.func(N)
    
    # Lower to get artifact
    target = "riscv_ame"
    artifact = tilelang.lower(func, target=target)
    
    print('✅ Lower succeeded!\n')
    
    # Print the kernel source from artifact
    if hasattr(artifact, 'kernel_source') and artifact.kernel_source:
        print('=== Generated Kernel Source ===')
        print(artifact.kernel_source)
    
    # Print device module IR
    if artifact.device_mod:
        print('\n=== Device Module IR ===')
        print(artifact.device_mod)
    
    print('\n✅ Code generation successful!')
