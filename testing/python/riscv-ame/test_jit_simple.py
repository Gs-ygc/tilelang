#!/usr/bin/env python3
"""
Simple test of RISCV AME JIT compilation - Vector Add
"""
import tilelang
import tilelang.language as T

@tilelang.jit(target='riscv_ame', out_idx=[-1])
def simple_add(N, dtype=T.float32):
    @T.prim_func
    def add_kernel(A: T.Tensor((N,), dtype), B: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)):
        # Use T.Kernel to mark as device kernel
        with T.Kernel(1, threads=1) as _:
            for i in T.serial(N):
                C[i] = A[i] + B[i]
    return add_kernel

if __name__ == "__main__":
    print('Compiling simple vector add for RISCV AME...')
    try:
        kernel = simple_add(256)
        print('✅ Success! Kernel compiled.')
        
        # Get generated code
        mod = kernel.mod
        if mod.imported_modules:
            source = mod.imported_modules[0].get_source()
            print('\nGenerated C code (first 800 chars):')
            print('='*80)
            print(source[:800])
            print('='*80)
        else:
            print('No device module generated')
            
    except Exception as e:
        print(f'❌ Compilation failed: {e}')
        import traceback
        traceback.print_exc()
