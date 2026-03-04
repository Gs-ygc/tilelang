#!/usr/bin/env python3
"""
Debug test - check device/host module split
"""
import tilelang
import tilelang.language as T
from tilelang.jit.adapter.utils import get_annotated_mod

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
    print('Creating kernel...')
    
    # Get the function without compiling
    N = 256
    func = simple_add.func(N)
    print(f'Function: {func}')
    
    # Try to lower it manually
    print('\nLowering with tilelang.lower...')
    try:
        target = "riscv_ame"
        artifact = tilelang.lower(func, target=target, enable_device_compile=False)
        
        print(f'\n✅ Lower succeeded!')
        print(f'Host module functions: {list(artifact.host_mod.functions.keys())}')
        print(f'Device module functions: {list(artifact.device_mod.functions.keys())}')
        
        # Check annotated split
        print('\nChecking get_annotated_mod split...')
        device_mod, host_mod = get_annotated_mod(func, target)
        print(f'Device functions: {list(device_mod.functions.keys())}')
        print(f'Host functions: {list(host_mod.functions.keys())}')
        
        # Print device module attrs
        print('\nHost module function attrs:')
        for name, f in host_mod.functions.items():
            print(f'\nFunction {name}:')
            for k, v in f.attrs.items():
                print(f'  {k}: {v}')
                
    except Exception as e:
        print(f'❌ Failed: {e}')
        import traceback
        traceback.print_exc()
