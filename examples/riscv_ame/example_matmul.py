"""
Example: Matrix Multiplication using RISCV AME Extension

This example demonstrates how to write a simple matrix multiplication kernel
for RISCV CPUs with the AME (Matrix Extension) using TileLang.
"""

import tilelang
import tilelang.language as T


@tilelang.jit(target="riscv_ame", out_idx=[-1])
def matmul_ame(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    """
    Matrix multiplication kernel for RISCV AME.
    
    This will be lowered to RISCV AME instructions:
    - msettilemi: Configure tile dimensions
    - mlae.*.m: Load matrix tile A
    - mlbe.*.m: Load matrix tile B
    - mfma.*.mm: Matrix multiply-accumulate
    - mse.*.m: Store result tile
    """
    @T.prim_func
    def matmul_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # Use T.Kernel for proper tiling (even though riscv_ame is CPU target)
        # The backend will recognize this pattern and generate AME instructions
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=1) as (bx, by):
            # Allocate local tiles (will map to AME tile registers)
            A_local = T.alloc_fragment((block_M, block_K), dtype)
            B_local = T.alloc_fragment((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            
            # Initialize accumulator to zero
            T.clear(C_local)
            
            # Loop over K dimension with tiling
            for k in T.serial(T.ceildiv(K, block_K)):
                # Load tiles from global memory
                # Backend will generate: mlae.hf.m / mlbe.hf.m
                T.copy(A[by * block_M, k * block_K], A_local)
                T.copy(B[k * block_K, bx * block_N], B_local)
                
                # Matrix multiply-accumulate
                # Backend will generate: mfma.hf.mm
                T.gemm(A_local, B_local, C_local)
            
            # Store result back to global memory
            # Backend will generate: mse.hf.m
            T.copy(C_local, C[by * block_M, bx * block_N])
    
    return matmul_kernel


def matmul_riscv_ame_example():
    """Test matrix multiplication with RISCV AME backend"""
    
    # Matrix dimensions
    M = 64
    N = 64
    K = 64
    
    # Tile dimensions (8x8 is optimal for most AME implementations)
    block_M = 8
    block_N = 8
    block_K = 8
    
    print("Compiling matrix multiplication kernel for RISCV AME...")
    print(f"  Matrix size: {M}x{K} @ {K}x{N} = {M}x{N}")
    print(f"  Tile size: {block_M}x{block_K} @ {block_K}x{block_N}")
    
    try:
        kernel = matmul_ame(M, N, K, block_M, block_N, block_K)
        
        # Get generated source code
        mod = kernel.mod
        source = mod.imported_modules[0].get_source() if mod.imported_modules else str(mod)
        
        print("\n" + "=" * 80)
        print("Generated RISCV AME Code:")
        print("=" * 80)
        print(source)
        print("=" * 80)
        
        # Check for AME instructions
        ame_found = []
        if "msettilemi" in source or "AME" in source:
            ame_found.append("✓ Tile configuration")
        if "mlae" in source or "mlbe" in source:
            ame_found.append("✓ Tile load operations")
        if "mfma" in source:
            ame_found.append("✓ Matrix multiply-accumulate")
        if "mse" in source:
            ame_found.append("✓ Tile store operations")
        
        if ame_found:
            print("\n✅ AME Instructions Found:")
            for item in ame_found:
                print(f"   {item}")
        else:
            print("\n⚠️  AME instructions not found in generated code")
            print("   The backend may have used scalar fallback for this configuration")
        
        return kernel
        
    except Exception as e:
        print(f"Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: Make sure LLVM-AME is installed and LLVM_AME_PATH is set.")
        print("You can build LLVM-AME using: make llvm in xsai-env directory")
        return None


@tilelang.jit(target="riscv_ame", out_idx=[-1])
def vector_add(N, dtype=T.float32):
    """Simple vector addition for RISCV AME (scalar operations)"""
    @T.prim_func
    def add_kernel(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        for i in T.serial(N):
            C[i] = A[i] + B[i]
    
    return add_kernel


def simple_add_example():
    """
    Simple element-wise addition example for RISCV AME.
    
    This is a simpler example that doesn't use matrix instructions,
    just to verify basic code generation.
    """
    
    N = 256
    
    print("\nCompiling vector addition kernel for RISCV AME...")
    print(f"  Vector size: {N}")
    
    try:
        kernel = vector_add(N)
        
        # Get generated source code
        mod = kernel.mod
        source = mod.imported_modules[0].get_source() if mod.imported_modules else str(mod)
        
        print("\n" + "=" * 80)
        print("Generated RISCV Code (Vector Add):")
        print("=" * 80)
        # Print first 1000 chars
        print(source[:1000] if len(source) > 1000 else source)
        if len(source) > 1000:
            print("... (truncated)")
        print("=" * 80)
        
        print("\n✅ Code generation successful!")
        
        return kernel
        
    except Exception as e:
        print(f"Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_with_data(kernel, M, N, K):
    """
    Test the compiled kernel with actual data (requires XSAI hardware or simulator).
    """
    print("\nTesting kernel with data...")
    
    try:
        import numpy as np
        
        # Generate test data
        a = np.random.randn(M, K).astype(np.float16)
        b = np.random.randn(K, N).astype(np.float16)
        c = np.zeros((M, N), dtype=np.float16)
        
        # Run kernel
        kernel(a, b, c)
        
        # Verify result
        ref = a @ b
        error = np.abs(c - ref).max()
        print(f"Maximum error: {error}")
        
        if error < 1e-2:
            print("✓ Test passed!")
        else:
            print("✗ Test failed!")
            
    except Exception as e:
        print(f"Testing failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: This requires XSAI hardware or NEMU simulator.")
        print("You can run the kernel on NEMU using:")
        print("  make test-matrix  # in xsai-env directory")


if __name__ == "__main__":
    print("=" * 80)
    print("TileLang RISCV AME Examples")
    print("=" * 80)
    
    # Example 1: Simple vector addition
    print("\n[Example 1] Simple Vector Addition")
    print("-" * 80)
    add_kernel = simple_add_example()
    
    # Example 2: Matrix multiplication
    print("\n[Example 2] Matrix Multiplication with AME")
    print("-" * 80)
    matmul_kernel = matmul_riscv_ame_example()
    
    # If compilation succeeded, optionally test with data
    if matmul_kernel is not None:
        # Uncomment to test with actual data (requires XSAI hardware/simulator)
        # test_with_data(matmul_kernel, 64, 64, 64)
        pass
    
    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)
    
    print("\n📖 Next Steps:")
    print("  1. Make sure LLVM-AME is built: make llvm")
    print("  2. Set up environment: source env.sh")
    print("  3. Run on NEMU simulator: make test-matrix")
    print("  4. Check documentation: docs/RISCV_AME_BACKEND.md")
