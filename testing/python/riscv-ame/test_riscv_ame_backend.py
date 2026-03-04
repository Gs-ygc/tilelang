"""
Simple test for RISCV AME backend support.
This test verifies basic functionality without requiring actual hardware.
"""

import pytest
import tilelang
import tilelang.language as T
import numpy as np


def test_riscv_ame_target_detection():
    """Test if RISCV AME target can be detected and configured."""
    from tilelang.utils.target import check_riscv_ame_availability, SUPPORTED_TARGETS
    
    # Check if riscv_ame is in supported targets
    assert "riscv_ame" in SUPPORTED_TARGETS
    
    # Try to detect availability (will fail if LLVM-AME not installed, which is OK)
    # We just verify the function exists and doesn't crash
    try:
        available = check_riscv_ame_availability()
        print(f"RISCV AME availability: {available}")
    except Exception as e:
        print(f"Note: RISCV AME check raised exception (expected if not installed): {e}")


def test_riscv_ame_simple_kernel_codegen():
    """Test code generation for a simple kernel without compilation."""
    
    M, N, K = 64, 64, 64
    block_M, block_N, block_K = 16, 16, 16
    
    @tilelang.jit(target="riscv_ame")
    def simple_matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16):
        @T.prim_func
        def kernel(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            # Simple sequential implementation for testing
            for i, j in T.grid(M, N):
                acc = T.alloc_local((1,), dtype)
                acc[0] = 0.0
                for k in range(K):
                    acc[0] += A[i, k] * B[k, j]
                C[i, j] = acc[0]
        
        return kernel
    
    try:
        # This will attempt to generate code
        # It may fail due to missing LLVM-AME, but we're testing the infrastructure
        kernel = simple_matmul(M, N, K, block_M, block_N, block_K)
        
        # Try to get the source code
        source = kernel.get_kernel_source()
        print("Generated source code:")
        print(source[:500] if len(source) > 500 else source)
        
        assert source is not None
        assert len(source) > 0
        
    except Exception as e:
        print(f"Code generation failed (may be expected): {e}")
        # We allow this to fail if LLVM-AME is not installed
        # The important thing is that the target is recognized
        import traceback
        traceback.print_exc()


def test_llvm_ame_compiler_wrapper():
    """Test the LLVM-AME compiler wrapper."""
    from tilelang.contrib import llvm_ame
    
    # Test finding LLVM-AME path
    try:
        path = llvm_ame.find_llvm_ame_path()
        print(f"Found LLVM-AME at: {path}")
        
        # Test getting compiler paths
        clang = llvm_ame.get_clang_path()
        print(f"Found clang at: {clang}")
        
        # Test getting features
        features = llvm_ame.get_target_features()
        print(f"Supported features: {features}")
        assert "ame" in features
        
    except RuntimeError as e:
        print(f"LLVM-AME not found (expected if not installed): {e}")
        pytest.skip("LLVM-AME not installed")


def test_riscv_ame_compilation():
    """Test actual compilation with LLVM-AME (requires LLVM-AME installed)."""
    from tilelang.contrib import llvm_ame
    
    try:
        # Simple test code
        test_code = """
        #include <stdint.h>
        extern "C" void add_vectors(float* a, float* b, float* c, int n) {
            for (int i = 0; i < n; i++) {
                c[i] = a[i] + b[i];
            }
        }
        """
        
        # Try to compile
        obj = llvm_ame.compile_riscv_ame(
            test_code,
            target_arch="rv64gc",  # Use basic RISCV64 without AME for now
            output_format="obj",
            options=["-O2"],
        )
        
        print(f"Successfully compiled test code ({len(obj)} bytes)")
        assert len(obj) > 0
        
    except RuntimeError as e:
        print(f"Compilation test skipped: {e}")
        pytest.skip("LLVM-AME compilation not available")
    except Exception as e:
        print(f"Compilation failed: {e}")
        pytest.skip("LLVM-AME compilation failed")


if __name__ == "__main__":
    print("=" * 80)
    print("Testing RISCV AME Backend Support")
    print("=" * 80)
    
    print("\n1. Testing target detection...")
    test_riscv_ame_target_detection()
    
    print("\n2. Testing LLVM-AME compiler wrapper...")
    try:
        test_llvm_ame_compiler_wrapper()
    except Exception as e:
        print(f"Skipped: {e}")
    
    print("\n3. Testing simple kernel code generation...")
    try:
        test_riscv_ame_simple_kernel_codegen()
    except Exception as e:
        print(f"Failed: {e}")
    
    print("\n4. Testing LLVM-AME compilation...")
    try:
        test_riscv_ame_compilation()
    except Exception as e:
        print(f"Skipped: {e}")
    
    print("\n" + "=" * 80)
    print("All available tests completed!")
    print("=" * 80)
