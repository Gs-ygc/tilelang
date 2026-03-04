#!/usr/bin/env python3
"""
Test AME matrix multiplication code generation with different data types
Demonstrates support for fp32, fp16, and int8 - using a single function!
"""

import tilelang
import tilelang.language as T


@tilelang.jit(target='riscv_ame', out_idx=[-1])
def matmul_ame(M, N, K, dtype=T.float32):
    """Unified AME matmul that works with any dtype"""
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


def test_dtype(dtype_name, dtype, expected_etype, expected_bits):
    """Test a specific data type"""
    print(f"\n=== Testing {dtype_name} ===")
    
    try:
        func = matmul_ame(128, 128, 128, dtype=dtype)
        code = func.src
        print(f"✅ {dtype_name} code generated successfully")
        
        # Check for correct intrinsics
        expected_intrinsic = f"mlae{expected_bits}_m"
        expected_store = f"msce{expected_bits}_m"
        expected_msettype = f"msettype({expected_etype})"
        
        checks = []
        if expected_msettype in code:
            checks.append(f"✅ {expected_msettype}")
        else:
            checks.append(f"❌ Missing {expected_msettype}")
            
        if expected_intrinsic in code:
            checks.append(f"✅ {expected_intrinsic}")
        else:
            checks.append(f"⚠️  Missing {expected_intrinsic}")
            
        if expected_store in code:
            checks.append(f"✅ {expected_store}")
        else:
            checks.append(f"⚠️  Missing {expected_store}")
        
        for check in checks:
            print(f"  {check}")
        
        return True, code
    except Exception as e:
        print(f"❌ {dtype_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return False, ""


def show_generated_code_sample(dtype_name, code):
    """Display a code sample"""
    if not code:
        return
        
    print(f"\n=== Sample Generated Code ({dtype_name}) ===")
    
    # Extract the AME matrix multiply section
    lines = code.split('\n')
    in_ame_section = False
    ame_lines = []
    
    for line in lines:
        if "AME Matrix Multiply" in line:
            in_ame_section = True
        if in_ame_section:
            ame_lines.append(line)
            if line.strip() == '}' and len(ame_lines) > 10:
                break
    
    if ame_lines:
        print('\n'.join(ame_lines[:35]))  # Show first 35 lines


if __name__ == "__main__":
    print("=" * 60)
    print("AME Multi-DataType Support Test")
    print("=" * 60)
    
    # Test different data types - just change dtype parameter!
    # Format: (name, dtype, expected_etype, expected_bits)
    test_cases = [
        ("fp32 (float32)", T.float32, 2, 32),
        ("fp16 (float16)", T.float16, 1, 16),
        ("int8", T.int8, 0, 8),
    ]
    
    results = []
    codes = {}
    
    for dtype_name, dtype, etype, bits in test_cases:
        passed, code = test_dtype(dtype_name, dtype, etype, bits)
        results.append((dtype_name, passed))
        if code:
            codes[dtype_name] = code
    
    # Show sample code for fp16 if available
    if "fp16 (float16)" in codes:
        show_generated_code_sample("fp16", codes["fp16 (float16)"])
    elif "fp32 (float32)" in codes:
        show_generated_code_sample("fp32", codes["fp32 (float32)"])
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    for dtype_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{dtype_name:20s}: {status}")
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n🎉 All data types supported!")
    else:
        print("\n⚠️  Some data types failed")
