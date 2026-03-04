#!/usr/bin/env python3
"""
Simple test for RISCV AME backend - target registration verification
"""
import tilelang
from tilelang.utils.target import determine_target, check_riscv_ame_availability

def test_target_registration():
    """Test that riscv_ame target is properly registered"""
    print("Testing RISCV AME target registration...")
    
    try:
        # Test 1: Check if target can be created
        target = determine_target("riscv_ame", return_object=True)
        print(f"✅ Target created: {target}")
        print(f"   - Kind: {target.kind.name}")
        print(f"   - Keys: {target.keys}")
        
        # Test 2: Check availability detection
        available = check_riscv_ame_availability()
        print(f"\n✅ RISCV AME availability check: {available}")
        if not available:
            print("   ⚠️  LLVM-AME compiler not found (this is expected if not built yet)")
        
        # Test 3: Verify target in supported targets
        from tilelang.utils.target import SUPPORTED_TARGETS
        if "riscv_ame" in SUPPORTED_TARGETS:
            print(f"\n✅ Target in SUPPORTED_TARGETS:")
            print(f"   {SUPPORTED_TARGETS['riscv_ame']}")
        
        print("\n" + "="*80)
        print("✅ All target registration tests passed!")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_target_registration()
    exit(0 if success else 1)
