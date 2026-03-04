"""
LLVM-AME compiler wrapper for RISCV AME backend.
This module provides utilities to compile C/C++ code using the LLVM-AME compiler
with RISCV Matrix Extension (AME) support.
"""

import os
import subprocess
import tempfile
from typing import List, Optional
from pathlib import Path


def find_llvm_ame_path() -> str:
    """
    Find the LLVM-AME installation path.
    
    Returns:
        str: Path to LLVM-AME installation
        
    Raises:
        RuntimeError: If LLVM-AME cannot be found
    """
    # Check environment variable first
    llvm_ame_path = os.environ.get('LLVM_AME_PATH')
    if llvm_ame_path and os.path.exists(llvm_ame_path):
        return llvm_ame_path
    
    # Check XS_PROJECT_ROOT/local/llvm
    xs_root = os.environ.get('XS_PROJECT_ROOT')
    if xs_root:
        llvm_path = os.path.join(xs_root, 'local/llvm')
        if os.path.exists(llvm_path):
            return llvm_path
    
    # Check common installation paths
    common_paths = [
        '/usr/local/llvm-ame',
        '/opt/llvm-ame',
        os.path.expanduser('~/llvm-ame'),
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    raise RuntimeError(
        "LLVM-AME compiler not found. "
        "Please set LLVM_AME_PATH environment variable or install LLVM-AME to a standard location. "
        "You can build it from llvm-project-ame in your xsai-env workspace."
    )


def get_clang_path() -> str:
    """
    Get the path to clang compiler from LLVM-AME.
    
    Returns:
        str: Path to clang executable
    """
    llvm_ame_path = find_llvm_ame_path()
    clang = os.path.join(llvm_ame_path, 'bin/clang')
    
    if not os.path.exists(clang):
        raise RuntimeError(f"clang not found at {clang}")
    
    return clang


def get_clangxx_path() -> str:
    """
    Get the path to clang++ compiler from LLVM-AME.
    
    Returns:
        str: Path to clang++ executable
    """
    llvm_ame_path = find_llvm_ame_path()
    clangxx = os.path.join(llvm_ame_path, 'bin/clang++')
    
    if not os.path.exists(clangxx):
        raise RuntimeError(f"clang++ not found at {clangxx}")
    
    return clangxx


def compile_riscv_ame(
    code: str,
    target_arch: str = "rv64gcv_ame",
    output_format: str = "obj",  # "obj", "asm", or "both"
    options: Optional[List[str]] = None,
    llvm_path: Optional[str] = None,
    verbose: bool = False,
) -> bytes:
    """
    Compile C/C++ code with LLVM-AME compiler for RISCV AME target.
    
    Args:
        code: Source code string to compile
        target_arch: Target architecture string (default: rv64gcv_ame)
        output_format: Output format - 'obj' for object file, 'asm' for assembly
        options: Additional compiler options
        llvm_path: Optional override for LLVM-AME path
        verbose: Enable verbose output
        
    Returns:
        bytes: Compiled binary content
        
    Raises:
        RuntimeError: If compilation fails
    """
    if options is None:
        options = []
    
    # Get compiler path
    if llvm_path:
        os.environ['LLVM_AME_PATH'] = llvm_path
    
    clangxx = get_clangxx_path()
    
    # Get include paths
    tilelang_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tilelang_src_include = os.path.join(os.path.dirname(tilelang_root), "src")
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        src_file = os.path.join(tmpdir, "kernel.cc")
        obj_file = os.path.join(tmpdir, "kernel.o")
        asm_file = os.path.join(tmpdir, "kernel.s")
        
        if output_format == "obj":
            # Compile to object file, then link to shared library
            out_file = os.path.join(tmpdir, "kernel.so")
            needs_linking = True
            needs_asm = False
        elif output_format == "asm":
            out_file = asm_file
            needs_linking = False
            needs_asm = True
        elif output_format == "both":
            # Generate both assembly and shared library
            out_file = os.path.join(tmpdir, "kernel.so")
            needs_linking = True
            needs_asm = True
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        # Write source code
        with open(src_file, "w") as f:
            f.write(code)
        
        # Step 1: Generate assembly if requested
        if needs_asm:
            asm_cmd = [
                clangxx,
                "-S",  # Generate assembly
                f"-march={target_arch}",
                "-target", "riscv64-unknown-linux-gnu",
                f"-I{tilelang_src_include}",
                "-o", asm_file,
                src_file,
            ] + options
            
            if verbose:
                print(f"Generating assembly with: {' '.join(asm_cmd)}")
            
            try:
                result = subprocess.run(
                    asm_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                    text=True,
                )
                
                if verbose and result.stdout:
                    print("Assembly gen stdout:", result.stdout)
                if verbose and result.stderr:
                    print("Assembly gen stderr:", result.stderr)
                    
            except subprocess.CalledProcessError as e:
                error_msg = (
                    f"LLVM-AME assembly generation failed:\n"
                    f"Command: {' '.join(asm_cmd)}\n"
                    f"Return code: {e.returncode}\n"
                    f"Stdout: {e.stdout}\n"
                    f"Stderr: {e.stderr}"
                )
                raise RuntimeError(error_msg) from e
            
            # Print assembly for debugging
            if verbose or output_format == "asm":
                with open(asm_file, "r") as f:
                    asm_content = f.read()
                    print("\n" + "=" * 60)
                    print("Generated Assembly:")
                    print("=" * 60)
                    print(asm_content)
                    print("=" * 60 + "\n")
            
            if output_format == "asm":
                # Return assembly as bytes
                with open(asm_file, "rb") as f:
                    return f.read()
        
        # Step 2: Compile to object file
        compile_cmd = [
            clangxx,
            "-c",  # Compile only
            f"-march={target_arch}",
            "-target", "riscv64-unknown-linux-gnu",
            f"-I{tilelang_src_include}",  # Add include path for riscv_ame.h
            "-o", obj_file,
            src_file,
        ] + options
        
        if verbose:
            print(f"Compiling with: {' '.join(compile_cmd)}")
        
        # Run compiler
        try:
            result = subprocess.run(
                compile_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True,
            )
            
            if verbose and result.stdout:
                print("Compiler stdout:", result.stdout)
            if verbose and result.stderr:
                print("Compiler stderr:", result.stderr)
                
        except subprocess.CalledProcessError as e:
            error_msg = (
                f"LLVM-AME compilation failed:\n"
                f"Command: {' '.join(compile_cmd)}\n"
                f"Return code: {e.returncode}\n"
                f"Stdout: {e.stdout}\n"
                f"Stderr: {e.stderr}"
            )
            raise RuntimeError(error_msg) from e
        
        # Step 2: Link to shared library if needed
        if needs_linking:
            link_cmd = [
                clangxx,
                "-shared",  # Create shared library
                "-fPIC",
                "-o", out_file,
                obj_file,
            ]
            
            if verbose:
                print(f"Linking with: {' '.join(link_cmd)}")
            
            try:
                result = subprocess.run(
                    link_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                    text=True,
                )
                
                if verbose and result.stdout:
                    print("Linker stdout:", result.stdout)
                if verbose and result.stderr:
                    print("Linker stderr:", result.stderr)
                    
            except subprocess.CalledProcessError as e:
                error_msg = (
                    f"LLVM-AME linking failed:\n"
                    f"Command: {' '.join(link_cmd)}\n"
                    f"Return code: {e.returncode}\n"
                    f"Stdout: {e.stdout}\n"
                    f"Stderr: {e.stderr}"
                )
                raise RuntimeError(error_msg) from e
        
        # Read output file
        with open(out_file, "rb") as f:
            return f.read()


def get_target_features() -> List[str]:
    """
    Get list of supported RISCV features for AME target.
    
    Returns:
        List of feature strings
    """
    return [
        "rv64gc",      # Base RISCV64 with standard extensions
        "v",           # Vector extension
        "ame",         # Matrix extension (custom)
        "m",           # Integer multiply/divide
        "a",           # Atomic operations
        "f",           # Single-precision floating-point
        "d",           # Double-precision floating-point
        "c",           # Compressed instructions
    ]


def check_ame_support() -> bool:
    """
    Check if LLVM-AME compiler supports AME instructions.
    
    Returns:
        bool: True if AME is supported
    """
    try:
        clang = get_clang_path()
        
        # Try to compile a simple AME test
        test_code = """
        void test_ame() {
            // This would contain AME intrinsics
            // For now, just check if compiler exists
        }
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
            f.write(test_code)
            test_file = f.name
        
        try:
            result = subprocess.run(
                [clang, '-march=rv64gcv_ame', '-c', test_file, '-o', '/dev/null'],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        finally:
            os.unlink(test_file)
            
    except Exception:
        return False


if __name__ == "__main__":
    # Self-test
    print("Testing LLVM-AME compiler wrapper...")
    
    try:
        llvm_path = find_llvm_ame_path()
        print(f"✓ Found LLVM-AME at: {llvm_path}")
        
        clang = get_clang_path()
        print(f"✓ Found clang at: {clang}")
        
        clangxx = get_clangxx_path()
        print(f"✓ Found clang++ at: {clangxx}")
        
        features = get_target_features()
        print(f"✓ Supported features: {', '.join(features)}")
        
        # Test compilation
        test_code = """
        #include <stdint.h>
        extern "C" void test_kernel(float* a, float* b, float* c, int n) {
            for (int i = 0; i < n; i++) {
                c[i] = a[i] + b[i];
            }
        }
        """
        
        obj = compile_riscv_ame(test_code, verbose=True)
        print(f"✓ Successfully compiled test code ({len(obj)} bytes)")
        
        print("\nAll tests passed!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import sys
        sys.exit(1)
