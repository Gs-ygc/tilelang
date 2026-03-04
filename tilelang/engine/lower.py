"""The compiler for TL programs."""

from __future__ import annotations

from typing import Callable
import tilelang.transform
from tilelang import tvm as tvm
from tvm import tir
import tvm_ffi
from tvm.ir import CallingConv
from tvm.target import Target
from tilelang.contrib import hipcc, nvcc
from tilelang.env import COMPOSABLE_KERNEL_INCLUDE_DIR, CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH
from tilelang.transform import PassConfigKey
from tilelang.transform.metal import MarkHostMetalContext
from tilelang.engine.param import KernelParam, CompiledArtifact
from tilelang.utils.target import determine_target
from tilelang.engine.phase import (
    PreLowerSemanticCheck,
    LowerAndLegalize,
    OptimizeForTarget,
)


def is_cpu_device_backend(target: Target):
    return target.kind.name in ["c", "riscv_ame"]


def has_device_kernel_launch(attrs) -> bool:
    """Check if the attributes indicate a device kernel launch."""
    return bool(attrs and "calling_conv" in attrs and attrs["calling_conv"] == CallingConv.DEVICE_KERNEL_LAUNCH)


def is_device_call_c_device(func: tir.PrimFunc):
    attrs = func.attrs
    calling_conv = attrs.get("calling_conv", CallingConv.DEFAULT)
    is_cpacked = calling_conv == CallingConv.C_PACKED_FUNC

    # Check if it's a C or RISCV AME target
    if "target" in attrs:
        target_kind = attrs["target"].kind.name
        if target_kind in ["c", "riscv_ame"] and not is_cpacked:
            return True

    return has_device_kernel_launch(attrs)


def is_device_call(func: tir.PrimFunc):
    return has_device_kernel_launch(func.attrs)


def get_device_call(is_device_c: bool = False) -> Callable[[tir.PrimFunc], bool]:
    return is_device_call_c_device if is_device_c else is_device_call


def get_host_call(is_device_c: bool = False) -> Callable[[tir.PrimFunc], bool]:
    return lambda func: not get_device_call(is_device_c)(func)


@tvm_ffi.register_global_func("tilelang_callback_cuda_compile", override=True)
def tilelang_callback_cuda_compile(code, target, pass_config=None):
    target_arch = nvcc.get_target_arch(nvcc.get_target_compute_version(target))

    arch = [f"-arch=sm_{target_arch}"]
    compile_format = "cubin"

    # Read pass-config keys (string-valued) like in jit.adapter.libgen.compile_lib
    cfg = pass_config or {}
    enable_fast_math = bool(cfg.get(PassConfigKey.TL_ENABLE_FAST_MATH, False))

    ptxas_usage_level = cfg.get(PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL, None)
    verbose_ptxas_output = bool(cfg.get(PassConfigKey.TL_ENABLE_PTXAS_VERBOSE_OUTPUT, False))

    options = [
        "-std=c++17",
        "-I" + TILELANG_TEMPLATE_PATH,
        "-I" + CUTLASS_INCLUDE_DIR,
    ]
    # Merge extra device compiler flags from pass config, if provided
    extra_flags = cfg.get(PassConfigKey.TL_DEVICE_COMPILE_FLAGS, None)
    if extra_flags:
        import shlex

        if isinstance(extra_flags, str):
            tokens = shlex.split(extra_flags)
        else:
            tokens = []
            for flag in extra_flags:
                if isinstance(flag, str):
                    tokens.extend(shlex.split(flag))
                else:
                    tokens.append(str(flag))
        options += tokens

    verbose = False
    if enable_fast_math:
        options.append("--use_fast_math")
    if ptxas_usage_level is not None:
        options.append(f"--ptxas-options=--register-usage-level={ptxas_usage_level}")
    if verbose_ptxas_output:
        options.append("--ptxas-options=--verbose")
        options.append("-w")  # Suppress warnings to make ptxas output more readable
        verbose = True

    ptx = nvcc.compile_cuda(
        code,
        compile_format,
        arch,
        options=options,
        verbose=verbose,
    )

    return ptx


@tvm_ffi.register_global_func("tilelang_callback_hip_compile", override=True)
def tilelang_callback_hip_compile(code, target):
    hsaco = hipcc.compile_hip(
        code,
        target_format="hsaco",
        options=[
            "-std=c++17",
            "-I" + TILELANG_TEMPLATE_PATH,
            "-I" + COMPOSABLE_KERNEL_INCLUDE_DIR,
        ],
        verbose=False,
    )

    return hsaco


@tvm_ffi.register_global_func("tilelang_callback_riscv_ame_compile", override=True)
def tilelang_callback_riscv_ame_compile(code, target, pass_config=None):
    """Compile C code with AME intrinsics using LLVM-AME compiler."""
    import logging
    logger = logging.getLogger(__name__)
    
    # 添加调试信息 - 看看是否真的调用了这个函数
    logger.warning("=" * 60)
    logger.warning("🔧 tilelang_callback_riscv_ame_compile 被调用!")
    logger.warning("=" * 60)
    print("\n" + "=" * 60)
    print("🔧 RISCV AME 编译回调被调用!")
    print("=" * 60)
    
    from tilelang.contrib import llvm_ame
    
    # Get LLVM-AME compiler path
    llvm_ame_path = os.environ.get('LLVM_AME_PATH')
    if not llvm_ame_path:
        xs_root = os.environ.get('XS_PROJECT_ROOT')
        if xs_root:
            llvm_ame_path = osp.join(xs_root, 'local/llvm')
    
    print(f"LLVM_AME_PATH: {llvm_ame_path}")
    
    # Read pass-config keys
    cfg = pass_config or {}
    
    # Build compiler options
    options = [
        "-std=c++17",
        "-O3",
        "-fPIC",
        "-march=rv64gcv_zvfh_zvfhmin",  # 添加明确的 march 以支持 vector
        "-menable-experimental-extensions",  # 启用实验性扩展
        "-Wno-error",  # 暂时忽略警告作为错误
    ]
    
    print(f"编译选项: {options}")
    
    # Merge extra compiler flags from pass config
    extra_flags = cfg.get(PassConfigKey.TL_DEVICE_COMPILE_FLAGS, None)
    if extra_flags:
        import shlex
        if isinstance(extra_flags, str):
            tokens = shlex.split(extra_flags)
        else:
            tokens = []
            for flag in extra_flags:
                if isinstance(flag, str):
                    tokens.extend(shlex.split(flag))
                else:
                    tokens.append(str(flag))
        options += tokens
        print(f"添加了额外的编译选项: {tokens}")
    
    # Get target architecture from target
    target_arch = "rv64gcv_ame"  # Default RISCV64 with vector and AME
    if hasattr(target, 'attrs') and 'march' in target.attrs:
        target_arch = target.attrs['march']
    
    print(f"目标架构: {target_arch}")
    print(f"代码长度: {len(code)} bytes")
    print("=" * 60 + "\n")
    
    # Compile with LLVM-AME
    try:
        obj = llvm_ame.compile_riscv_ame(
            code,
            target_arch=target_arch,
            output_format="both",
            options=options,
            llvm_path=llvm_ame_path,
            verbose=True,  # 强制启用 verbose 输出
        )
        print(f"✅ 编译成功! 目标文件大小: {len(obj)} bytes\n")
        return obj
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"❌ RISCV AME compilation failed: {e}")
        logger.error(f"Generated code:\n{code}")
        print(f"\n❌ 编译失败: {e}\n")
        print("生成的代码:")
        print(code)
        raise


def extrac_params(func: tir.PrimFunc) -> list[KernelParam]:
    tensor_types = []
    for var in func.params:
        if var in func.buffer_map:
            tensor_types.append(KernelParam.from_buffer(func.buffer_map[var]))
        else:
            tensor_types.append(KernelParam.from_var(var))
    return tensor_types


def canon_target_host(target: str | Target, target_host: str | Target | None):
    if not target_host:
        # For CPU targets (c, riscv_ame), use the same target as host
        if isinstance(target, Target) and target.kind.name in ["c", "riscv_ame"]:
            return target
        elif isinstance(target, str) and target in ["c", "riscv_ame"]:
            return target
        # For GPU targets, use llvm or c as host
        target_host = "llvm" if tvm.runtime.enabled("llvm") else "c"

    return target_host


def host_codegen(host_mod: tvm.IRModule, target_host: Target, target: Target | None = None) -> tvm.IRModule:
    """Generate host-side code from the lowered IR module.

    Parameters
    ----------
    host_mod : tvm.IRModule
        The host-side IR module to compile.
    target_host : Target
        The host compilation target (e.g. "llvm" or "c").
    target : Target, optional
        The device target.  When the device target is Metal, the pass
        MarkHostMetalContext is applied so that the generated host code
        contains the Metal/MPS synchronisation logic.
    """
    host_mod = tir.transform.BindTarget(target_host)(host_mod)
    host_mod = tir.transform.FP8StorageLegalize()(host_mod)
    host_mod = tir.transform.BF16StorageLegalize()(host_mod)
    host_mod = tir.transform.LowerTVMBuiltin()(host_mod)
    host_mod = tir.transform.LowerCustomDatatypes()(host_mod)
    host_mod = tilelang.transform.LowerIntrin()(host_mod)
    host_mod = tilelang.transform.LowerDeviceStorageAccessInfo()(host_mod)
    host_mod = tir.transform.CombineContextCall()(host_mod)
    if target is not None and target.kind.name == "metal":
        host_mod = MarkHostMetalContext()(host_mod)
    if target_host.kind.name == "llvm":
        host_mod = tvm.ffi.get_global_func("target.build.llvm")(host_mod, target_host)
    elif target_host.kind.name == "c":
        host_mod = tvm.ffi.get_global_func("target.build.tilelang_c")(host_mod, target_host)
    elif target_host.kind.name == "riscv_ame":
        # RISCV AME uses same build as C for host code
        host_mod = tvm.ffi.get_global_func("target.build.tilelang_c")(host_mod, target_host)
    else:
        raise ValueError(f"Target host {target_host.kind.name} is not supported")
    return host_mod


def device_codegen(device_mod: tvm.IRModule, target: Target) -> tvm.IRModule:
    device_mod = tilelang.transform.LowerDeviceStorageAccessInfo()(device_mod)
    device_mod = tilelang.transform.LowerIntrin()(device_mod)
    device_mod = tir.transform.Simplify()(device_mod)
    device_mod = tilelang.transform.HoistBroadcastValues()(device_mod)

    if target.kind.name == "cuda":
        global_func = "target.build.tilelang_" + ("cutedsl" if "cutedsl" in target.keys else "cuda")
        device_mod = tvm.ffi.get_global_func(global_func)(device_mod, target)
    elif target.kind.name == "hip":
        device_mod = tvm.ffi.get_global_func("target.build.tilelang_hip")(device_mod, target)
    elif target.kind.name == "metal":
        device_mod = tvm.ffi.get_global_func("target.build.metal")(device_mod, target)
    elif target.kind.name == "riscv_ame" or (hasattr(target, 'keys') and "riscv_ame" in target.keys):
        device_mod = tvm.ffi.get_global_func("target.build.tilelang_riscv_ame")(device_mod, target)
    else:
        raise ValueError(f"Target {target.kind.name} is not supported")

    return device_mod


def device_codegen_without_compile(device_mod: tvm.IRModule, target: Target) -> tvm.IRModule:
    device_mod = tilelang.transform.LowerDeviceStorageAccessInfo()(device_mod)
    device_mod = tilelang.transform.LowerIntrin()(device_mod)
    device_mod = tir.transform.Simplify()(device_mod)
    device_mod = tilelang.transform.HoistBroadcastValues()(device_mod)

    if target.kind.name == "cuda":
        global_func = "target.build.tilelang_" + ("cutedsl" if "cutedsl" in target.keys else "cuda") + "_without_compile"
        device_mod = tvm.ffi.get_global_func(global_func)(device_mod, target)
    elif target.kind.name == "hip":
        device_mod = tvm.ffi.get_global_func("target.build.tilelang_hip_without_compile")(device_mod, target)
    elif target.kind.name == "c":
        device_mod = tvm.ffi.get_global_func("target.build.tilelang_cpp")(device_mod, target)
    elif target.kind.name == "llvm":
        device_mod = tvm.ffi.get_global_func("target.build.llvm")(device_mod, target)
    elif target.kind.name == "webgpu":
        device_mod = tvm.ffi.get_global_func("target.build.webgpu")(device_mod, target)
    elif target.kind.name == "metal":
        device_mod = tvm.ffi.get_global_func("target.build.metal")(device_mod, target)
    elif target.kind.name == "riscv_ame" or (hasattr(target, 'keys') and "riscv_ame" in target.keys):
        device_mod = tvm.ffi.get_global_func("target.build.tilelang_riscv_ame_without_compile")(device_mod, target)
    else:
        raise ValueError(f"Target {target.kind.name} is not supported")

    return device_mod


def lower(
    func_or_mod: tir.PrimFunc | tvm.IRModule,
    target: str | Target = "auto",
    target_host: str | Target | None = None,
    runtime_only=False,
    enable_host_codegen=False,
    enable_device_compile=False,
) -> CompiledArtifact:
    """
    enable_host_codegen: whether to enable host codegen, default is False, as we have our
    own host codegen implementation in jit.
    enable_device_compile: whether to enable device codegen, default is False, as we have our
    own device codegen implementation in jit.
    """

    mod = func_or_mod
    params = None
    if isinstance(func_or_mod, tir.PrimFunc):
        func = func_or_mod
        params = extrac_params(func) if not runtime_only else None
        mod = tvm.IRModule({func.attrs["global_symbol"]: func})

    if isinstance(target, str):
        target = determine_target(target)

    target_host = canon_target_host(target, target_host)

    target_host = tvm.target.Target.canon_target(target_host)
    target = tvm.target.Target(target, target_host)

    _is_host_call = get_host_call(is_device_c=is_cpu_device_backend(target))
    _is_device_call = get_device_call(is_device_c=is_cpu_device_backend(target))

    # Before lowering, do semantic check
    PreLowerSemanticCheck(mod)

    # Phase 1: Lower and legalize the IR
    mod = LowerAndLegalize(mod, target)

    # Phase 2: Optimize the IR for the target
    mod = OptimizeForTarget(mod, target)

    host_mod = tir.transform.Filter(_is_host_call)(mod)
    device_mod = tir.transform.Filter(_is_device_call)(mod)

    codegen_mod = device_codegen(device_mod, target) if enable_device_compile else device_codegen_without_compile(device_mod, target)

    if enable_host_codegen:
        host_mod = host_codegen(host_mod, target_host, target=target)
        host_mod.import_module(codegen_mod)
        return CompiledArtifact(host_mod, device_mod, params, codegen_mod.inspect_source(), rt_mod=host_mod)

    return CompiledArtifact(host_mod, device_mod, params, codegen_mod.inspect_source())
