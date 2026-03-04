#!/usr/bin/env python3
"""
直接测试 RISCV AME 代码生成 - 使用 TVM script
"""
import tilelang
from tilelang import tvm

# 使用 TVMScript 创建简单的矩阵乘法
from tvm.script import tir as T

M, N, K = 16, 16, 16

@T.prim_func
def matmul_tir(
    A: T.Buffer((M, K), "float16"),
    B: T.Buffer((K, N), "float16"),
    C: T.Buffer((M, N), "float16"),
):
    for i, j in T.grid(M, N):
        with T.block("init"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = T.float16(0)
    
    for i, j, k in T.grid(M, N, K):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

# 创建 IRModule
mod = tvm.IRModule({"main": matmul_tir})

# 设置目标
target = tilelang.utils.target.determine_target("riscv_ame", return_object=True)

# 给 IRModule 中的每个函数设置 target 属性
for name, func in mod.functions.items():
    if isinstance(func, tvm.tir.PrimFunc):
        mod[name] = func.with_attr("target", target)

print("="*80)
print("Original TIR:")
print("="*80)
print(mod)
print("\n" + "="*80)
print(f"Target: {target}")
print("="*80)

# 直接调用代码生成（不编译）
print("\n生成 RISCV AME 代码...")
try:
    # 使用完整的 lower 流程
    result = tilelang.lower(mod, target=target, enable_device_compile=False)
    
    # 获取生成的代码
    source = result.inspect_source()
    print("\n" + "="*80)
    print("生成的 C 代码:")
    print("="*80)
    print(source)
    print("="*80)
    
    # 检查 AME 指令
    ame_instructions_found = []
    if "msettilemi" in source:
        ame_instructions_found.append("msettilemi (配置 tile)")
    if "mlae" in source:
        ame_instructions_found.append("mlae.*.m (加载 A)")
    if "mlbe" in source:
        ame_instructions_found.append("mlbe.*.m (加载 B)")
    if "mfma" in source:
        ame_instructions_found.append("mfma.*.mm (矩阵乘加)")
    if "mse" in source:
        ame_instructions_found.append("mse.*.m (存储结果)")
    
    if ame_instructions_found:
        print("\n✅ 找到 AME 指令:")
        for instr in ame_instructions_found:
            print(f"   - {instr}")
    else:
        print("\n⚠️  未找到 AME 指令（使用标量代码）")
    
    print("\n✅ 代码生成成功！")
        
except Exception as e:
    print(f"\n❌ 代码生成失败: {e}")
    import traceback
    traceback.print_exc()
