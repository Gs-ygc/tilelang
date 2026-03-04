/*!
 * \file target/rt_mod_riscv_ame.cc
 * \brief Runtime module builder for RISCV AME backend
 */

#include "codegen_riscv_ame.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/target/target_kind.h>

namespace tvm {
namespace codegen {

/*!
 * \brief Extract function information from IR module
 * \param mod The IR module
 * \return Map from function name to function info
 */
static std::unordered_map<std::string, runtime::FunctionInfo>
ExtractFuncInfo(const IRModule& mod) {
  std::unordered_map<std::string, runtime::FunctionInfo> fmap;

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<tir::PrimFuncNode>())
        << "Can only lower IR Module with PrimFuncs";
    auto f = Downcast<tir::PrimFunc>(kv.second);

    runtime::FunctionInfo info;
    for (size_t i = 0; i < f->params.size(); ++i) {
      DataType dtype = f->params[i].dtype();
      // CPU runtime can handle bool, but normalize to int32 for consistency
      if (dtype.is_bool()) {
        dtype = DataType::Int(32);
      }
      info.arg_types.push_back(dtype);
    }
    
    auto global_symbol = f->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
    fmap[static_cast<std::string>(global_symbol.value())] = info;
  }
  return fmap;
}

/*!
 * \brief Build RISCV AME module with compilation
 * \param mod The IR module to build
 * \param target Target specification
 * \return Runtime module
 */
ffi::Module BuildRISCVAME(IRModule mod, Target target) {
  // Generate C code with AME intrinsics
  CodeGenRISCVAME cg;
  cg.Init(false);  // output_ssa = false

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenRISCVAME: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    cg.AddFunction(gvar, f);
  }

  std::string code = cg.Finish();
  
  // Apply post-processing if callback is registered
  if (const auto f = ffi::Function::GetGlobal("tilelang_callback_riscv_ame_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }
  
  // For now, return source-only module
  // In future, this could compile and return a shared library module
  return CSourceModuleCreate(code, "cc", cg.GetFunctionNames());
}

/*!
 * \brief Build RISCV AME module without compilation (source only)
 * \param mod The IR module to build
 * \param target Target specification
 * \return Runtime module containing source code
 */
ffi::Module BuildRISCVAMEWithoutCompile(IRModule mod, Target target) {
  // Generate C code with AME intrinsics
  CodeGenRISCVAME cg;
  cg.Init(false);  // output_ssa = false

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenRISCVAME: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    cg.AddFunction(gvar, f);
  }

  std::string code = cg.Finish();
  
  // Apply post-processing if callback is registered
  if (const auto f = ffi::Function::GetGlobal("tilelang_callback_riscv_ame_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }
  
  // Return source module
  return CSourceModuleCreate(code, "cc", cg.GetFunctionNames());
}

// Register the build functions
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("target.build.tilelang_riscv_ame", BuildRISCVAME)
      .def("target.build.tilelang_riscv_ame_without_compile",
           BuildRISCVAMEWithoutCompile);
}

// Register the riscv_ame target kind
TVM_REGISTER_TARGET_KIND("riscv_ame", kDLCPU)
    .add_attr_option<ffi::String>("mcpu")
    .add_attr_option<ffi::String>("march")
    .add_attr_option<ffi::String>("mattr")
    .add_attr_option<int>("tile_m")
    .add_attr_option<int>("tile_n")
    .add_attr_option<int>("tile_k")
    .set_default_keys({"cpu", "riscv_ame"});

}  // namespace codegen
}  // namespace tvm
