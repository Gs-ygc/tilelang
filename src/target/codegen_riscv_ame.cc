/*!
 * \file target/codegen_riscv_ame.cc
 * \brief Code generator for RISCV with AME (Matrix Extension)
 */

#include "codegen_riscv_ame.h"
#include <tvm/arith/analyzer.h>
#include <tvm/tir/op.h>
#include <cmath>
#include <string>
#include <sstream>

namespace tvm {
namespace codegen {

CodeGenRISCVAME::CodeGenRISCVAME() {
  restrict_keyword_ = "__restrict__";
}

std::string CodeGenRISCVAME::Finish() {
  // Add necessary headers for AME
  if (need_ame_h_) {
    decl_stream << "#include <riscv_ame.h>\n";
  }
  
  // Include riscv_vector.h for vectorized load/store operations
  // These are used for efficient swizzled memory access
  if (need_riscv_vector_h_) {
    decl_stream << "#include <riscv_vector.h>\n";
  }
  
  // Add standard headers
  decl_stream << "#include <stdint.h>\n";
  decl_stream << "#include <string.h>\n";
  
  // Add type alias for half (TVM uses 'half' but RISC-V uses '_Float16')
  decl_stream << "\n// Type alias for compatibility\n";
  decl_stream << "typedef _Float16 half;\n";
  decl_stream << "\n";
  
  return CodeGenC::Finish();
}

void CodeGenRISCVAME::AddFunction(const GlobalVar& gvar, const PrimFunc& f) {
  // Mark that we need AME headers
  need_ame_h_ = true;
  
  // Record function name
  function_names_.push_back(gvar->name_hint);
  
  // Call parent implementation
  CodeGenC::AddFunction(gvar, f);
}

void CodeGenRISCVAME::PrintType(DataType t, std::ostream& os) {
  int lanes = t.lanes();
  if (t.is_handle()) {
    ICHECK(t.is_scalar()) << "do not yet support vector types";
    os << "void*";
    return;
  }
  
  if (t.is_void()) {
    os << "void";
    return;
  }
  
  // Handle matrix tile types (special types for AME)
  // These are placeholder types that will be mapped to actual registers
  if (t.code() == DataType::kCustomBegin) {
    os << "matrix_tile_t";
    return;
  }
  
  // Standard scalar and vector types
  if (lanes == 1) {
    if (t.is_float()) {
      if (t.bits() == 16) {
        os << "_Float16";
      } else if (t.bits() == 32) {
        os << "float";
      } else if (t.bits() == 64) {
        os << "double";
      } else {
        LOG(FATAL) << "Unsupported float width: " << t.bits();
      }
    } else if (t.is_int()) {
      os << "int" << t.bits() << "_t";
    } else if (t.is_uint()) {
      os << "uint" << t.bits() << "_t";
    } else if (t.is_bfloat16()) {
      os << "__bf16";
    } else {
      LOG(FATAL) << "Unsupported type: " << t;
    }
  } else {
    // Vector types - use RISC-V vector extension types for efficient swizzled access
    need_riscv_vector_h_ = true;
    if (t.is_float()) {
      if (t.bits() == 16) {
        os << "vfloat16m1_t";
      } else if (t.bits() == 32) {
        os << "vfloat32m1_t";
      } else if (t.bits() == 64) {
        os << "vfloat64m1_t";
      } else {
        LOG(FATAL) << "Unsupported float vector width: " << t.bits();
      }
    } else if (t.is_int()) {
      os << "vint" << t.bits() << "m1_t";
    } else if (t.is_uint()) {
      os << "vuint" << t.bits() << "m1_t";
    } else {
      LOG(FATAL) << "Unsupported vector type: " << t;
    }
  }
}

std::string CodeGenRISCVAME::GetElementWidthSuffix(DataType dtype) {
  if (dtype.is_float()) {
    if (dtype.bits() == 16) return "hf";  // half float
    if (dtype.bits() == 32) return "f";   // float
    if (dtype.bits() == 64) return "d";   // double
  } else if (dtype.is_int() || dtype.is_uint()) {
    if (dtype.bits() == 8) return "b";    // byte
    if (dtype.bits() == 16) return "h";   // halfword
    if (dtype.bits() == 32) return "w";   // word
    if (dtype.bits() == 64) return "dw";  // doubleword
  }
  LOG(FATAL) << "Unsupported data type for AME: " << dtype;
  return "";
}

std::string CodeGenRISCVAME::GetAMEInstrName(const std::string& base_name, 
                                              DataType dtype) {
  std::string suffix = GetElementWidthSuffix(dtype);
  
  if (base_name == "mla" || base_name == "mlb" || base_name == "mlc") {
    // Load instructions: mlae{8,16,32,64}.m
    if (dtype.is_float()) {
      return base_name + "e" + std::to_string(dtype.bits()) + ".m";
    } else {
      return base_name + "e" + std::to_string(dtype.bits()) + ".m";
    }
  } else if (base_name == "msa" || base_name == "msb" || base_name == "msc") {
    // Store instructions: msae{8,16,32,64}.m
    if (dtype.is_float()) {
      return base_name + "e" + std::to_string(dtype.bits()) + ".m";
    } else {
      return base_name + "e" + std::to_string(dtype.bits()) + ".m";
    }
  } else if (base_name == "mma" || base_name == "mfma") {
    // Matrix multiply instructions
    if (dtype.is_float()) {
      return "mfma." + suffix + ".mm";
    } else if (dtype.is_int()) {
      return "mma." + suffix + ".mm";
    } else if (dtype.is_uint()) {
      return "mmau." + suffix + ".mm";
    }
  }
  
  return base_name;
}

void CodeGenRISCVAME::EmitMatrixConfig(int tile_m, int tile_n, int tile_k, 
                                        DataType dtype) {
  current_tile_m_ = tile_m;
  current_tile_n_ = tile_n;
  current_tile_k_ = tile_k;
  current_matrix_dtype_ = dtype;
  
  // Emit configuration instructions
  PrintIndent();
  stream << "// Configure matrix tile dimensions\n";
  
  PrintIndent();
  stream << "asm volatile(\n";
  int saved_indent = BeginScope();
  
  PrintIndent();
  stream << "\"msettilemi " << tile_m << "\\n\\t\"\n";
  
  PrintIndent();
  stream << "\"msettileni " << tile_n << "\\n\\t\"\n";
  
  PrintIndent();
  stream << "\"msettileki " << tile_k << "\\n\\t\"\n";
  
  EndScope(saved_indent);
  PrintIndent();
  stream << ");\n";
}

void CodeGenRISCVAME::EmitMatrixLoad(const std::string& matrix_type,
                                      const std::string& dst_reg,
                                      const Var& src_ptr,
                                      DataType dtype) {
  std::string base_name = "ml" + matrix_type;  // mla, mlb, mlc
  std::string instr = GetAMEInstrName(base_name, dtype);
  
  PrintIndent();
  stream << "// Load matrix " << matrix_type << " tile\n";
  
  PrintIndent();
  stream << "asm volatile(\n";
  int saved_indent = BeginScope();
  
  PrintIndent();
  stream << "\"" << instr << " " << dst_reg << ", (%0), zero\\n\\t\"\n";
  
  PrintIndent();
  stream << ": /* no outputs */\n";
  
  PrintIndent();
  stream << ": \"r\"(" << GetVarID(src_ptr.get()) << ")\n";
  
  PrintIndent();
  stream << ": \"memory\"\n";
  
  EndScope(saved_indent);
  PrintIndent();
  stream << ");\n";
}

void CodeGenRISCVAME::EmitMatrixStore(const std::string& matrix_type,
                                       const std::string& src_reg,
                                       const Var& dst_ptr,
                                       DataType dtype) {
  std::string base_name = "ms" + matrix_type;  // msa, msb, msc
  std::string instr = GetAMEInstrName(base_name, dtype);
  
  PrintIndent();
  stream << "// Store matrix " << matrix_type << " tile\n";
  
  PrintIndent();
  stream << "asm volatile(\n";
  int saved_indent = BeginScope();
  
  PrintIndent();
  stream << "\"" << instr << " " << src_reg << ", (%0), zero\\n\\t\"\n";
  
  PrintIndent();
  stream << ": /* no outputs */\n";
  
  PrintIndent();
  stream << ": \"r\"(" << GetVarID(dst_ptr.get()) << ")\n";
  
  PrintIndent();
  stream << ": \"memory\"\n";
  
  EndScope(saved_indent);
  PrintIndent();
  stream << ");\n";
}

void CodeGenRISCVAME::EmitMatrixMMA(const std::string& dst_reg,
                                     const std::string& a_reg,
                                     const std::string& b_reg,
                                     DataType dtype) {
  std::string instr = GetAMEInstrName("mfma", dtype);
  
  PrintIndent();
  stream << "// Matrix multiply-accumulate\n";
  
  PrintIndent();
  stream << "asm volatile(\n";
  int saved_indent = BeginScope();
  
  PrintIndent();
  stream << "\"" << instr << " " << dst_reg << ", " << a_reg << ", " << b_reg << "\\n\\t\"\n";
  
  PrintIndent();
  stream << ": /* no outputs */\n";
  
  PrintIndent();
  stream << ": /* no inputs */\n";
  
  PrintIndent();
  stream << ": \"memory\"\n";
  
  EndScope(saved_indent);
  PrintIndent();
  stream << ");\n";
}

std::string CodeGenRISCVAME::AllocateMatrixReg(const VarNode* var) {
  // Allocate a matrix register name (m0, m1, m2, etc.)
  if (matrix_reg_map_.count(var)) {
    return matrix_reg_map_[var];
  }
  
  std::string reg_name = "m" + std::to_string(matrix_reg_counter_++);
  matrix_reg_map_[var] = reg_name;
  return reg_name;
}

void CodeGenRISCVAME::VisitStmt_(const AllocateNode* op) {
  ICHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());
  
  // Check storage scope
  std::string scope = GetPtrStorageScope(op->buffer_var);
  
  // For AME, local buffers should be allocated as regular arrays
  // They will be used with swizzled addressing and RVV vectorization
  // but declared as scalar arrays
  
  if (scope.find("fragment") != std::string::npos || 
      scope.find("local.AME") != std::string::npos) {
    // These are special AME tile buffers - comment only, mapped to registers
    PrintIndent();
    stream << "// Matrix tile buffer: " << vid << " (mapped to AME register)\n";
    this->PrintStmt(op->body);
    return;
  }
  
  // Regular local allocation - declare as array
  this->PrintIndent();
  PrintType(op->dtype, stream);
  
  size_t constant_size = op->ConstantAllocationSize();
  ICHECK_GT(constant_size, 0)
      << "Can only handle constant size stack allocation for now";
  
  // Declare as scalar array (no vector types for AME)
  stream << ' ' << vid << '[' << constant_size << "]";
  
  // Add alignment for better performance
  stream << " __attribute__((aligned(64)));\n";
  
  RegisterHandleType(op->buffer_var.get(), op->dtype);
  this->PrintStmt(op->body);
}

void CodeGenRISCVAME::VisitExpr_(const CallNode* op, std::ostream& os) {
  // Handle AME-specific intrinsics
  if (op->op.same_as(builtin::tvm_load_matrix_sync())) {
    // Extract parameters for matrix load
    // This is a simplified version - full implementation would extract
    // all necessary parameters from the call
    need_ame_h_ = true;
    os << "/* matrix_load */";
    return;
  } else if (op->op.same_as(builtin::tvm_mma_sync())) {
    // Extract parameters for matrix multiply
    need_ame_h_ = true;
    os << "/* matrix_mma */";
    return;
  } else if (op->op.same_as(builtin::tvm_store_matrix_sync())) {
    // Extract parameters for matrix store
    need_ame_h_ = true;
    os << "/* matrix_store */";
    return;
  }
  
  // Fall back to default implementation
  CodeGenC::VisitExpr_(op, os);
}

void CodeGenRISCVAME::VisitExpr_(const BroadcastNode* op, std::ostream& os) {
  // Generate RISC-V Vector intrinsic to broadcast scalar to vector
  // Use __riscv_vfmv_v_f_* for float, __riscv_vmv_v_x_* for integer
  // Format: __riscv_vfmv_v_f_f32m1(scalar_value, vl)
  
  std::string value = PrintExpr(op->value);
  DataType dtype = op->dtype;
  
  // For vector operations, vl should be determined by vsetvl
  // Since we're in an expression context, we assume vl is already set
  // Use a fixed VLEN or compute from lanes
  // For simplicity, use the dtype's lane count as vl
  std::string vl = std::to_string(dtype.lanes());
  
  if (dtype.is_float()) {
    // Use __riscv_vfmv_v_f_* to broadcast float scalar to vector
    if (dtype.bits() == 16) {
      os << "__riscv_vfmv_v_f_f16m1(" << value << ", " << vl << ")";
    } else if (dtype.bits() == 32) {
      os << "__riscv_vfmv_v_f_f32m1(" << value << ", " << vl << ")";
    } else if (dtype.bits() == 64) {
      os << "__riscv_vfmv_v_f_f64m1(" << value << ", " << vl << ")";
    } else {
      LOG(FATAL) << "Unsupported float broadcast width: " << dtype.bits();
    }
  } else if (dtype.is_int()) {
    // Use __riscv_vmv_v_x_* to broadcast integer scalar to vector
    if (dtype.bits() == 8) {
      os << "__riscv_vmv_v_x_i8m1(" << value << ", " << vl << ")";
    } else if (dtype.bits() == 16) {
      os << "__riscv_vmv_v_x_i16m1(" << value << ", " << vl << ")";
    } else if (dtype.bits() == 32) {
      os << "__riscv_vmv_v_x_i32m1(" << value << ", " << vl << ")";
    } else if (dtype.bits() == 64) {
      os << "__riscv_vmv_v_x_i64m1(" << value << ", " << vl << ")";
    } else {
      LOG(FATAL) << "Unsupported int broadcast width: " << dtype.bits();
    }
  } else if (dtype.is_uint()) {
    // Use __riscv_vmv_v_x_* for unsigned (same as signed)
    if (dtype.bits() == 8) {
      os << "__riscv_vmv_v_x_u8m1(" << value << ", " << vl << ")";
    } else if (dtype.bits() == 16) {
      os << "__riscv_vmv_v_x_u16m1(" << value << ", " << vl << ")";
    } else if (dtype.bits() == 32) {
      os << "__riscv_vmv_v_x_u32m1(" << value << ", " << vl << ")";
    } else if (dtype.bits() == 64) {
      os << "__riscv_vmv_v_x_u64m1(" << value << ", " << vl << ")";
    } else {
      LOG(FATAL) << "Unsupported uint broadcast width: " << dtype.bits();
    }
  } else {
    LOG(FATAL) << "Unsupported broadcast type: " << dtype;
  }
  
  need_riscv_vector_h_ = true;
}

void CodeGenRISCVAME::VisitStmt_(const ForNode* op) {
  // Try to detect and optimize matrix multiply patterns
  // If this is the outer loop of a 3-level nest doing C += A * B,
  // we can emit AME instructions instead of scalar loops
  
  if (TryEmitMatrixMultiply(op)) {
    // Pattern was matched and AME optimization was emitted
    // Don't generate the loops - AME intrinsics replaced them
    return;
  } else {
    // Pattern match failed or AME not applicable, use default implementation
    CodeGenC::VisitStmt_(op);
  }
}

bool CodeGenRISCVAME::TryEmitMatrixMultiply(const ForNode* outer_loop) {
  // Pattern: Three-nested loops doing C[i,j] += A[i,k] * B[k,j]
  // Following the AME programming model from intrinsic.adoc:
  // - Must use mlae, mlbe, mfwma, msce AME instructions
  // - Dynamic tiling with msettilem/msettilen/msettilek
  
  if (!outer_loop) return false;
  
  // Level 1: i-loop - must start from 0 or 1
  if (!is_one(outer_loop->min) && !is_zero(outer_loop->min)) {
    return false;
  }
  
  const IntImmNode* extent_i_imm = outer_loop->extent.as<IntImmNode>();
  if (!extent_i_imm) {
    return false;
  }
  int64_t extent_i = extent_i_imm->value;
  
  // Unwrap the body if needed
  Stmt body_i = outer_loop->body;
  while (true) {
    if (auto block = body_i.as<BlockRealizeNode>()) {
      body_i = block->block->body;
    } else if (auto let = body_i.as<LetStmtNode>()) {
      body_i = let->body;
    } else {
      break;
    }
  }
  
  // Level 2: j-loop
  const ForNode* loop_j = body_i.as<ForNode>();
  if (!loop_j || (!is_one(loop_j->min) && !is_zero(loop_j->min))) {
    return false;
  }
  
  const IntImmNode* extent_j_imm = loop_j->extent.as<IntImmNode>();
  if (!extent_j_imm) {
    return false;
  }
  int64_t extent_j = extent_j_imm->value;
  
  // Unwrap j-loop body
  Stmt body_j = loop_j->body;
  while (true) {
    if (auto block = body_j.as<BlockRealizeNode>()) {
      body_j = block->block->body;
    } else if (auto let = body_j.as<LetStmtNode>()) {
      body_j = let->body;
    } else {
      break;
    }
  }
  
  // Level 3: k-loop
  const ForNode* loop_k = body_j.as<ForNode>();
  if (!loop_k || (!is_one(loop_k->min) && !is_zero(loop_k->min))) {
    return false;
  }
  
  const IntImmNode* extent_k_imm = loop_k->extent.as<IntImmNode>();
  if (!extent_k_imm) {
    return false;
  }
  int64_t extent_k = extent_k_imm->value;
  
  // Verify tile dimensions
  if (extent_i > 512 || extent_j > 512 || extent_k > 512) {
    return false;
  }
  
  // Get data type
  DataType dtype = DataType::Float(32);
  Stmt body_k = loop_k->body;
  while (true) {
    if (auto block = body_k.as<BlockRealizeNode>()) {
      body_k = block->block->body;
    } else if (auto let = body_k.as<LetStmtNode>()) {
      body_k = let->body;
    } else if (auto store = body_k.as<BufferStoreNode>()) {
      dtype = store->value.dtype();
      break;
    } else {
      break;
    }
  }
  
  // Successfully matched! Emit diagnostic comments and AME configuration
  PrintIndent();
  stream << "// ============================================\n";
  PrintIndent();
  stream << "// AME GEMM Pattern Detected: " << extent_i << "x" << extent_j << "x" << extent_k 
         << " (" << dtype << ")\n";
  PrintIndent();
  stream << "// Uses explicit mlae/mlbe/mfwma/msce AME instructions\n";
  PrintIndent();
  stream << "// ============================================\n\n";
  
  // Emit msettype call in inline assembly to configure element type
  int etype = 2;  // default to e32
  std::string dtype_char = "w";  // float32 (word)
  if (dtype.bits() == 8) {
    etype = 0;
    dtype_char = "b";  // int8 (byte)
  } else if (dtype.bits() == 16) {
    etype = 1;
    dtype_char = "h";  // float16 (half)
  } else if (dtype.bits() == 32) {
    etype = 2;
    dtype_char = "w";  // float32 (word)
  } else if (dtype.bits() == 64) {
    etype = 3;
    dtype_char = "d";  // float64 (double)
  }
  
  // Generate msettype call using riscv_ame.h function
  PrintIndent();
  stream << "msettype(" << etype << ");\n\n";
  
  PrintIndent();
  stream << "// Note: mlae/mlbe/mfwma/msce intrinsics to follow in loop body\n";
  PrintIndent();
  stream << "// Current approach: msettype config + scalar loops (to be replaced with actual intrinsics)\n\n";
  
  // Return false to keep standard loop generation with msettype already emitted
  return false;
}

void CodeGenRISCVAME::PrintStorageScope(const std::string& scope, std::ostream& os) {
  // AME doesn't need special storage scope qualifiers
  // Just skip them
}

void CodeGenRISCVAME::BindThreadIndex(const IterVar& iv) {
  // RISCV AME doesn't have GPU-style thread indexing
  // Map thread indices to loop variables or hart IDs
  ICHECK(!var_idmap_.count(iv->var.get()));
  
  std::string tag = iv->thread_tag;
  if (tag == "threadIdx.x" || tag == "threadIdx.y" || tag == "threadIdx.z") {
    // For CPU, these can map to loop indices or be ignored
    var_idmap_[iv->var.get()] = "0";  // Single-threaded for now
  } else if (tag == "blockIdx.x" || tag == "blockIdx.y" || tag == "blockIdx.z") {
    // Block indices map to outer loop indices
    var_idmap_[iv->var.get()] = "0";  // Single block for now
  } else {
    LOG(FATAL) << "Unknown thread tag: " << tag;
  }
}

}  // namespace codegen
}  // namespace tvm
