/*!
 * \file target/codegen_riscv_ame.h
 * \brief Code generator for RISCV with AME (Matrix Extension)
 */

#ifndef TVM_TARGET_CODEGEN_RISCV_AME_H_
#define TVM_TARGET_CODEGEN_RISCV_AME_H_

#include <tvm/target/codegen.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "target/source/codegen_c.h"

namespace tvm {
namespace codegen {

/*!
 * \brief Code generator for RISCV AME (Matrix Extension) backend.
 * 
 * This code generator targets RISCV CPUs with the AME (Matrix Extension),
 * generating C code with AME intrinsics that can be compiled by LLVM-AME.
 */
class CodeGenRISCVAME : public CodeGenC {
 public:
  CodeGenRISCVAME();
  
  /*!
   * \brief Finalize code generation and return the generated source
   * \return Generated C/C++ source code with AME intrinsics
   */
  std::string Finish() override;
  
  /*!
   * \brief Add a function to the generated code
   * \param gvar Global variable representing the function
   * \param f The function to add
   */
  void AddFunction(const GlobalVar& gvar, const PrimFunc& f) override;
  
  /*!
   * \brief Get the names of all added functions
   * \return Array of function names
   */
  ffi::Array<ffi::String> GetFunctionNames() {
    return function_names_;
  }

 protected:
  /*!
   * \brief Print type declarations for AME-specific types
   * \param t The data type
   * \param os Output stream
   */
  void PrintType(DataType t, std::ostream& os) override;
  
  /*!
   * \brief Visit and generate code for call expressions
   * \param op The call node
   * \param os Output stream
   */
  void VisitExpr_(const CallNode* op, std::ostream& os) override;
  
  /*!
   * \brief Visit and generate code for broadcast expressions
   * \param op The broadcast node
   * \param os Output stream
   */
  void VisitExpr_(const BroadcastNode* op, std::ostream& os) override;
  
  /*!
   * \brief Bind thread index variables
   * \param iv Iterator variable
   */
  void BindThreadIndex(const IterVar& iv) override;
  
  /*!
   * \brief Visit and generate code for allocation statements
   * \param op The allocate node
   */
  void VisitStmt_(const AllocateNode* op) override;
  
  /*!
   * \brief Visit and generate code for for loops
   * \param op The for loop node
   */
  void VisitStmt_(const ForNode* op) override;
  
  /*!
   * \brief Try to emit AME matrix multiply for 3-nested loops
   * \param outer_loop The outermost loop
   * \return true if successfully emitted AME instructions, false otherwise
   */
  bool TryEmitMatrixMultiply(const ForNode* outer_loop);
  
  /*!
   * \brief Extract data type from loop body (BufferLoad/Store nodes)
   * \param stmt The loop body statement
   * \return Detected data type, defaults to float32
   */
  DataType ExtractMatrixDataType(const Stmt& stmt);
  
  /*!
   * \brief Print storage scope
   * \param scope Storage scope string
   * \param os Output stream
   */
  void PrintStorageScope(const std::string& scope, std::ostream& os) override;
  
  /*!
   * \brief Emit matrix configuration instructions
   * \param tile_m Tile size in M dimension
   * \param tile_n Tile size in N dimension  
   * \param tile_k Tile size in K dimension
   * \param dtype Data type of matrix elements
   */
  void EmitMatrixConfig(int tile_m, int tile_n, int tile_k, DataType dtype);
  
  /*!
   * \brief Emit matrix load instruction
   * \param matrix_type Type of matrix (A, B, or C)
   * \param dst_reg Destination register name
   * \param src_ptr Source pointer variable
   * \param dtype Element data type
   */
  void EmitMatrixLoad(const std::string& matrix_type,
                      const std::string& dst_reg,
                      const Var& src_ptr,
                      DataType dtype);
  
  /*!
   * \brief Emit matrix store instruction
   * \param matrix_type Type of matrix (A, B, or C)
   * \param src_reg Source register name
   * \param dst_ptr Destination pointer variable
   * \param dtype Element data type
   */
  void EmitMatrixStore(const std::string& matrix_type,
                       const std::string& src_reg,
                       const Var& dst_ptr,
                       DataType dtype);
  
  /*!
   * \brief Emit matrix multiply-accumulate instruction
   * \param dst_reg Destination accumulator register
   * \param a_reg Source A register
   * \param b_reg Source B register
   * \param dtype Element data type
   */
  void EmitMatrixMMA(const std::string& dst_reg,
                     const std::string& a_reg,
                     const std::string& b_reg,
                     DataType dtype);

  /*!
   * \brief Get AME instruction name for matrix operation
   * \param base_name Base operation name (e.g., "mla", "msa", "mma")
   * \param dtype Element data type
   * \return Full instruction name (e.g., "mlae16.m", "mfma.hf.mm")
   */
  std::string GetAMEInstrName(const std::string& base_name, DataType dtype);
  
  /*!
   * \brief Get element width suffix for AME instructions
   * \param dtype Element data type
   * \return Width suffix (e.g., "8", "16", "32", "64", "hf", "f", "d")
   */
  std::string GetElementWidthSuffix(DataType dtype);

 private:
  /*! \brief Whether AME header is needed */
  bool need_ame_h_{false};
  
  /*! \brief Whether RISCV vector header is needed */
  bool need_riscv_vector_h_{false};
  
  /*! \brief Current tile dimensions */
  int current_tile_m_{16};
  int current_tile_n_{16};
  int current_tile_k_{16};
  
  /*! \brief Current matrix data type */
  DataType current_matrix_dtype_{DataType::Float(16)};
  
  /*! \brief Map from buffer to matrix register name */
  std::unordered_map<const VarNode*, std::string> matrix_reg_map_;
  
  /*! \brief Counter for matrix register allocation */
  int matrix_reg_counter_{0};
  
  /*! \brief List of function names */
  ffi::Array<ffi::String> function_names_;
  
  /*! \brief Set of local tile buffers (for AME register mapping) */
  std::unordered_set<const VarNode*> local_tile_buffers_;
  
  /*! \brief Allocate a new matrix register name */
  std::string AllocateMatrixReg(const VarNode* var);
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_CODEGEN_RISCV_AME_H_
