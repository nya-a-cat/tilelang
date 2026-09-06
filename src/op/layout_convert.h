/*!
 * \file layout_convert.h
 * \brief Internal value-preserving fragment conversion operator.
 */
#ifndef TVM_TL_OP_LAYOUT_CONVERT_H_
#define TVM_TL_OP_LAYOUT_CONVERT_H_
#include "operator.h"

namespace tvm {
namespace tl {
class LayoutConvertNode : public TileOperatorNode {
public:
  tirx::Buffer source, target;
  tirx::Stmt Lower(const LowerArgs &args, arith::Analyzer *analyzer) const final;
  LayoutMap InferLayout(const LayoutInferArgs &, InferLevel) const final { return {}; }
  TileOperator Clone() const final;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.LayoutConvert", LayoutConvertNode, TileOperatorNode);
  static void RegisterReflection() {
    ffi::reflection::ObjectDef<LayoutConvertNode>().def_ro("source", &LayoutConvertNode::source)
        .def_ro("target", &LayoutConvertNode::target);
  }
};
class LayoutConvert : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(LayoutConvert, TileOperator, LayoutConvertNode);
  LayoutConvert(ffi::Array<PrimExpr> args, ffi::Map<ffi::String, ffi::ObjectRef> annotations = {});
  static const Op &Get();
};
tirx::Stmt MakeLayoutConversion(const tirx::Buffer &source, const tirx::Buffer &target);
} // namespace tl
} // namespace tvm
#endif
