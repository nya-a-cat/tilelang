/*!
 * \file graph_interface.cc
 * \brief Native operator semantics used by finite graph layout selection.
 */
#include "../../op/operator.h"
#include "../../op/parallel.h"
#include "../../op/reducer.h"
#include "../../op/builtin.h"
#include "../../layout/utils.h"
#include <tvm/tirx/stmt_functor.h>

namespace tvm {
namespace tl {
using namespace tirx;
using namespace ffi;
namespace {

TileOperator GraphOperator(const Stmt &stmt, const BlockAnnotations &annotations) {
  if (const auto *loop = stmt.as<ForNode>(); loop && loop->kind == ForKind::kParallel)
    return ParallelOp(GetRef<For>(loop));
  return ParseOperator(stmt, annotations);
}

Map<String, Any> Accesses(Stmt stmt, BlockAnnotations annotations) {
  auto op = GraphOperator(stmt, annotations);
  if (!op.defined())
    return {{"supported", false}};
  Array<Buffer> reads, writes;
  if (auto parallel = op.as<ParallelOpNode>()) {
    for (const auto &buffer : parallel->GetAccessOrder()) {
      const auto &access = parallel->GetIndiceMap().at(buffer);
      if (access.is_read)
        reads.push_back(buffer);
      if (access.is_write)
        writes.push_back(buffer);
    }
    // Reducer state has addend semantics and must participate in dependencies.
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (auto call = node.as<CallNode>(); call && call->op.same_as(reducer_update())) {
        auto update = ParseReducerUpdate(call);
        reads.push_back(update.reducer);
        writes.push_back(update.reducer);
      }
    });
    return {{"supported", true}, {"reads", reads}, {"writes", writes},
            {"kind", String("parallel")}};
  }
  for (const auto &region : op->GetReadBeforeWriteRegions())
    reads.push_back(region->buffer);
  for (const auto &region : op->GetAccessRegions().writes)
    writes.push_back(region->buffer);
  return {{"supported", true}, {"reads", reads}, {"writes", writes},
          {"kind", String(op->GetTypeKey())}};
}

// Each call reparses the operator so mutable inference state is never shared
// across candidate configurations. The caller supplies all authoritative pins.
Map<String, Any> Infer(Stmt stmt, BlockAnnotations annotations, Target target,
                       Range threads, PrimExpr thread_index, LayoutMap seed, LayoutMap pins,
                       Map<Var, PrimExpr> bindings, bool in_pipeline) {
  auto op = GraphOperator(stmt, annotations);
  ICHECK(op.defined());
  arith::Analyzer analyzer;
  LayoutMap layouts = seed;
  for (const auto &[buffer, layout] : pins)
    layouts.Set(buffer, layout);
  try {
    for (auto level : {InferLevel::kStrict, InferLevel::kCommon, InferLevel::kFree}) {
      LayoutInferArgs args{target, threads, layouts, &analyzer, {}, bindings,
                           in_pipeline, pins};
      auto updates = op->InferLayout(args, level);
      for (const auto &[buffer, layout] : updates) {
        if (pins.count(buffer) && !layout->IsEqual(pins[buffer].get()))
          throw LayoutConflictException("operator candidate violates a pinned layout");
        if (seed.count(buffer) && !layout->IsEqual(seed[buffer].get()))
          throw LayoutConflictException("operator candidate changes its supplied layout");
        layouts.Set(buffer, layout);
      }
    }
    Map<String, Any> result{{"valid", true}, {"layouts", layouts}, {"statement", stmt}};
    if (auto parallel = op.as<ParallelOpNode>()) {
      auto layout = parallel->GetLoopLayout();
      if (!layout.defined())
        throw LayoutConflictException("operator candidate has no loop layout");
      For rewritten = Downcast<For>(stmt);
      auto *node = rewritten.CopyOnWrite();
      node->annotations.Set(attr::kParallelLoopLayout, layout);
      node->annotations.Set(attr::kParallelLoopRequiresPaddingGuard,
                            Bool(parallel->LoopLayoutRequiresPaddingGuard()));
      auto predicate = parallel->GetPredicate(thread_index);
      node->annotations.erase(attr::kParallelLoopPredicate);
      if (predicate.defined())
        node->annotations.Set(attr::kParallelLoopPredicate, predicate.value());
      result.Set("statement", rewritten);
    }
    return result;
  } catch (const LayoutConflictException &error) {
    return {{"valid", false}, {"reason", String(error.what())}};
  }
}

// Remap all accesses and underlying data handles together, including region
// intrinsics and pointer expressions. Allocation ownership stays with the pass.
class Remapper : public StmtExprMutator {
public:
  explicit Remapper(Map<Buffer, Buffer> remap) : buffers_(remap) {
    for (const auto &[source, target] : buffers_) {
      if (variables_.count(source->data))
        ICHECK(variables_[source->data].same_as(target->data));
      variables_.Set(source->data, target->data);
    }
  }
  PrimExpr VisitExpr_(const VarNode *op) final {
    auto var = GetRef<Var>(op);
    return variables_.count(var) ? variables_[var] : var;
  }
  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    auto value = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    if (buffers_.count(op->buffer))
      value.CopyOnWrite()->buffer = buffers_[op->buffer];
    return value;
  }
  Stmt VisitStmt_(const BufferStoreNode *op) final {
    auto value = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    if (buffers_.count(op->buffer))
      value.CopyOnWrite()->buffer = buffers_[op->buffer];
    return value;
  }
private:
  Map<Buffer, Buffer> buffers_;
  Map<Var, Var> variables_;
};
} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef()
      .def("tl.layout.graph_accesses", Accesses)
      .def("tl.layout.graph_infer", Infer)
      .def("tl.layout.graph_remap", [](Stmt stmt, Map<Buffer, Buffer> remap) {
        return Remapper(remap)(stmt);
      });
}
} // namespace tl
} // namespace tvm
