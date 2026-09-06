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
#include <unordered_set>

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

class GraphAccessCollector : public StmtExprVisitor {
public:
  Array<Buffer> reads, writes;
  std::unordered_set<const BufferNode *> full_writes;
  GraphAccessCollector(BlockAnnotations annotations, Map<Var, PrimExpr> bindings)
      : annotations_(annotations), bindings_(bindings) {}

  bool FullRegion(const BufferRegion &region) {
    arith::Analyzer analyzer;
    for (size_t d = 0; d < region->region.size(); ++d)
      if (!analyzer.CanProveEqual(region->region[d]->min, Integer(0)) ||
          !analyzer.CanProveEqual(region->region[d]->extent, region->buffer->shape[d]))
        return false;
    return true;
  }

  void AddOperator(const TileOperator &op) {
    for (const auto &region : op->GetReadBeforeWriteRegions())
      reads.push_back(region->buffer);
    for (const auto &region : op->GetAccessRegions().writes) {
      writes.push_back(region->buffer);
      if (!conditional_ && FullRegion(region))
        full_writes.insert(region->buffer.get());
    }
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    reads.push_back(op->buffer);
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(reducer_update())) {
      auto update = ParseReducerUpdate(op);
      reads.push_back(update.reducer);
      writes.push_back(update.reducer);
      VisitExpr(update.value);
      return;
    }
    auto tile = ParseOperator(GetRef<Call>(op), annotations_);
    if (tile.defined()) {
      AddOperator(tile);
      return;
    }
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const BufferStoreNode *op) final {
    writes.push_back(op->buffer);
    if (!conditional_) {
      Array<PrimExpr> indices;
      Map<Var, Range> ranges;
      for (const auto &axis : axes_)
        ranges.Set(axis->var, axis->dom);
      for (const auto &index : op->indices)
        indices.push_back(Substitute(index, bindings_));
      arith::Analyzer analyzer;
      auto mapping = arith::DetectIterMap(indices, ranges, Integer(1),
                                          arith::IterMapLevel::Bijective, &analyzer);
      if (mapping->errors.empty()) {
        auto shape = Layout(axes_, indices)->OutputShape();
        bool full = shape.size() == op->buffer->shape.size();
        for (size_t d = 0; full && d < shape.size(); ++d)
          full &= analyzer.CanProveEqual(mapping->indices[d]->base, Integer(0)) &&
                  analyzer.CanProveEqual(shape[d], op->buffer->shape[d]);
        if (full)
          full_writes.insert(op->buffer.get());
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const ForNode *op) final {
    axes_.push_back(IterVar(Range::FromMinExtent(op->min, op->extent), op->loop_var,
                            IterVarType::kDataPar));
    StmtExprVisitor::VisitStmt_(op);
    axes_.pop_back();
  }
  void VisitStmt_(const BindNode *op) final {
    bindings_.Set(op->var, Substitute(op->value, bindings_));
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const IfThenElseNode *op) final {
    bool previous = conditional_;
    conditional_ = true;
    StmtExprVisitor::VisitStmt_(op);
    conditional_ = previous;
  }
private:
  BlockAnnotations annotations_;
  Map<Var, PrimExpr> bindings_;
  Array<IterVar> axes_;
  bool conditional_{false};
};

Map<String, Any> Accesses(Stmt stmt, BlockAnnotations annotations, Map<Var, PrimExpr> bindings) {
  auto op = GraphOperator(stmt, annotations);
  if (!op.defined())
    return {{"supported", false}};
  GraphAccessCollector collector(annotations, bindings);
  if (op.as<ParallelOpNode>()) {
    collector(stmt);
  } else {
    collector.AddOperator(op);
  }
  // A partial write consumes the old tensor value for its untouched elements.
  // This dependency is required even when the source syntax has no BufferLoad.
  Array<Buffer> partial;
  for (const auto &buffer : collector.writes) {
    if (!collector.full_writes.count(buffer.get())) {
      partial.push_back(buffer);
      if (std::none_of(collector.reads.begin(), collector.reads.end(),
                       [&](const Buffer &read) { return read.same_as(buffer); }))
        collector.reads.push_back(buffer);
    }
  }
  return {{"supported", true}, {"reads", collector.reads}, {"writes", collector.writes},
          {"partial_writes", partial}, {"kind", String(op->GetTypeKey())}};
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
