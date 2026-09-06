/*!
 * \file graph_selection.cc
 * \brief Preserve control-flow scopes while selecting finite operator layouts.
 */
#include "../../layout/solver_config.h"
#include "../../op/operator.h"
#include "../../op/parallel.h"
#include "../common/pipeline_utils.h"
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/transform.h>
#include <unordered_set>

namespace tvm {
namespace tl {
using namespace tirx;
using namespace ffi;
namespace {

class GraphSelection : public StmtExprMutator {
public:
  explicit GraphSelection(PrimFunc function) : function_(function) {
    callback_ = Function::GetGlobal("tl.layout.select_graph_region_v1");
    ICHECK(callback_.has_value()) << "full layout selection callback is unavailable";
  }

  PrimFunc Run() {
    auto body = VisitStmt(function_->body);
    // LayoutInference stores a function-wide map on each SBlock. Keep that
    // convention for all newly allocated physical layout variants as well.
    auto annotate = TypedFunction<ObjectRef(ObjectRef)>([&](ObjectRef node) -> ObjectRef {
      if (auto block = node.as<SBlockNode>()) {
        SBlock value = GetRef<SBlock>(block);
        auto *copy = value.CopyOnWrite();
        auto layouts = copy->annotations.Get(attr::kLayoutMap)
                           ->cast<LayoutMap>();
        for (const auto &[buffer, layout] : additions_)
          layouts.Set(buffer, layout);
        copy->annotations.Set(attr::kLayoutMap, layouts);
        return value;
      }
      return node;
    });
    body = IRTransform(body, nullptr, annotate);
    function_.CopyOnWrite()->body = body;
    return function_;
  }

private:
  bool IsOperator(const Stmt &stmt) {
    if (auto loop = stmt.as<ForNode>(); loop && loop->kind == ForKind::kParallel)
      return true;
    return ParseOperator(stmt, annotations_).defined();
  }

  Stmt Select(const Array<Stmt> &region) {
    ICHECK(!region.empty());
    ICHECK(!allocations_.empty()) << "full layout selection requires an allocation SBlock";
    auto result = (*callback_)(region, function_, layouts_, pins_, annotations_,
                              thread_bounds_, thread_index_, bindings_, in_pipeline_,
                              divergent_scope_)
                      .cast<Map<String, Any>>();
    for (const auto &buffer : result["allocations"].cast<Array<Buffer>>())
      allocations_.back().push_back(buffer);
    for (const auto &[buffer, layout] : result["layouts"].cast<LayoutMap>())
      additions_.Set(buffer, layout);
    return result["statement"].cast<Stmt>();
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> output, region;
    auto flush = [&]() {
      if (!region.empty()) {
        output.push_back(Select(region));
        region.clear();
      }
    };
    for (const auto &stmt : op->seq) {
      if (IsOperator(stmt)) {
        region.push_back(stmt);
      } else {
        flush();
        output.push_back(VisitStmt(stmt));
      }
    }
    flush();
    return SeqStmt::Flatten(output);
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    auto old_annotations = annotations_;
    auto old_layouts = layouts_;
    auto old_pins = pins_;
    annotations_ = op->annotations;
    if (auto value = annotations_.Get(attr::kLayoutMap))
      layouts_ = value->cast<LayoutMap>();
    if (auto value = annotations_.Get(kGraphPinnedLayouts))
      pins_ = value->cast<LayoutMap>();
    allocations_.emplace_back();
    auto block = Downcast<SBlock>(StmtExprMutator::VisitStmt_(op));
    auto *copy = block.CopyOnWrite();
    for (const auto &buffer : allocations_.back())
      copy->alloc_buffers.push_back(buffer);
    allocations_.pop_back();
    annotations_ = old_annotations;
    layouts_ = old_layouts;
    pins_ = old_pins;
    return block;
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    auto old_bounds = thread_bounds_;
    auto old_index = thread_index_;
    bool old_divergent = divergent_scope_;
    if (op->attr_key == tirx::attr::thread_extent) {
      auto axis = Downcast<IterVar>(op->node);
      if (axis->thread_tag == "threadIdx.x") {
        thread_bounds_ = Range::FromMinExtent(Integer(0), op->value);
        thread_index_ = axis->var;
      } else if (axis->thread_tag == "threadIdx.y" || axis->thread_tag == "threadIdx.z") {
        auto extent = op->value.as<IntImmNode>();
        // The conversion primitive currently uses a single linear thread axis.
        // Preserve native storage when a second nontrivial axis is present.
        divergent_scope_ |= !extent || extent->value != 1;
      }
    }
    auto result = StmtExprMutator::VisitStmt_(op);
    thread_bounds_ = old_bounds;
    thread_index_ = old_index;
    divergent_scope_ = old_divergent;
    return result;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    if (op->kind == ForKind::kParallel)
      return Select({GetRef<For>(op)});
    bool old_pipeline = in_pipeline_;
    bool old_divergent = divergent_scope_;
    in_pipeline_ |= GetPipelineNumStages(op).defined();
    divergent_scope_ |= ThreadDependent(op->min) || ThreadDependent(op->extent);
    auto result = StmtExprMutator::VisitStmt_(op);
    in_pipeline_ = old_pipeline;
    divergent_scope_ = old_divergent;
    return result;
  }

  bool ThreadDependent(PrimExpr expression) {
    // Bind values are SSA; expand only until the dependency reaches the
    // logical thread axis. An already-visited Var terminates recursion.
    std::unordered_set<const VarNode *> seen;
    std::function<bool(PrimExpr)> depends = [&](PrimExpr value) {
      return UsesVar(value, [&](const VarNode *var) {
        if (thread_index_.same_as(GetRef<Var>(var)))
          return true;
        if (!seen.insert(var).second)
          return false;
        auto reference = GetRef<Var>(var);
        return bindings_.count(reference) && depends(bindings_[reference]);
      });
    };
    return depends(expression);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    bool old_divergent = divergent_scope_;
    divergent_scope_ |= ThreadDependent(op->condition);
    auto result = StmtExprMutator::VisitStmt_(op);
    divergent_scope_ = old_divergent;
    return result;
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    auto stmt = GetRef<Evaluate>(op);
    return IsOperator(stmt) ? Select({stmt}) : stmt;
  }

  Stmt VisitStmt_(const BindNode *op) final {
    bindings_.Set(op->var, op->value);
    return GetRef<Bind>(op);
  }

  PrimFunc function_;
  Optional<Function> callback_;
  LayoutMap layouts_, pins_, additions_;
  BlockAnnotations annotations_;
  Map<Var, PrimExpr> bindings_;
  Range thread_bounds_ = Range::FromMinExtent(Integer(0), Integer(1));
  PrimExpr thread_index_ = Integer(0);
  bool in_pipeline_{false};
  bool divergent_scope_{false};
  std::vector<Array<Buffer>> allocations_;
};
} // namespace

tvm::transform::Pass GraphLayoutSelection() {
  auto run = [](PrimFunc function, IRModule, tvm::transform::PassContext context) {
    auto strategy = context->GetConfig<String>(kLayoutSolver, String("root")).value();
    if (strategy == "root" || strategy == "maxsat")
      return function;
    ICHECK(strategy == "maxsat-full" || strategy == "treewidth" ||
           strategy == "local" || strategy == "greedy");
    return GraphSelection(function).Run();
  };
  return tirx::transform::CreatePrimFuncPass(run, 0, "tl.GraphLayoutSelection", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def("tl.transform.GraphLayoutSelection", GraphLayoutSelection);
}
} // namespace tl
} // namespace tvm
