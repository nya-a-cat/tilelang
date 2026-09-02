/*!
 * \file layout_cost_model.h
 * \brief Cost models that rank free-mode layout attempts.
 *
 * The inference engine enumerates one attempt per candidate root inside a
 * connected component and keeps the cheapest complete layout assignment.
 * What "cheapest" means is a pluggable policy behind LayoutCostModel:
 *
 *  - RegisterCountCostModel: total fragment register slots, nothing else.
 *  - IOAwareCostModel (layout RFC, design B2): walks the component's
 *    global-memory-touching statements (fragment<->global copies and
 *    parallel loops with direct global accesses) and charges each one
 *    max(bandwidth bytes, issue-equivalent bytes) under the attempt's
 *    tentative layouts; registers remain the lexicographic tiebreak.
 *    Selected by the target-aware default on CUDA and available explicitly
 *    through `tl.layout_cost_model="io-aware"` for A/B comparisons.
 *  - IOAwareRegularizedCostModel: uses the same statement traffic model and
 *    adds a byte-equivalent price for each fragment register slot. This
 *    single-lowering scalarization is available as
 *    `tl.layout_cost_model="io-aware-regularized"` for rollback and A/B.
 *    Other targets keep RegisterCountCostModel as their target-aware default.
 *
 * Concrete models live in the .cc; callers go through Create().
 */

#ifndef TVM_TL_TRANSFORM_LAYOUT_INFERENCE_LAYOUT_COST_MODEL_H_
#define TVM_TL_TRANSFORM_LAYOUT_INFERENCE_LAYOUT_COST_MODEL_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <tvm/target/target.h>

#include "../../op/operator.h"

namespace tvm {
namespace tl {

/*! \brief Score of one complete free-mode layout assignment. Compared
 *  lexicographically: `rank` first, total register count as the tiebreak.
 *  Existing policies set `rank == mem`; a regularized policy may add a
 *  byte-equivalent register charge while leaving `mem` decomposable.
 *  `mem` includes the estimated local-memory traffic of
 *  register-array spills (a thread-dependent register-array index demotes
 *  the whole array to local memory), priced in bytes so it competes
 *  honestly with the io-aware model's global traffic instead of vetoing
 *  it. Models that do not estimate global memory leave the global part at
 *  0, so their ordering is spill bytes, then register count — attempts
 *  without spills keep the historical register-count ordering untouched. */
struct AttemptCost {
  int64_t rank{0};
  int64_t mem{0};
  int64_t regs{0};
  // Diagnostic decomposition of `mem`.  These fields do not participate in
  // ordering; they make a traced attempt suitable for calibrating a future
  // architecture-aware model without changing the current policies.
  int64_t spill{0};
  int64_t global_mem{0};
  int64_t global_bw{0};
  int64_t global_issue{0};
  int64_t measured_statements{0};
  int64_t worst_case_statements{0};
  int64_t unavailable_statements{0};
  int64_t register_price{0};
  bool BetterThan(const AttemptCost &other) const {
    if (rank != other.rank) {
      return rank < other.rank;
    }
    return regs < other.regs;
  }
};

/*! \brief Policy interface: rank one attempt of a component.
 *
 *  `members` indexes the component's operators inside `infer_list` (which
 *  carries the attempt's solved state, e.g. loop layouts), and
 *  `tmp_layout_map` holds the attempt's tentative buffer layouts. */
class LayoutCostModel {
public:
  virtual ~LayoutCostModel() = default;

  virtual AttemptCost Score(const std::vector<int> &members,
                            const std::vector<TileOperator> &infer_list,
                            const LayoutMap &tmp_layout_map) const = 0;

  /*! \brief Model name for diagnostics. */
  virtual const char *Name() const = 0;

  /*! \brief Instantiate the model selected by `tl.layout_cost_model`
   *  by name ("target-default", "io-aware", "io-aware-regularized", or
   *  "register-count"). "target-default" resolves to io-aware on CUDA and
   *  register-count elsewhere.
   *  unknown names are a hard error listing the valid values. `target`
   *  feeds the vectorizer's shared width-cap policy (MaxVectorLoadBits);
   *  the legacy model ignores it. */
  static std::unique_ptr<LayoutCostModel> Create(const std::string &name,
                                                 Target target);
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_LAYOUT_INFERENCE_LAYOUT_COST_MODEL_H_
