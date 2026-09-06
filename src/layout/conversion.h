/*!
 * \file conversion.h
 * \brief Checked physical ownership plans for value-preserving conversions.
 */
#ifndef TVM_TL_LAYOUT_CONVERSION_H_
#define TVM_TL_LAYOUT_CONVERSION_H_

#include "layout.h"
#include <vector>

namespace tvm {
namespace tl {

struct FragmentConversionPlan {
  // 0: same-thread registers, 1: warp shuffle, 2: shared-memory exchange.
  int kind{0};
  int threads{0}, elements{0}, source_slots{0}, target_slots{0};
  std::vector<int> source_shape, target_shape;
  // Indexed by physical slot * threads + relative thread ID. -1 is padding.
  std::vector<int> source_values, target_values;
  std::vector<int> source_thread, source_slot;
  std::vector<int> canonical_thread, canonical_slot;
};

FragmentConversionPlan PlanFragmentConversion(const Fragment &source,
                                               const Fragment &target,
                                               int threads,
                                               int max_cells = 1048576);

} // namespace tl
} // namespace tvm
#endif
