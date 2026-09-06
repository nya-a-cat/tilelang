/*!
 * \file conversion.cc
 * \brief Enumerate and validate ownership for a finite fragment conversion.
 */
#include "conversion.h"
#include "utils.h"
#include <tvm/tirx/stmt_functor.h>
#include <algorithm>
#include <deque>
#include <limits>
#include <stdexcept>

namespace tvm {
namespace tl {
using namespace tirx;
using namespace ffi;
namespace {

int Constant(const PrimExpr &expr, arith::Analyzer *analyzer) {
  auto value = analyzer->Simplify(expr);
  auto imm = value.as<IntImmNode>();
  if (!imm || imm->value < 0 || imm->value > std::numeric_limits<int>::max())
    throw LayoutConflictException("conversion requires finite nonnegative integer coordinates");
  return static_cast<int>(imm->value);
}

std::vector<int> Shape(Array<PrimExpr> shape, arith::Analyzer *analyzer) {
  std::vector<int> result;
  for (const auto &extent : shape) {
    int n = Constant(extent, analyzer);
    if (n < 1)
      throw LayoutConflictException("conversion requires positive static shape");
    result.push_back(n);
  }
  return result;
}

int Product(const std::vector<int> &shape, int budget) {
  int n = 1;
  for (int extent : shape) {
    if (n > budget / extent)
      throw LayoutConflictException("conversion ownership enumeration budget");
    n *= extent;
  }
  return n;
}

std::vector<int> Enumerate(const Fragment &layout, int threads,
                           int max_cells, std::vector<int> *storage_shape) {
  if (layout.as<PartialFragmentNode>())
    throw LayoutConflictException("partial reducer addends are not interchangeable replicas");
  arith::Analyzer analyzer;
  auto shape = Shape(layout->InputShape(), &analyzer);
  *storage_shape = Shape(layout->OutputShape(), &analyzer);
  int elements = Product(shape, max_cells);
  int slots = Product(*storage_shape, max_cells / threads);
  int replicas = Constant(layout->ReplicateExtent(), &analyzer);
  if (replicas < 1 || elements > max_cells / replicas)
    throw LayoutConflictException("conversion ownership enumeration budget");
  std::vector<int> values(slots * threads, -1);
  for (int element = 0; element < elements; ++element) {
    Map<Var, PrimExpr> substitute;
    int remaining = element;
    for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
      substitute.Set(InputPlaceholder(d), Integer(remaining % shape[d]));
      remaining /= shape[d];
    }
    for (int r = 0; r < replicas; ++r) {
      substitute.Set(ReplicationPlaceholder(), Integer(r));
      int thread = Constant(Substitute(layout->GetForwardThread(), substitute), &analyzer);
      int slot = Constant(Substitute(layout->GetLinearizedForwardIndex(), substitute), &analyzer);
      if (thread >= threads || slot >= slots)
        throw LayoutConflictException("fragment ownership exceeds allocated physical shape");
      int &value = values[slot * threads + thread];
      if (value != -1 && value != element)
        throw LayoutConflictException("fragment ownership aliases distinct logical elements");
      value = element;
    }
  }
  return values;
}

std::vector<int> CachedEnumerate(const Fragment &layout, int threads,
                                 int max_cells, std::vector<int> *storage_shape) {
  // A domain's same immutable layout is visited for every directed pair.
  // Keep successful ownership enumerations per compiler thread. Strong object
  // references prevent pointer reuse; thread count and budget remain in the key.
  struct Entry {
    Fragment layout;
    int threads, max_cells;
    std::vector<int> shape, values;
  };
  static thread_local std::deque<Entry> cache;
  static thread_local size_t cells = 0;
  constexpr size_t kMaxCells = 8 * 1024 * 1024; // 32 MiB of ownership integers.
  for (const auto &entry : cache) {
    if (entry.layout.same_as(layout) && entry.threads == threads &&
        entry.max_cells == max_cells) {
      *storage_shape = entry.shape;
      return entry.values;
    }
  }
  auto values = Enumerate(layout, threads, max_cells, storage_shape);
  if (values.size() <= kMaxCells) {
    while (!cache.empty() &&
           (cache.size() >= 256 || cells + values.size() > kMaxCells)) {
      cells -= cache.front().values.size();
      cache.pop_front();
    }
    cells += values.size();
    cache.push_back({layout, threads, max_cells, *storage_shape, values});
  }
  return values;
}
} // namespace

FragmentConversionPlan PlanFragmentConversion(const Fragment &source,
                                               const Fragment &target,
                                               int threads, int max_cells) {
  if (threads < 1 || threads > 1024 || max_cells < threads)
    throw LayoutConflictException("invalid conversion thread or enumeration budget");
  FragmentConversionPlan plan;
  plan.threads = threads;
  arith::Analyzer analyzer;
  auto shape = Shape(source->InputShape(), &analyzer);
  if (shape != Shape(target->InputShape(), &analyzer))
    throw LayoutConflictException("conversion must preserve logical tensor shape");
  plan.elements = Product(shape, max_cells);
  plan.source_values = CachedEnumerate(source, threads, max_cells, &plan.source_shape);
  plan.target_values = CachedEnumerate(target, threads, max_cells, &plan.target_shape);
  plan.source_slots = static_cast<int>(plan.source_values.size()) / threads;
  plan.target_slots = static_cast<int>(plan.target_values.size()) / threads;
  std::vector<std::vector<std::pair<int, int>>> owners(plan.elements);
  for (int slot = 0; slot < plan.source_slots; ++slot)
    for (int thread = 0; thread < threads; ++thread) {
      int element = plan.source_values[slot * threads + thread];
      if (element >= 0)
        owners[element].emplace_back(thread, slot);
    }
  for (auto &owner : owners) {
    if (owner.empty())
      throw LayoutConflictException("source layout does not cover a logical value");
    std::sort(owner.begin(), owner.end());
    plan.canonical_thread.push_back(owner.front().first);
    plan.canonical_slot.push_back(owner.front().second);
  }
  plan.source_thread.resize(plan.target_values.size(), -1);
  plan.source_slot.resize(plan.target_values.size(), -1);
  std::vector<bool> covered(plan.elements, false);
  for (int slot = 0; slot < plan.target_slots; ++slot)
    for (int thread = 0; thread < threads; ++thread) {
      int index = slot * threads + thread;
      int element = plan.target_values[index];
      if (element < 0)
        continue;
      covered[element] = true;
      auto best = owners[element].front();
      int kind = 2;
      for (auto candidate : owners[element]) {
        int candidate_kind = candidate.first == thread ? 0 : candidate.first / 32 == thread / 32 ? 1 : 2;
        if (candidate_kind < kind) {
          best = candidate;
          kind = candidate_kind;
        }
      }
      plan.kind = std::max(plan.kind, kind);
      plan.source_thread[index] = best.first;
      plan.source_slot[index] = best.second;
    }
  if (std::find(covered.begin(), covered.end(), false) != covered.end())
    throw LayoutConflictException("target layout does not cover a logical value");
  // A partial final warp needs a more restricted mask; use shared memory.
  if (plan.kind == 1 && threads % 32 != 0)
    plan.kind = 2;
  return plan;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def("tl.layout.conversion_kind", [](Fragment source, Fragment target, int threads) {
    return PlanFragmentConversion(source, target, threads).kind;
  });
}
} // namespace tl
} // namespace tvm
