/*!
 * \file tl/cuda/op/finalize_reducer.cc
 * \brief CUDA implementation for tl.finalize_reducer AllReduce lowering.
 */

#include "backend/common/op/finalize_reducer.h"

#include "cuda/target_utils.h"

#include <sstream>

namespace tvm {
namespace tl {

using namespace tirx;

namespace cuda {

struct FinalizeReducer : backend::FinalizeReducerLowerer<FinalizeReducer> {
  static int WarpSize(Target target) { return TargetCudaGetWarpSize(target); }

  static bool ShouldUseHierarchicalAllReduce(int reducing_threads, int scale,
                                             PrimExpr thread_offset,
                                             PrimExpr all_threads,
                                             Target target) {
    return TargetCudaPrefersHierarchicalAllReduce(target) &&
           backend::reduce::CanUseHierarchicalAllReduce(
               reducing_threads, scale, thread_offset, all_threads,
               TargetCudaGetWarpSize(target));
  }

  static int GetAllReduceWorkspaceStride(int reducing_threads, int scale,
                                         PrimExpr thread_offset,
                                         PrimExpr all_threads,
                                         int fallback_stride, Target target) {
    if (ShouldUseHierarchicalAllReduce(reducing_threads, scale, thread_offset,
                                       all_threads, target)) {
      return reducing_threads / TargetCudaGetWarpSize(target);
    }
    return fallback_stride;
  }

  static bool AllReduceHasLeadingBarrier(int reducing_threads, int scale,
                                         PrimExpr thread_offset,
                                         PrimExpr all_threads, Target target) {
    return !ShouldUseHierarchicalAllReduce(reducing_threads, scale,
                                           thread_offset, all_threads, target);
  }

  static std::string MakeBatchAllReduce(std::string reducer,
                                        int reducing_threads, int scale,
                                        PrimExpr thread_offset,
                                        PrimExpr all_threads, int batch,
                                        int workspace_stride, Target target) {
    bool hierarchical = ShouldUseHierarchicalAllReduce(
        reducing_threads, scale, thread_offset, all_threads, target);
    std::stringstream ss;
    ss << "tl::AllReduce<" << reducer << ", " << reducing_threads << ", "
       << scale << ", " << thread_offset;
    if (TargetSupportsNamedBarrier(target)) {
      ss << ", tl::NamedBarrier<" << all_threads << ">";
    } else {
      ss << ", tl::SyncThreadsBarrier";
    }
    ss << ", " << batch << ", " << workspace_stride;
    if (hierarchical) {
      ss << ", true";
    }
    ss << ">::run_batch";
    return ss.str();
  }

  static std::string MakeScalarAllReduce(std::string reducer,
                                         int reducing_threads, int scale,
                                         PrimExpr thread_offset,
                                         PrimExpr all_threads, Target target) {
    bool hierarchical = ShouldUseHierarchicalAllReduce(
        reducing_threads, scale, thread_offset, all_threads, target);
    std::stringstream ss;
    ss << "tl::AllReduce<" << reducer << ", " << reducing_threads << ", "
       << scale << ", " << thread_offset;
    if (TargetSupportsNamedBarrier(target)) {
      ss << ", tl::NamedBarrier<" << all_threads << ">";
    } else if (hierarchical) {
      ss << ", tl::SyncThreadsBarrier";
    }
    if (hierarchical) {
      ss << ", 1, 0, true";
    }
    ss << ">::run";
    return ss.str();
  }
};

} // namespace cuda

namespace {

bool MatchCudaFinalizeReducerTarget(Target target) {
  return TargetIsCuda(target) || TargetIsCuTeDSL(target);
}

bool RegisterCudaFinalizeReducer() {
  RegisterFinalizeReducerImpl(FinalizeReducerImpl{
      "cuda.FinalizeReducer",
      MatchCudaFinalizeReducerTarget,
      cuda::FinalizeReducer::Lower,
  });
  return true;
}

const bool cuda_finalize_reducer_registered = RegisterCudaFinalizeReducer();

} // namespace

} // namespace tl
} // namespace tvm
