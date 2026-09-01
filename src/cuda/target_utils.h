/*!
 * \file tl/cuda/target_utils.h
 * \brief CUDA target attribute helpers.
 */

#ifndef TVM_TL_CUDA_TARGET_UTILS_H_
#define TVM_TL_CUDA_TARGET_UTILS_H_

#include <tvm/runtime/data_type.h>
#include <tvm/target/target.h>

namespace tvm {
namespace tl {

bool TargetIsCuda(Target target);
bool TargetIsCuTeDSL(Target target);

bool TargetIsVolta(Target target);
bool TargetIsTuring(Target target);
bool TargetIsAmpere(Target target);
bool TargetIsHopper(Target target);
bool TargetIsSm100(Target target);
bool TargetIsSM120(Target target);

bool TargetCudaHasAsyncCopy(Target target);
int TargetCudaGetWarpSize(Target target);
bool TargetHasLdmatrix(Target target);
bool TargetHasStmatrix(Target target, bool is_m16n8 = false);
bool TargetHasTmem(Target target);
bool TargetHasBulkCopy(Target target);
bool TargetHasRegReconfiguration(Target target);
bool TargetSupportsNamedBarrier(Target target);
/*! \brief Whether CUDA AllReduce should prefer the hierarchical algorithm.
 *
 * This is the target profitability decision. Collective-specific legality is
 * checked separately by the reduction lowerer. The pass config
 * `tl.cuda_allreduce_strategy` can force either implementation for JIT-time
 * A/B measurements without rebuilding TileLang.
 */
bool TargetCudaPrefersHierarchicalAllReduce(Target target);
/*! \brief Whether hierarchical AllReduce may use hardware warp redux for its
 * second-level aggregate reduction. The pass config
 * `tl.disable_warp_aggregate_redux` provides a same-build rollback path.
 */
bool CudaWarpAggregateReduxEnabled();
bool TargetSupportVectorize256(Target target);
bool TargetHasSMVersionGE(Target target, int version);

bool IsCudaVectorizableFP8(DataType dtype);
bool IsCudaVectorizableCast(DataType from_ty, DataType target_ty);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_CUDA_TARGET_UTILS_H_
