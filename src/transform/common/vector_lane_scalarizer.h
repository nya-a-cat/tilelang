/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#ifndef TVM_TL_COMMON_VECTOR_LANE_SCALARIZER_H_
#define TVM_TL_COMMON_VECTOR_LANE_SCALARIZER_H_

#include "support/check.h"
#include <tvm/tirx/expr_functor.h>
#include <tvm/tirx/op.h>

namespace tvm {
namespace tl {

using namespace tirx;

// Extract a fixed vector lane by structure. Arithmetic provers accept the
// resulting scalar expression more reliably than Shuffle::ExtractElement.
class FixedVectorLaneScalarizer : public ExprMutator {
public:
  explicit FixedVectorLaneScalarizer(int lane) : lane_(lane) {}

private:
  int lane_;

  PrimExpr VisitExpr_(const RampNode *op) final {
    PrimExpr base = VisitExpr(op->base);
    PrimExpr stride = VisitExpr(op->stride);
    return base + stride * IntImm(stride.dtype(), lane_);
  }

  PrimExpr VisitExpr_(const BroadcastNode *op) final {
    return VisitExpr(op->value);
  }

  PrimExpr VisitExpr_(const ShuffleNode *op) final {
    ICHECK_LT(lane_, op->indices.size());
    const int64_t *idx = as_const_int(op->indices[lane_]);
    ICHECK(idx) << "Vector lane scalarization requires constant Shuffle "
                   "indices: "
                << ffi::GetRef<Shuffle>(op);
    int64_t src_lane = *idx;
    for (const PrimExpr &vec : op->vectors) {
      ICHECK(!vec.dtype().is_scalable_vector());
      int lanes = vec.dtype().lanes();
      if (src_lane < lanes) {
        if (vec.dtype().is_scalar()) {
          ICHECK_EQ(src_lane, 0);
          return VisitExpr(vec);
        }
        return FixedVectorLaneScalarizer(static_cast<int>(src_lane))(vec);
      }
      src_lane -= lanes;
    }
    ICHECK(false) << "Shuffle index out of range: " << ffi::GetRef<Shuffle>(op);
    return PrimExpr();
  }

  PrimExpr VisitExpr_(const CastNode *op) final {
    PrimExpr value = VisitExpr(op->value);
    DataType dtype =
        op->dtype.is_fixed_length_vector() ? op->dtype.element_of() : op->dtype;
    return value.dtype() == dtype ? value : Cast(dtype, value);
  }
};

inline PrimExpr ScalarizeFixedVectorLane(const PrimExpr &expr, int lane) {
  ICHECK(!expr.dtype().is_scalable_vector());
  ICHECK_LT(lane, expr.dtype().lanes());
  return FixedVectorLaneScalarizer(lane)(expr);
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_COMMON_VECTOR_LANE_SCALARIZER_H_
