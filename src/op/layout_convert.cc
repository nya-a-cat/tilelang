/*!
 * \file layout_convert.cc
 * \brief Lower checked fragment conversions to registers, shuffle or shared.
 */
#include "layout_convert.h"
#include "utils.h"
#include "../layout/conversion.h"
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt_functor.h>
#include <algorithm>

namespace tvm {
namespace tl {
using namespace tirx;
using namespace ffi;
namespace {
Array<PrimExpr> Indices(int flat, const std::vector<int> &shape) {
  Array<PrimExpr> result;
  std::vector<int> indices(shape.size());
  for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
    indices[d] = flat % shape[d];
    flat /= shape[d];
  }
  for (int i : indices)
    result.push_back(Integer(i));
  return result;
}

PrimExpr Lookup(const std::vector<int> &values, PrimExpr thread) {
  ICHECK(!values.empty());
  if (values.size() > 1) {
    int delta = values[1] - values[0];
    bool affine = true;
    for (size_t i = 2; i < values.size(); ++i)
      affine &= values[i] == values[0] + static_cast<int>(i) * delta;
    if (affine)
      return Integer(values[0]) + thread * Integer(delta);
  }
  PrimExpr result = Integer(values.back());
  // Runs of identical values share a comparison; simplify later folds affine
  // patterns and constant tables. Register accesses always use constant slots.
  for (int end = static_cast<int>(values.size()) - 1; end > 0;) {
    int start = end;
    while (start > 0 && values[start - 1] == values[end])
      --start;
    if (start == 0)
      break;
    result = Select(thread < Integer(start), Integer(values[start - 1]), result);
    end = start - 1;
  }
  return result;
}

PrimExpr ShuffleBits(PrimExpr value, PrimExpr lane) {
  DataType dtype = value.dtype();
  ICHECK_EQ(dtype.lanes(), 1);
  ICHECK(dtype.bits() == 8 || dtype.bits() == 16 || dtype.bits() == 32 || dtype.bits() == 64);
  DataType bits_type = DataType::UInt(dtype.bits());
  auto bits = reinterpret(bits_type, value);
  auto shuffle = [&](PrimExpr word) {
    return Call(DataType::UInt(32), builtin::tvm_warp_shuffle(),
                {IntImm(DataType::UInt(32), 0xffffffffU), word, lane, Integer(32), Integer(32)});
  };
  PrimExpr result;
  if (dtype.bits() == 64) {
    auto low = cast(bits_type, shuffle(cast(DataType::UInt(32), bits)));
    auto high = cast(bits_type, shuffle(cast(DataType::UInt(32), bits >> Integer(32))));
    result = low | (high << Integer(32));
  } else {
    result = cast(bits_type, shuffle(cast(DataType::UInt(32), bits)));
  }
  return reinterpret(dtype, result);
}
} // namespace

LayoutConvert::LayoutConvert(Array<PrimExpr> args, Map<String, ObjectRef>) {
  ICHECK_EQ(args.size(), 2);
  auto node = make_object<LayoutConvertNode>();
  auto src = NormalizeToAccessRegion(args[0], kAccessRead);
  auto dst = NormalizeToAccessRegion(args[1], kAccessWrite);
  node->source = src.region->buffer;
  node->target = dst.region->buffer;
  ICHECK(IsFragmentBuffer(node->source) && IsFragmentBuffer(node->target));
  ICHECK(node->source->dtype == node->target->dtype);
  ICHECK(!node->source->data.same_as(node->target->data))
      << "layout conversion requires distinct storage";
  arith::Analyzer analyzer;
  for (const auto &region : {src.region, dst.region}) {
    ICHECK_EQ(region->region.size(), region->buffer->shape.size());
    for (size_t d = 0; d < region->region.size(); ++d) {
      ICHECK(analyzer.CanProveEqual(region->region[d]->min, Integer(0)) &&
             analyzer.CanProveEqual(region->region[d]->extent, region->buffer->shape[d]))
          << "layout conversion requires the full logical buffer";
    }
  }
  node->SetAccessRegions({src, dst});
  data_ = std::move(node);
}

TileOperator LayoutConvertNode::Clone() const {
  return LayoutConvert(make_object<LayoutConvertNode>(*this));
}

Stmt LayoutConvertNode::Lower(const LowerArgs &args, arith::Analyzer *analyzer) const {
  ICHECK_EQ(args.target->GetTargetDeviceType(), kDLCUDA);
  auto source_layout = Downcast<Fragment>(args.layout_map[source]);
  auto target_layout = Downcast<Fragment>(args.layout_map[target]);
  int threads = args.thread_bounds->extent.as<IntImmNode>()->value;
  auto plan = PlanFragmentConversion(source_layout, target_layout, threads);
  auto src = args.buffer_remap[source], dst = args.buffer_remap[target];
  PrimExpr thread = args.thread_index - args.thread_bounds->min;
  Array<Stmt> statements;
  if (plan.kind == 2) {
    ICHECK(args.add_workspace);
    auto pointer = args.add_workspace(plan.elements, source->dtype);
    auto data = GetVarFromAccessPtr(pointer);
    Buffer scratch(data, source->dtype, {Integer(plan.elements)}, {}, Integer(0),
                    "layout_exchange", 0, 1, BufferType::kDefault);
    for (int slot = 0; slot < plan.source_slots; ++slot) {
      std::vector<int> logical(threads, -1);
      for (int t = 0; t < threads; ++t) {
        int element = plan.source_values[slot * threads + t];
        if (element >= 0 && plan.canonical_thread[element] == t && plan.canonical_slot[element] == slot)
          logical[t] = element;
      }
      if (std::all_of(logical.begin(), logical.end(), [](int i) { return i < 0; }))
        continue;
      PrimExpr index = analyzer->Simplify(Lookup(logical, thread));
      auto store = BufferStore(scratch, BufferLoad(src, Indices(slot, plan.source_shape)), {index});
      statements.push_back(IfThenElse(index >= Integer(0), store));
    }
    auto sync = Evaluate(Call(DataType::Int(32), builtin::tvm_storage_sync(), {StringImm("shared")}));
    statements.push_back(sync);
    for (int slot = 0; slot < plan.target_slots; ++slot) {
      std::vector<int> logical(plan.target_values.begin() + slot * threads,
                                plan.target_values.begin() + (slot + 1) * threads);
      PrimExpr index = analyzer->Simplify(Lookup(logical, thread));
      statements.push_back(IfThenElse(index >= Integer(0),
          BufferStore(dst, BufferLoad(scratch, {index}), Indices(slot, plan.target_shape))));
    }
    // Protect reuse by later conversions and loop iterations.
    statements.push_back(sync);
  } else {
    for (int slot = 0; slot < plan.target_slots; ++slot) {
      std::vector<int> source_slots(plan.source_slot.begin() + slot * threads,
                                    plan.source_slot.begin() + (slot + 1) * threads);
      std::vector<int> source_lanes(threads);
      for (int t = 0; t < threads; ++t)
        source_lanes[t] = std::max(0, plan.source_thread[slot * threads + t]) % 32;
      PrimExpr selected = analyzer->Simplify(Lookup(source_slots, thread));
      PrimExpr lane = analyzer->Simplify(Lookup(source_lanes, thread));
      PrimExpr value = make_zero(source->dtype);
      auto unique_slots = source_slots;
      std::sort(unique_slots.begin(), unique_slots.end());
      unique_slots.erase(std::unique(unique_slots.begin(), unique_slots.end()), unique_slots.end());
      for (int source_slot : unique_slots) {
        if (source_slot < 0)
          continue;
        PrimExpr loaded = BufferLoad(src, Indices(source_slot, plan.source_shape));
        if (plan.kind == 1) {
          std::vector<int> valid(threads);
          for (int t = 0; t < threads; ++t)
            valid[t] = plan.source_values[source_slot * threads + t] >= 0;
          loaded = if_then_else(Lookup(valid, thread) != Integer(0), loaded, make_zero(source->dtype));
          // Keep each collective outside lane-dependent control flow.
          Var shuffled("layout_shuffle", source->dtype);
          statements.push_back(Bind(shuffled, ShuffleBits(loaded, lane)));
          loaded = shuffled;
        }
        value = Select(selected == Integer(source_slot), loaded, value);
      }
      statements.push_back(IfThenElse(selected >= Integer(0),
          BufferStore(dst, value, Indices(slot, plan.target_shape))));
    }
  }
  return SeqStmt(statements);
}

Stmt MakeLayoutConversion(const Buffer &source, const Buffer &target) {
  auto region = [](const Buffer &buffer, int mask) {
    Array<PrimExpr> zeros;
    for (size_t i = 0; i < buffer->shape.size(); ++i)
      zeros.push_back(Integer(0));
    Array<PrimExpr> args{BufferLoad(buffer, zeros), Integer(mask)};
    for (const auto &extent : buffer->shape)
      args.push_back(extent);
    return Call(DataType::Handle(), Op::Get("tl.region"), args);
  };
  return Evaluate(Call(DataType::Handle(), LayoutConvert::Get(),
                       {region(source, kAccessRead), region(target, kAccessWrite)}));
}

TIR_REGISTER_TL_TILE_OP(LayoutConvert, layout_convert)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() {
  LayoutConvertNode::RegisterReflection();
  reflection::GlobalDef().def("tl.layout.make_conversion", MakeLayoutConversion);
}
} // namespace tl
} // namespace tvm
