/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file index.cc
 * \brief indexing operators.
 */

#include "index.h"

#include <utility>
#include <vector>

namespace tvm {
namespace relax {

/* relax.take */
TVM_REGISTER_NODE_TYPE(TakeAttrs);

Expr take(Expr data, Expr indices, Optional<Integer> axis) {
  ObjectPtr<TakeAttrs> attrs = make_object<TakeAttrs>();
  attrs->axis = std::move(axis);

  static const Op& op = Op::Get("relax.take");
  return Call(op, {std::move(data), std::move(indices)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.take").set_body_typed(take);

StructInfo InferStructInfoTake(const Call& call, const BlockBuilder& ctx) {
  Array<TensorStructInfo> input_sinfo = GetInputTensorStructInfo(call, ctx);
  TensorStructInfo data_sinfo = input_sinfo[0];
  TensorStructInfo indices_sinfo = input_sinfo[1];
  if (indices_sinfo->ndim != 1) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Take op requires the input indices to be 1-dimensional tensor. However, "
                        "the given indices ndim is "
                     << indices_sinfo->ndim);
  } else if (!indices_sinfo->IsUnknownDtype() &&
             !(indices_sinfo->dtype.is_int() || indices_sinfo->dtype.is_uint())) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Take op requires the input indices to have integer dtype. However, the "
                        "given indices dtype is "
                     << indices_sinfo->dtype);
  }

  const auto* attrs = call->attrs.as<TakeAttrs>();
  if (!attrs->axis.defined() && data_sinfo->ndim != 1) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Take op expects the input data to be 1-dimensional tensor when the axis "
                        "is not specified. However, the given data tensor has ndim "
                     << data_sinfo->ndim);
  }
  if (data_sinfo->IsUnknownNdim()) {
    return TensorStructInfo(data_sinfo->dtype, kUnknownNDim);
  }

  int axis = attrs->axis.defined()
                 ? NormalizeAxis(call, ctx, data_sinfo->ndim, attrs->axis.value()->value)
                 : 0;
  const auto* data_shape = data_sinfo->shape.as<ShapeExprNode>();
  const auto* indices_shape = indices_sinfo->shape.as<ShapeExprNode>();
  if (data_shape == nullptr || indices_shape == nullptr) {
    return TensorStructInfo(data_sinfo->dtype, data_sinfo->ndim);
  }

  Array<PrimExpr> output_shape = data_shape->values;
  output_shape.Set(axis, indices_shape->values[0]);
  return TensorStructInfo(ShapeExpr(output_shape), data_sinfo->dtype);
}

TVM_REGISTER_OP("relax.take")
    .set_attrs_type<TakeAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("indices", "Tensor", "The indices tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoTake);

/* relax.strided_slice */
TVM_REGISTER_NODE_TYPE(StridedSliceAttrs);

Expr strided_slice(Expr data,              //
                   Array<Integer> axes,    //
                   Array<PrimExpr> begin,  //
                   Array<PrimExpr> end,    //
                   Optional<Array<PrimExpr>> strides) {
  int n_axis = axes.size();
  CHECK_EQ(static_cast<int>(begin.size()), n_axis)
      << "StridedSlice requires the number of begin indices to equal the number of axes.";
  CHECK_EQ(static_cast<int>(end.size()), n_axis)
      << "StridedSlice requires the number of end indices to equal the number of axes.";
  if (strides.defined()) {
    CHECK_EQ(static_cast<int>(strides.value().size()), n_axis)
        << "StridedSlice requires the number of strides to equal the number of axes.";
  }

  ObjectPtr<StridedSliceAttrs> attrs = make_object<StridedSliceAttrs>();
  attrs->axes = std::move(axes);
  attrs->begin = std::move(begin);
  attrs->end = std::move(end);
  attrs->strides = std::move(strides);

  static const Op& op = Op::Get("relax.strided_slice");
  return Call(op, {std::move(data)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.strided_slice").set_body_typed(strided_slice);

StructInfo InferStructInfoStridedSlice(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo data_sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  const auto* attrs = call->attrs.as<StridedSliceAttrs>();
  if (attrs->axes.empty()) {
    return data_sinfo;
  }

  if (data_sinfo->IsUnknownNdim()) {
    return TensorStructInfo(data_sinfo->dtype, kUnknownNDim);
  }

  std::vector<int> axes = NormalizeAxes(call, ctx, data_sinfo->ndim, attrs->axes);
  const auto* data_shape = data_sinfo->shape.as<ShapeExprNode>();
  if (data_shape == nullptr) {
    return TensorStructInfo(data_sinfo->dtype, data_sinfo->ndim);
  }

  int n_axis = axes.size();
  Array<PrimExpr> strides = attrs->strides.defined()
                                ? attrs->strides.value()
                                : Array<PrimExpr>(n_axis, IntImm(DataType::Int(64), 1));
  std::vector<int> int_strides;
  int_strides.reserve(n_axis);
  // Only do output shape inference when all the begin/end/stride values are integers.
  for (int i = 0; i < n_axis; ++i) {
    const auto* int_begin = attrs->begin[i].as<IntImmNode>();
    const auto* int_end = attrs->end[i].as<IntImmNode>();
    const auto* int_stride = strides[i].as<IntImmNode>();
    if (!int_begin || !int_end || !int_stride) {
      return TensorStructInfo(data_sinfo->dtype, data_sinfo->ndim);
    }
    int_strides.push_back(int_stride->value);
  }

  Array<PrimExpr> output_shape = data_shape->values;
  for (int i = 0; i < n_axis; ++i) {
    PrimExpr len = int_strides[i] < 0 ? ceildiv(attrs->begin[i] - attrs->end[i], -int_strides[i])
                                      : ceildiv(attrs->end[i] - attrs->begin[i], int_strides[i]);
    output_shape.Set(axes[i], len);
  }
  return TensorStructInfo(ShapeExpr(output_shape), data_sinfo->dtype);
}

TVM_REGISTER_OP("relax.strided_slice")
    .set_attrs_type<StridedSliceAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoStridedSlice);

}  // namespace relax
}  // namespace tvm
