/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file binary.cc
 * \brief binary broadcast operators.
 */

#include "binary.h"

#include <algorithm>

namespace tvm {
namespace relax {

template <typename FType>
StructInfo InferStructInfoBroadcast(const Call& call, const BlockBuilder& ctx,
                                    FType f_compute_out_dtype) {
  Array<TensorStructInfo> input_sinfo = GetInputTensorStructInfo(call, ctx);
  TensorStructInfo lhs_sinfo = input_sinfo[0];
  TensorStructInfo rhs_sinfo = input_sinfo[1];

  // DateType
  DataType output_dtype = f_compute_out_dtype(call, ctx, lhs_sinfo, rhs_sinfo);

  // ndims
  int output_ndim;
  if (lhs_sinfo->IsUnknownNdim() || rhs_sinfo->IsUnknownNdim()) {
    output_ndim = kUnknownNDim;
  } else {
    output_ndim = std::max(lhs_sinfo->ndim, rhs_sinfo->ndim);
  }

  auto* lhs_shape = lhs_sinfo->shape.as<ShapeExprNode>();
  auto* rhs_shape = rhs_sinfo->shape.as<ShapeExprNode>();
  // Shapes and ndims
  if (lhs_shape && rhs_shape) {
    // If all inputs have shapes, directly infer shapes
    Optional<Array<PrimExpr>> output_shape =
        InferBinaryBroadcastShape(call, ctx, lhs_shape->values, rhs_shape->values);
    if (!output_shape.defined()) {
      return TensorStructInfo(output_dtype, /*ndim=*/output_ndim);
    } else {
      ICHECK_EQ(static_cast<int>(output_shape.value().size()), output_ndim);
      return TensorStructInfo(ShapeExpr(output_shape.value()), output_dtype);
    }
  } else if (lhs_sinfo->shape.defined() && lhs_sinfo->shape.same_as(rhs_sinfo->shape)) {
    return TensorStructInfo(lhs_sinfo->shape.value(), output_dtype);
  } else {
    return TensorStructInfo(output_dtype, /*ndim=*/output_ndim);
  }
}

StructInfo InferStructInfoBroadcastArith(const Call& call, const BlockBuilder& ctx) {
  return InferStructInfoBroadcast(call, ctx, InferBinaryArithOpOutDtype);
}

StructInfo InferStructInfoBroadcastCMP(const Call& call, const BlockBuilder& ctx) {
  return InferStructInfoBroadcast(
      call, ctx,
      [](const Call& call, const BlockBuilder& ctx, const TensorStructInfo& lhs_sinfo,
         const TensorStructInfo& rhs_sinfo) { return DataType::Bool(); });
}

RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(add).describe("Elementwise addition with broadcasting");

RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(subtract).describe(
    "Elementwise subtraction with broadcasting");

RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(multiply).describe(
    "Elementwise multiplication with broadcasting");

RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(divide).describe(
    "Elementwise division with broadcasting");

RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(floor_divide)
    .describe("Elementwise floor-division with broadcasting");

RELAX_REGISTER_CMP_OP_AND_IMPL(less);

}  // namespace relax
}  // namespace tvm
