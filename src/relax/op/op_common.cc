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

#include "op_common.h"

#include <algorithm>

namespace tvm {
namespace relax {

Array<TensorStructInfo> GetInputTensorStructInfo(const Call& call, const BlockBuilder& ctx,
                                                 const Array<String>& input_names,
                                                 const String& op_name) {
  int n_input = input_names.size();
  if (static_cast<int>(call->args.size()) != n_input) {
    ctx->ReportFatal(Diagnostic::Error(call) << op_name << " op should have 2 arguments");
  }
  Array<TensorStructInfo> input_tensor_sinfo;
  input_tensor_sinfo.reserve(n_input);
  for (int i = 0; i < n_input; ++i) {
    const auto* sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[i]);
    if (sinfo == nullptr) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << op_name << " requires the input " << input_names[i]
                       << " to be Tensor. However, the given one is "
                       << call->args[i]->struct_info_->GetTypeKey());
    }
    input_tensor_sinfo.push_back(GetRef<TensorStructInfo>(sinfo));
  }
  return input_tensor_sinfo;
}

Optional<Array<PrimExpr>> InferBinaryBroadcastShape(const Call& call, const BlockBuilder& ctx,
                                                    const Array<PrimExpr>& lhs_shape,
                                                    const Array<PrimExpr>& rhs_shape,
                                                    const String& op_name) {
  arith::Analyzer* analyzer = ctx->GetAnalyzer();
  int lhs_ndim = lhs_shape.size();
  int rhs_ndim = rhs_shape.size();
  int max_ndim = std::max(lhs_ndim, rhs_ndim);

  std::vector<PrimExpr> output_shape;
  output_shape.reserve(max_ndim);

  int i = 1;
  for (; i <= std::min(lhs_ndim, rhs_ndim); ++i) {
    const PrimExpr& dim0 = lhs_shape[lhs_ndim - i];
    const PrimExpr& dim1 = rhs_shape[rhs_ndim - i];
    const auto* int_dim0 = dim0.as<IntImmNode>();
    const auto* int_dim1 = dim1.as<IntImmNode>();
    if (int_dim0 != nullptr && int_dim0->value == 1) {
      output_shape.push_back(dim1);
    } else if (int_dim1 != nullptr && int_dim1->value == 1) {
      output_shape.push_back(dim0);
    } else if (analyzer->CanProveEqual(dim0, dim1)) {
      output_shape.push_back(dim0);
    } else if (int_dim0 && int_dim1 && int_dim0->value != int_dim1->value) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "In " << op_name << ", the lhs shape at dim " << lhs_ndim - i << " is "
                       << dim0 << " and the rhs shape at dim " << rhs_ndim - i << " is " << dim1
                       << ", which are not broadcastable.");
    } else {
      // Use simple fallback when shape mismatch.
      return NullOpt;
    }
  }
  auto& longer_shape = (lhs_ndim > rhs_ndim) ? lhs_shape : rhs_shape;
  for (; i <= max_ndim; ++i) {
    output_shape.push_back(longer_shape[max_ndim - i]);
  }
  return Array<PrimExpr>(output_shape.rbegin(), output_shape.rend());
}

Array<Integer> CheckAxesInRangeNonRepetitive(const Call& call, const BlockBuilder& ctx, int ndim,
                                             const Array<Integer>& axes, const String op_name) {
  std::vector<bool> appeared_dims_set;
  Array<Integer> axes_non_neg;
  appeared_dims_set.resize(ndim);
  axes_non_neg.reserve(axes.size());
  for (const Integer& axis : axes) {
    int _axis = axis->value;
    if (_axis < -ndim || _axis >= ndim) {
      ctx->ReportFatal(Diagnostic::Error(call) << "In " << op_name << ", the input axis " << _axis
                                               << " is out of range. The input tensor has " << ndim
                                               << " dimensions, so axis should be in range ["
                                               << -ndim << ", " << ndim << ").");
    } else if (_axis < 0) {
      _axis = ndim + _axis;
    }

    if (appeared_dims_set[_axis]) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "In " << op_name
                       << ", the input axes is required to be non-repetitive. However, there are "
                          "multiple given axes referring to axis "
                       << _axis);
    }
    appeared_dims_set[_axis] = true;
    axes_non_neg.push_back(Integer(_axis));
  }
  return axes_non_neg;
}

}  // namespace relax
}  // namespace tvm
