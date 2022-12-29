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
 * \file op_common.h
 * \brief A set of utilities and common functionality
 * for Relax ops.
 */
#ifndef TVM_RELAX_OP_OP_COMMON_H_
#define TVM_RELAX_OP_OP_COMMON_H_

#include <tvm/arith/analyzer.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/tir/data_layout.h>

#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace relax {

/************ Op input struct info getter ************/

/*!
 * \brief Get the tensor struct info of the operator input.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \return The tensor struct info of each input.
 * \note This function require every input to be Tensor. The number of call arguments is required
 * to match the number of inputs of the op being called.
 */
Array<TensorStructInfo> GetInputTensorStructInfo(const Call& call, const BlockBuilder& ctx);

/*!
 * \brief Get the tensor struct info of the unary operator input.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \return The tensor struct info of the unary operator input.
 * \throw Throw exception if the number of input is not one, or the struct info of the input is not
 * a tensor struct info.
 */
inline TensorStructInfo GetUnaryInputTensorStructInfo(const Call& call, const BlockBuilder& ctx) {
  return GetInputTensorStructInfo(call, ctx)[0];
}

/************ Op registration macro ************/

/*!
 * \brief Quick helper macro
 * - Expose a make function to construct the node.
 * - Register op to the registry.
 * \param OpName The name of operator to register. The name passed in will
 *  1. be prepended with a prefix "relax.op." as the FFI key string for the make function,
 *  2. be prepended with a prefix "relax." as the key string in the operator registry.
 */
#define RELAX_REGISTER_UNARY_OP(OpName)                               \
  TVM_REGISTER_GLOBAL("relax.op." OpName).set_body_typed([](Expr e) { \
    static const Op& op = Op::Get("relax." OpName);                   \
    return Call(op, {e}, Attrs(), {});                                \
  });                                                                 \
  TVM_REGISTER_OP("relax." OpName)                                    \
      .set_num_inputs(1)                                              \
      .add_argument("e", "Tensor", "The input tensor.")               \
      .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoUnary)

inline StructInfo InferStructInfoUnary(const Call& call, const BlockBuilder& ctx) {
  return GetUnaryInputTensorStructInfo(call, ctx);
}

/************ Utilities ************/

/*!
 * \brief Infer the output datatype for binary arithmetic operators.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \param lhs_sinfo The struct info of the left operand
 * \param rhs_sinfo The struct info of the right operand
 * \return The inferred output dtype.
 * \throw Throw exception if the dtype of lhs and rhs don’t match
 */
inline DataType InferBinaryArithOpOutDtype(const Call& call, const BlockBuilder& ctx,
                                           const TensorStructInfo& lhs_sinfo,
                                           const TensorStructInfo& rhs_sinfo) {
  if (lhs_sinfo->IsUnknownDtype() || rhs_sinfo->IsUnknownDtype()) {
    return DataType::Void();
  } else if (lhs_sinfo->dtype != rhs_sinfo->dtype) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Data types " << lhs_sinfo->dtype << " and " << rhs_sinfo->dtype
                     << " must be equal for binary operators");
  }
  return lhs_sinfo->dtype;
}

/*!
 * \brief Infer the output shape for binary broadcast operators.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \param lhs_shape The shape of the lhs operand.
 * \param rhs_shape The shape of the rhs operand.
 * \return The inferred output shape after broadcasting. Or `NullOpt` if the output shape cannot be
 * determined due to symbolic broadcast.
 */
Optional<Array<PrimExpr>> InferBinaryBroadcastShape(const Call& call, const BlockBuilder& ctx,
                                                    const Array<PrimExpr>& lhs_shape,
                                                    const Array<PrimExpr>& rhs_shape);

/*!
 * \brief Complete the padding to a 4-length array.
 * - If the padding length is 1, the same padding is used on all top/left/bottom/right sides
 * - If the padding length is 2, top/bottom sides use padding[0] and left/right use padding[1]
 * - If the padding length is 4, padding is in the order of (top, left, bottom, right)
 * \param padding The given padding to be completed
 * \return The completed padding.
 * \throws Throws error if the input padding length is neither 1, 2 or 4.
 */
inline Array<PrimExpr> GetCompletePadding2D(Array<PrimExpr> padding) {
  if (padding.size() == 1) {
    return {padding[0], padding[0], padding[0], padding[0]};
  } else if (padding.size() == 2) {
    return {padding[0], padding[1], padding[0], padding[1]};
  } else if (padding.size() == 4) {
    return padding;
  }
  LOG(FATAL) << "The input padding length is expected to be either 1, 2 or 4. However, the given "
                "padding is "
             << padding;
  throw;
}

/*!
 * \brief Check if the given tensor layout can be converted to the given target layout.
 * If convertible, return the tensor layout and the bijective conversion in tir::Layout and
 * tir::BijectiveLayout accordingly.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \param tensor_layout The tensor layout to be checked
 * \param tgt_layout The target layout to be matched
 * \param tensor_name The name of the input tensor
 * \return The tensor layout and the bijective conversion in tir::Layout and tir::BijectiveLayout
 * accordingly.
 */
inline std::pair<tir::Layout, tir::BijectiveLayout> CheckTensorLayout(const Call& call,
                                                                      const BlockBuilder& ctx,
                                                                      const String& tensor_layout,
                                                                      const String& tgt_layout,
                                                                      const String& tensor_name) {
  tir::Layout _tensor_layout(tensor_layout, DataType::Int(64));
  tir::BijectiveLayout tensor2tgt(_tensor_layout, tir::Layout(tgt_layout, DataType::Int(64)));
  if (!tensor2tgt.defined()) {
    ctx->ReportFatal(Diagnostic::Error(call) << call->op << " requires the given " << tensor_name
                                             << " layout to be convertible from " << tgt_layout
                                             << " layout. However, the given layout "
                                             << tensor_layout << " is not convertible.");
  }
  return {_tensor_layout, tensor2tgt};
}

/*!
 * \brief Check if the given tensor struct info has expected ndim per the given layout (or the ndim
 * is unknown), and try to cast the shape to ShapeExpr.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \param sinfo The input tensor struct info to be checked.
 * \param layout The layout that the given tensor is expected to have.
 * \return The shape of the input tensor in ShapeExpr, or `NullOpt` if the shape is unknown.
 */
inline Optional<ShapeExpr> CheckNdimPerLayoutAndGetShape(const Call& call, const BlockBuilder& ctx,
                                                         const TensorStructInfo& sinfo,
                                                         const tir::Layout& layout) {
  if (!sinfo->IsUnknownNdim() && sinfo->ndim != static_cast<int>(layout.ndim())) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "In " << call->op << ", layout " << layout << " requires the input to be "
                     << layout.ndim() << "-dim tensor. However, the given input has ndim "
                     << sinfo->ndim);
  }
  return Downcast<Optional<ShapeExpr>>(sinfo->shape);
}

/*!
 * \brief Check if the given array of axes are all in range and non-repetitive with regards to the
 * given ndim. And convert all axes to non-negative index.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \param ndim The ndim constraint, which is required to be known already.
 * \param axes The axis indices to be checked
 * \return The input axes in non-negative indexing.
 */
std::vector<int> CheckAxesInRangeNonRepetitive(const Call& call, const BlockBuilder& ctx, int ndim,
                                               const Array<Integer>& axes);

/*!
 * \brief Check if the given axis is in range with regards to the given ndim. And convert it to
 * non-negative index.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \param ndim The ndim constraint.
 * \param axis The axis index to be checked
 * \return The input axis in non-negative indexing.
 */
inline int CheckAxisInRange(const Call& call, const BlockBuilder& ctx, int ndim, int axis) {
  return CheckAxesInRangeNonRepetitive(call, ctx, ndim, {axis})[0];
}

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_OP_COMMON_H_
