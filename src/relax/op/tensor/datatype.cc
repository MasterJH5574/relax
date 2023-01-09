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
 * \file datatype.cc
 * \brief Datatype operators.
 */

#include "datatype.h"

#include <utility>

namespace tvm {
namespace relax {

/* relax.astype */
TVM_REGISTER_NODE_TYPE(AstypeAttrs);

Expr astype(Expr x, DataType dtype) {
  ObjectPtr<AstypeAttrs> attrs = make_object<AstypeAttrs>();
  attrs->dtype = dtype;

  static const Op& op = Op::Get("relax.astype");
  return Call(op, {std::move(x)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.astype").set_body_typed(astype);

StructInfo InferStructInfoAstype(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  const auto* attrs = call->attrs.as<AstypeAttrs>();
  ObjectPtr<TensorStructInfoNode> new_sinfo = make_object<TensorStructInfoNode>(*sinfo.get());
  new_sinfo->dtype = attrs->dtype;
  return TensorStructInfo(new_sinfo);
}

TVM_REGISTER_OP("relax.astype")
    .set_attrs_type<AstypeAttrs>()
    .set_num_inputs(1)
    .add_argument("x", "Tensor", "The input tensor")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoAstype)
    .set_attr<FMixedPrecision>("FMixedPrecision", InferMixedPrecisionFollow);

/* relax.wrap_param */
TVM_REGISTER_NODE_TYPE(WrapParamAttrs);

Expr MakeWrapParam(Expr data, DataType dtype) {
  ObjectPtr<WrapParamAttrs> attrs = make_object<WrapParamAttrs>();
  attrs->dtype = dtype;

  static const Op& op = Op::Get("relax.wrap_param");
  return Call(op, {std::move(data)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.wrap_param").set_body_typed(MakeWrapParam);

StructInfo InferStructInfoWrapParam(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  const auto* attrs = call->attrs.as<WrapParamAttrs>();
  ObjectPtr<TensorStructInfoNode> new_sinfo = make_object<TensorStructInfoNode>(*sinfo.get());
  new_sinfo->dtype = attrs->dtype;
  return TensorStructInfo(new_sinfo);
}

TVM_REGISTER_OP("relax.wrap_param")
    .set_attrs_type<WrapParamAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoWrapParam)
    .set_attr<FMixedPrecision>("FMixedPrecision", InferMixedPrecisionNever);

/* relax.cumsum */
TVM_REGISTER_NODE_TYPE(CumsumAttrs);

RELAX_REGISTER_OP("relax.cumsum")
    .set_attrs_type<CumsumAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferShape>("FInferShape", InferShapeCumsum)
    .set_attr<FInferType>("FInferType", InferTypeCumsum);

Expr MakeCumsum(Expr data, Optional<Integer> axis) {
  ObjectPtr<CumsumAttrs> attrs = make_object<CumsumAttrs>();
  attrs->axis = axis;

  static const Op& op = Op::Get("relax.cumsum");
  return Call(op, {std::move(data)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.cumsum").set_body_typed(MakeCumsum);

Optional<Expr> InferShapeCumsum(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Cumsum op should have 1 argument");
  }

  const auto* shape = call->args[0]->shape().as<ShapeExprNode>();
  const auto* attrs = call->attrs.as<CumsumAttrs>();
  if (shape == nullptr) {
    return RuntimeDepShape();
  }

  if (attrs->axis.defined()) {
    return GetRef<ShapeExpr>(shape);
  }

  PrimExpr prod = tir::make_const(DataType::Int(64), 1);
  for (const PrimExpr& shape_dim : shape->values) {
    prod = prod * shape_dim;
  }
  return ShapeExpr({prod});
}

Type InferTypeCumsum(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Cumsum op should have 1 argument");
  }

  const auto* input_type = call->args[0]->checked_type().as<DynTensorTypeNode>();
  if (input_type == nullptr) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The op input should has type DynTensorType, but actually it is "
                       << call->args[0]->checked_type()->GetTypeKey()
                       << ". Please make sure the input has type DynTensorType.");
  }

  const auto* attrs = call->attrs.as<CumsumAttrs>();
  if (attrs->axis.defined()) {
    return GetRef<DynTensorType>(input_type);
  } else {
    return DynTensorType(/*ndim=*/1, input_type->dtype);
  }
}

/* relax.collapse_sum_like */
RELAX_REGISTER_OP("relax.collapse_sum_like")
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("collapse_target", "Tensor",
                  "The tensor whose shape is the shape to collapse to.")
    .set_attr<FInferShape>("FInferShape", InferShapeCollapseSumLike)
    .set_attr<FInferType>("FInferType", InferTypeCollapseSumLike);

Expr MakeCollapseSumLike(Expr data, Expr collapse_target) {
  static const Op& op = Op::Get("relax.collapse_sum_like");
  return Call(op, {std::move(data), std::move(collapse_target)}, Attrs(), {});
}

TVM_REGISTER_GLOBAL("relax.op.collapse_sum_like").set_body_typed(MakeCollapseSumLike);

Expr InferShapeCollapseSumLike(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "collapse_sum_like op should have 2 arguments");
  }

  Expr shape = call->args[1]->shape();
  auto* s = shape.as<ShapeExprNode>();
  if (s) {
    return ShapeExpr(s->values);
  } else {
    return RuntimeDepShape();
  }
}

Type InferTypeCollapseSumLike(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "collapse_sum_like op should have 2 arguments");
  }

  auto* input_ty = call->args[1]->checked_type().as<DynTensorTypeNode>();
  if (!input_ty) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The input tensor should be DynTensor, but got "
                       << call->args[0]->checked_type()->GetTypeKey());
  }

  return GetRef<DynTensorType>(input_ty);
}

/* relax.collapse_sum_to */
RELAX_REGISTER_OP("relax.collapse_sum_to")
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("shape", "ShapeExpr", "The shape to collapse to..")
    .set_attr<FInferShape>("FInferShape", InferShapeCollapseSumTo)
    .set_attr<FInferType>("FInferType", InferTypeCollapseSumTo);

Expr MakeCollapseSumTo(Expr data, Expr shape) {
  static const Op& op = Op::Get("relax.collapse_sum_to");
  return Call(op, {std::move(data), std::move(shape)}, Attrs(), {});
}

TVM_REGISTER_GLOBAL("relax.op.collapse_sum_to").set_body_typed(MakeCollapseSumTo);

Expr InferShapeCollapseSumTo(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "collapse_sum_to op should have 2 arguments");
  }

  return call->args[1];
}

Type InferTypeCollapseSumTo(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "collapse_sum_to op should have 2 arguments");
  }

  const auto* orig_type = call->args[0]->checked_type().as<DynTensorTypeNode>();
  const auto* shape_type = call->args[1]->checked_type().as<ShapeTypeNode>();
  if (shape_type == nullptr) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The input shape should has type ShapeType, but actually it is "
                       << call->args[1]->checked_type()->GetTypeKey()
                       << ". Please make sure the input data has type ShapeType.");
  }

  int ndim = -1;
  const auto* shape = call->args[1].as<ShapeExprNode>();
  if (shape != nullptr) {
    ndim = shape->values.size();
  }

  return DynTensorType(ndim, orig_type->dtype);
}

}  // namespace relax
}  // namespace tvm
