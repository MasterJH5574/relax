# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=no-else-return
# pylint: disable=unidiomatic-typecheck
"""
This file contains the set of passes for Relax, which exposes an interface for
configuring the passes and scripting them in Python.
"""

from typing import Dict, List, Optional
from enum import IntEnum

import tvm
from tvm import tir
from tvm.relax.ty import Type
from tvm.relax.struct_info import StructInfo, FuncStructInfo
from tvm.relax.expr import DataflowBlock, GlobalVar, Var, Expr, Function, Binding, Call
from . import _ffi_api


def get_static_type(sinfo: StructInfo) -> Type:
    """Get the corresponding static type from a StructInfo.

    Parameters
    ----------
    sinfo : StructInfo
        The input struct info.

    Returns
    -------
    ret : Type
        The corresponding static type.
    """
    return _ffi_api.GetStaticType(sinfo)  # type: ignore


def get_legacy_shape_hint(sinfo: StructInfo) -> Optional[Expr]:
    """Get the corresponding shape from a StructInfo.

    Parameters
    ----------
    sinfo : StructInfo
        The input struct info.

    Returns
    -------
    ret : Type
        The corresponding shape.
    """
    return _ffi_api.GetLegacyShapeHint(sinfo)  # type: ignore


def erase_to_well_defined(
    sinfo: StructInfo,
    shape_var_map: Dict[tir.Var, tir.PrimExpr] = None,
    var_map: Dict[Var, Expr] = None,
) -> StructInfo:
    """Erase sinfo into a well defined form.

    This function removes the StructInfo's dependencies on shape and vars that
    are not defined in given maps.

    Parameters
    ----------
    sinfo : StructInfo
        The input struct info.

    shape_var_map : Dict[tir.Var, tir.PrimExpr]
        Specifies the defined shape vars and the values they should map to.

    var_map : Dict[Var, Expr]
        Specifies the defined vars and the values they should map to.

    Returns
    -------
    ret : StructInfo
        The corresponding erased struct info.
    """
    shape_var_map = {} if shape_var_map is None else shape_var_map
    var_map = {} if var_map is None else var_map

    return _ffi_api.EraseToWellDefined(sinfo, shape_var_map, var_map)  # type: ignore


class BaseCheckResult(IntEnum):
    """Return result of fine-grained base check.

    Note
    ----
    Base check comes with fine-grained fail levels.

    - FAIL_L0: The lhs and rhs have no intersection at all.
    - FAIL_L1: We get the failure by looking at static information.
    - FAIL_L2: We get the failure due to unknown symbolic variable relations.
    """

    FAIL_L0 = 0
    FAIL_L1 = 1
    FAIL_L2 = 2
    PASS = 3


def struct_info_base_check(base: StructInfo, derived: StructInfo) -> BaseCheckResult:
    """Run a base check to see if base subsumes derived.

    Parameters
    ----------
    base: StructInfo
        The base struct info.

    derived: StructInfo
        The derived struct info.

    Returns
    -------
    ret : StructInfo
        The derived return value struct info.
    """
    return _ffi_api.StructInfoBaseCheck(base, derived)  # type: ignore


def derive_call_ret_struct_info(
    func_sinfo: FuncStructInfo, call: Call, ctx: "tvm.relax.BlockBuilder"
) -> StructInfo:
    """Derive the call's ret value struct info from inputs.

    Parameters
    ----------
    func_sinfo: FuncStructInfo
        The call's function signature.

    call: Call
        The call expression

    ctx: tvm.relax.BlockBuilder
        The context block builder.

    Returns
    -------
    ret : StructInfo
        The derived return value struct info.

    Note
    ----
    This is an internal derivation function, call.op field is
    ignored in this case and the derivation only depends on func_sinfo.
    """
    return _ffi_api.DeriveCallRetStructInfo(func_sinfo, call, ctx)  # type: ignore


def struct_info_lca(lhs: StructInfo, rhs: StructInfo) -> StructInfo:
    """Unify the two struct info their least common ancestor.

    Parameters
    ----------
    lhs: StructInfo
        The left operand.

    rhs: StructInfo
        The right operand.

    Returns
    -------
    ret : StructInfo
        The corresponding lca result.
    """
    return _ffi_api.StructInfoLCA(lhs, rhs)  # type: ignore


def post_order_visit(expr, fvisit):
    """Recursively visit the ir in post DFS order node,
    apply fvisit. Each node is guaranteed to be visited
    only once.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression.

    fvisit : function
        The visitor function to be applied.
    """
    return _ffi_api.post_order_visit(expr, fvisit)  # type: ignore


def well_formed(mod: tvm.IRModule, check_struct_info: bool = True) -> bool:
    """Check if the IRModule is well formed.

    Parameters
    ----------
    mod : tvm.IRModule
        The input IRModule.

    check_struct_info : bool
        A boolean flag indicating if the property "every Expr must
        have defined structure info" will be checked.

    Returns
    -------
    ret: bool
        True if the IRModule is well formed, False if not.

    Note
    ----
    By default the structure info is always checked. It is only in test cases
    where `check_struct_info` might be false, so that other well-formed requirements
    will be well tested and will not be blocked by not having structure info.
    """
    return _ffi_api.well_formed(mod, check_struct_info)  # type: ignore


def get_var2val(func: Function) -> Dict[Var, Expr]:
    """
    Get a mapping from Var to Expr for each variable in the function.

    Parameters
    ----------
    func : Function
        The input function to be analyzed.

    Returns
    -------
    Dict[Var, Expr]
        A mapping from Var to Expr.
    """
    return _ffi_api.get_var2val(func)  # type: ignore


def udchain(dfb: DataflowBlock) -> Dict[Var, List[Var]]:
    """
    Analyze the variable use-def chain in a dataflow block.

    Parameters
    ----------
    dfb : DataflowBlock
        The dataflow block to analyze

    Returns
    -------
    Dict[Var, List[Var]]
        A mapping from variable definition to its uses.
    """
    return _ffi_api.udchain(dfb)  # type: ignore


def name_to_binding(func: Function) -> Dict[str, List[Binding]]:
    """Return a map from variable name to its bindings."""
    return _ffi_api.name_to_binding(func)  # type: ignore


def remove_all_unused(func: Function) -> Function:
    """Remove all unused variables from the function.

    Parameters
    ----------
    func : Function
        The input function to be analyzed.

    Returns
    -------
    Function
        The function with unused variables removed.
    """
    return _ffi_api.remove_all_unused(func)  # type: ignore


def shape_vars(expr: Expr) -> List[tir.Var]:
    """
    Returns all shape variables (TIR variables) in the given expression.

    Note that the expression is intended to be a shape expression, i.e.,
    one used as the `shape_` for another expression.

    Parameters
    ----------
    expr : Expr
        The expression. Meant to be a shape expression.

    Returns
    -------
    ret: List[tir.Var]
        A list of all shape variables (TIR variables) in the expression.
    """
    return _ffi_api.shape_vars(expr)  # type: ignore


def derive_func_ret_shape(args: List[Var], body: Expr) -> Expr:
    """
    Given the argument vars and body, derives a return shape for
    a function with those args and that body.
    If the body's shape contains free shape vars (those not used in the args), the
    return shape is relaxed to RuntimeDepShape; otherwise, the body's shape is used.

    Parameters
    ----------
    args: List[Var]
        The argument variables, ideally with the shape_ field filled in

    body: Expr
        The functino body, ideally with the shape_ field filled in

    Returns
    -------
    ret: Expr
        An expression that can serve as the return shape for the function
    """
    return _ffi_api.derive_func_ret_shape(args, body)  # type: ignore


def bound_vars(expr: Expr) -> List[Var]:
    """
    Return all bound variables from expression expr.

    Bound variables are all variables that are declared in the expr.
    They only have meaning inside that expr, and can only be used in it.

    Parameters
    ----------
    expr: Expr
        The expression.

    Returns
    -------
    ret: List[Var]
        List of bound vars in expr, in post-DFS order
    """
    return _ffi_api.bound_vars(expr)  # type: ignore


def free_vars(expr: Expr) -> List[Var]:
    """
    Return all free variables from expression expr.

    Free variables are variables that are not bound by a
    VarBinding or a function parameter in the expression.

    Parameters
    ----------
    expr: Expr
        The expression.

    Returns
    -------
    ret: List[Var]
        List of free vars in expr, in post-DFS order
    """
    return _ffi_api.free_vars(expr)  # type: ignore


def all_vars(expr: Expr) -> List[Var]:
    """
    Return all (local) variables from expression expr.

    Parameters
    ----------
    expr: Expr
        The expression.

    Returns
    -------
    ret: List[Var]
        List of vars in expr, in post-DFS order
    """
    return _ffi_api.all_vars(expr)  # type: ignore


def all_global_vars(expr: Expr) -> List[GlobalVar]:
    """
    Return all global variables from expression expr.

    Parameters
    ----------
    expr: Expr
        The expression.

    Returns
    -------
    ret: List[GlobalVar]
        List of global vars in expr, in post-DFS order
    """
    return _ffi_api.all_global_vars(expr)  # type: ignore


def called_global_vars(expr: Expr) -> List[GlobalVar]:
    """
    Return all global vars called (potentially recursively) from expr.

    Parameters
    ----------
    expr: Expr
        The expression

    Returns
    -------
    ret: List[GlobalVar]
        List of global vars that are used recursively in expr,
        in post-DFS order
    """
    return _ffi_api.called_global_vars(expr)  # type: ignore
