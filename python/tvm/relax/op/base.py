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
# pylint: disable=redefined-builtin
"""The base Relax operators."""
from typing import Union, List, Optional

import tvm
from tvm.runtime.object import Object

from . import _ffi_api
from ..expr import Expr, ShapeExpr, Tuple, Call, ExternFunc, _update_struct_info
from ..struct_info import ShapeStructInfo, TupleStructInfo, TensorStructInfo
from ..ty import DynTensorType, TupleType
from ...ir import Array, Type

py_print = print  # pylint: disable=invalid-name


def call_tir(
    func: Union[str, Expr],
    args: Union[Expr, Tuple, List[Expr]],
    output_tensor_sinfo: Union[TensorStructInfo, List[TensorStructInfo]],
    tir_vars: Optional[ShapeExpr] = None,
) -> Call:
    """
    Call a destination-passing-style function and return the output.

    Parameters
    ----------
    func : Union[str, Expr]
        The destination-passing-style function, can be ExternFunc or PrimFunc.

    args : Union[Expr, Tuple, List[Expr]]
        The input arguments.

    output_tensor_sinfo : Union[StructInfo, List[StructInfo]]
        The structure info of the output. Can be either a TensorStructInfo
        or a TupleStructInfo whose fields are all TensorStructInfo.
        Moreover, all the TensorStructInfo inside must have symbolic shape defined.

    tir_vars : ShapeExpr, optional
        ShapeExpr representing a tuple of integers to unpack when calling func. Is null if not used

    Returns
    -------
    ret: Call
        A call node for the call_tir operator.
    """
    if isinstance(func, str):
        func = ExternFunc(func)

    if isinstance(args, Expr):
        args = Tuple((args,))
    elif isinstance(args, (list, tuple)):
        args = Tuple(args)

    def _convert_shape_type(tensor_sinfo: TensorStructInfo):
        if not isinstance(tensor_sinfo, TensorStructInfo):
            raise TypeError(
                "The output struct info is only allowed to be TensorStructInfo. However, one "
                f"given struct info is {tensor_sinfo}"
            )
        if tensor_sinfo.dtype == "":
            raise ValueError("The output tensor should have known dtype")

        shape = tensor_sinfo.shape
        if shape is None:
            raise ValueError("The output tensor should have defined symbolic shape")
        shape_sinfo = shape.struct_info
        assert isinstance(shape_sinfo, ShapeStructInfo)
        if shape_sinfo.values is None:
            raise ValueError("The output tensor should have defined symbolic shape")

        ttype = DynTensorType(len(shape_sinfo.values), tensor_sinfo.dtype)
        return shape, ttype

    if isinstance(output_tensor_sinfo, (list, tuple, Array)):
        shapes, ttypes = zip(
            *[_convert_shape_type(tensor_sinfo) for tensor_sinfo in output_tensor_sinfo]
        )
        shapes = Tuple(shapes)
        _update_struct_info(shapes, TupleStructInfo([shape.struct_info for shape in shapes]))
        ttypes = TupleType(ttypes)
        return _ffi_api.call_tir(func, args, shapes, ttypes, tir_vars)  # type: ignore
    else:
        shape, ttype = _convert_shape_type(output_tensor_sinfo)
        return _ffi_api.call_tir(func, args, shape, ttype, tir_vars)  # type: ignore


def make_closure(
    func: Expr,
    args: Union[Tuple, List[Expr]],
) -> Object:
    """
    Create a closure with free variables and return the closure.

    Parameters
    ----------
    func : Expr
        The closure, can be ExternFunc or PrimFunc.

    args : Union[Tuple, List[Expr]]
        The input arguments.


    Returns
    -------
    ret: Object
        The VMClosure.
    """

    if isinstance(args, (list, tuple)):
        args = Tuple(args)

    return _ffi_api.make_closure(func, args)  # type: ignore


def invoke_closure(
    closure: Expr,
    args: Union[Tuple, List[Expr]],
    type_args: Union[List[Type], Type],
) -> Object:
    """
    Invoke a closure.

    Parameters
    ----------
    closure : Expr
        The VMClosure object.

    args : Union[Tuple, List[Expr]]
        The input arguments.

    type_args: Union[Tuple[Type], Type]
        The type_args of the CallNode

    Returns
    -------
    ret: Object
        The result.
    """

    if isinstance(args, (list, tuple)):
        args = Tuple(args)
    if not isinstance(type_args, (list, tuple)):
        type_args = (type_args,)

    return _ffi_api.invoke_closure(closure, args, type_args)  # type: ignore


def render_object(val: tvm.Object) -> str:
    """
    Given a TVM Object, renders it in string form. Used for Relax printing and assertions.

    Parameters
    ----------
    val: tvm.Object
        An object to render

    Returns
    -------
    ret: str
        A string representing the value, ideally human-readable
    """
    if isinstance(val, tvm.runtime.ndarray.NDArray):
        return str(val)
    # no pretty-printer by default, so if we don't handle this,
    # then we can't look inside tuples
    if isinstance(val, tvm.runtime.container.ADT):
        # the fields array of an ADT cannot be directly accessed in Python
        # so we have to get the length and index into the fields separately
        fields = ", ".join([render_object(val[i]) for i in range(len(val))])
        # special case: tag = 0 is a tuple
        if val.tag == 0:
            return f"({fields})"
        return f"ADT(tag={val.tag}, fields=[{fields}])"
    return str(val)


@tvm.register_func("relax.run.print")
def relax_print(format_str: str, *format_args: tvm.Object) -> None:
    """
    Takes a list of values to print, formats with the given format string.
    If the format string is empty, simply prints.

    Call from TVM script like this:
    `relax.print(value1, value2, ..., valueN, format=format_str)`
    or
    `relax.print(value1, value2, ..., valueN) # format_str defaults to ""`

    Parameters
    ----------
    format_str: str
        The last argument is a Python-style format string for printing the value

    format_args: List[Object]
        The values to print.
    """
    val_strs = map(render_object, format_args)
    if format_str == "":
        py_print(*val_strs)
    else:
        py_print(format_str.format(*val_strs))


def print(*values: List[Expr], format: str = "") -> Expr:
    """Print op to print the values

    Parameters
    ----------
    values : List[Expr]
        The values to print.

    format_str: str
        The format string.

    Returns
    -------
    result : Expr
        A relax Call, which will print the value during runtime.
    """
    return _ffi_api.print(values, format)  # type: ignore # pylint: disable=no-member


@tvm.register_func("relax.run.assert_op")
def relax_assert_op(condition: tvm.Object, format_str: str, *format_args: tvm.Object) -> None:
    """
    A variadic function. The first value serves as the assertion condition:
    If the condition is true, then the operator does nothing.
    If the condition is false, then the operator raises an assertion error.

    Arguments after the first value serve as format arguments for the error message;
    the last argument must be a format string for the error message (empty by default).
    If the format string is the empty string, then the error message will simply include
    a comma-separated list of the format arguments.
    The condition argument is not included in the format string.

    Parameters
    ----------
    condition: tvm.Object
        The assertion condition. Must be a boolean scalar.

    format_str: str
        The last argument is a Python-style format string for printing the value

    format_args: List[tvm.Object]
        Values used for formatting the string.
    """
    if not isinstance(format_str, str):
        raise ValueError(
            f"The format string argument to assert must be a string, given {type(format_str)})"
        )

    # should be guaranteed by the type system
    if not isinstance(condition, tvm.runtime.ndarray.NDArray):
        raise ValueError(f"The condition must be an NDArray, but given a {type(condition)}.")

    # may happen if the original program had unknown shape or dtype for the tensor's type
    dtype = condition.dtype
    if dtype != "bool":
        raise ValueError(f"The condition must be a bool scalar, but given a {dtype} tensor")
    shape = condition.shape
    if len(shape) != 0:
        raise ValueError(f"The condition must be a scalar, but it has a shape of {shape}")

    val = condition.numpy()
    if not val:
        error_message = "Assertion Failed"
        if format_args or format_str != "":
            rendered = map(render_object, format_args)
            if format_str != "":
                error_message = format_str.format(*rendered)
            else:
                error_message = ", ".join(rendered)
        raise AssertionError(error_message)


def assert_op(
    condition: Expr, format_args: Optional[Union[Expr, List[Expr]]] = None, format: str = ""
) -> Expr:
    """
    Create a call to Relax's assert_op operation (`assert` is reserved in Python,
    so the name must be distinct).

    Parameters
    ----------
    condition: Expr
        The assertion condition.

    format_args: Optional[Union[Expr, List[Expr]]]
        Format arguments for the error message if the condition fails.

    format_str: str
        The format string for the error message.

    Returns
    -------
    result : Expr
        A Call to the Relax assert operation.
    """
    if format_args is None:
        format_args = []
    if isinstance(format_args, Expr):  # type: ignore
        format_args = [format_args]
    return _ffi_api.assert_op(condition, format_args, format)  # type: ignore


def shape_of(expr: Expr) -> Expr:
    """Get shape of a tensor.

    Parameters
    ----------
    expr : Expr
        The input Expr.

    Returns
    -------
    result : Expr
        A relax Call, which gets the shape of the input
    """
    return _ffi_api.shape_of(expr)  # type: ignore # pylint: disable=no-member
