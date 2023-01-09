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
"""Datatype operators."""
from typing import Union

from tvm import DataType

from . import _ffi_api
from ..expr import Constant, Expr


def astype(x: Expr, dtype: Union[str, DataType]) -> Expr:
    """Cast input tensor to the given data type.

    Parameters
    ----------
    x : relax.Expr
        The input data to the operator.

    dtype: Union[str, DataType]
        The target data type

    Returns
    -------
    result : relax.Expr
        The casted result.
    """
    return _ffi_api.astype(x, dtype)  # type: ignore


def wrap_param(data: Expr, dtype: Union[str, DataType] = "float32") -> Expr:
    """Cast input tensor which is model param to data type if the dtype of the input data is not
    the same as the given dtype.

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    dtype : Union[str, DataType]
        The target data type

    Returns
    -------
    result : relax.Expr
        The casted result.
    """
    assert isinstance(data, Constant)
    return _ffi_api.wrap_param(data, dtype)  # type: ignore


def cumsum(data: Expr, axis: Optional[int] = None) -> Expr:
    """Numpy style cumsum op. Return the cumulative inclusive sum of the elements along
    a given axis.

    Parameters
    ----------
    data : relay.Expr
        The input data to the operator.

    axis : Optional[int]
        Axis along which the cumulative sum is computed. The default (None) is to compute
        the cumsum over the flattened array.

    Returns
    -------
    result : relax.Expr
        The result has the same size as data, and the same shape as data if axis is not None.
        If axis is None, the result is a 1-d array.

    Examples
    --------
    .. code-block:: python

        a = [[1,2,3], [4,5,6]]

        cumsum(a)  # if axis is not provided, cumsum is done over the flattened input.
        -> [ 1,  3,  6, 10, 15, 21]

        cumsum(a, axis=0)  # sum over rows for each of the 3 columns
        -> [[1, 2, 3],
            [5, 7, 9]]

        cumsum(a, axis=1)
        -> [[ 1,  3,  6],
            [ 4,  9, 15]]
    """
    return _ffi_api.cumsum(data, axis)  # type: ignore


def collapse_sum_like(data: Expr, collapse_target: Expr) -> Expr:
    """Return a summation of data to the shape of collapse_target.

    For details, please see relax.op.collapse_sum_to.

    Parameters
    ----------
    data : relax.Expr
        The input tensor.

    collapse_target : relax.Expr
        The tensor whose shape is the shape to collapse to.

    Returns
    -------
    result : relax.Expr
        The result tensor after summation.
    """
    return _ffi_api.collapse_sum_like(data, collapse_target)  # type: ignore


def collapse_sum_to(
    data: Expr, shape: Union[PrimExprLike, List[PrimExprLike], Tuple[PrimExprLike], Expr]
) -> Expr:
    """Return a summation of data to the given shape.

    collapse_sum_to is intended as the backward operator of tvm.relax.op.broadcast_to and
    other broadcast operators in the automatic differentiation process.

    We expect that data is the result of broadcasting some tensor of the given shape in some
    broadcast operation. Thus the given shape and data.shape must follow broadcast rules.

    During computation, the axes of data.shape and shape are checked from right to left. For every
    axis, if it either:
    - exist in data but not in collapse_target, or
    - is larger than 1 in data and equals to 1 in collapse_target,

    data will be summed over this axis.

    Parameters
    ----------
    data : relax.Expr
        The input tensor.

    shape : Union[PrimExprLike, List[PrimExprLike], Tuple[PrimExprLike], relax.Expr]
        The shape to collapse to.

    Returns
    -------
    result : relax.Expr
        The result tensor of the given shape after summation.
    """
    shape = _convert_shape_to_expr(shape)
    return _ffi_api.collapse_sum_to(data, shape)  # type: ignore
