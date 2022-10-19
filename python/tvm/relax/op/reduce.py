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
"""Reduction operators."""
from typing import List, Tuple, Optional, Union

from . import _ffi_api
from ..expr import Expr


def sum(
    data: Expr, axis: Optional[Union[int, List[int], Tuple[int]]] = None, keepdims: bool = False
) -> Expr:
    """Computes the sum of array elements over given axes.

    Parameters
    ----------
    data : relax.Expr
        The input data

    axis : Optional[Union[int, List[int], Tuple[int]]]
        Axis or axes along which a sum is performed. The default, axis=None,
        will sum all of the elements of the input array. If axis is
        negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast
        correctly against the input array.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    if isinstance(axis, int):
        axis = [axis]
    return _ffi_api.sum(data, axis, keepdims)


def mean(
    data: Expr, axis: Optional[Union[int, List[int], Tuple[int]]] = None, keepdims: bool = False
) -> Expr:
    """Computes the mean of array elements over given axes.

    Parameters
    ----------
    data : relax.Expr
        The input data

    axis : Optional[Union[int, List[int], Tuple[int]]]
        Axis or axes along which a mean operation is performed.
        The default, axis=None, will compute the mean of all elements in the input array.
        If axis is negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    if isinstance(axis, int):
        axis = [axis]
    return _ffi_api.mean(data, axis, keepdims)


def variance(
    data: Expr, axis: Optional[Union[int, List[int], Tuple[int]]] = None, keepdims: bool = False
) -> Expr:
    """Computes the variance of array elements over given axes.

    Parameters
    ----------
    data : relax.Expr
        The input data

    axis : Optional[Union[int, List[int], Tuple[int]]]
        Axis or axes along which a mean operation is performed.
        The default, axis=None, will compute the mean of all elements in the input array.
        If axis is negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    if isinstance(axis, int):
        axis = [axis]
    return _ffi_api.variance(data, axis, keepdims)


def max(
    data: Expr, axis: Optional[Union[int, List[int], Tuple[int]]] = None, keepdims: bool = False
) -> Expr:
    """Computes the max of array elements over given axes.

    Parameters
    ----------
    data : relax.Expr
        The input data

    axis : Optional[Union[int, List[int], Tuple[int]]]
        Axis or axes along which a mean operation is performed.
        The default, axis=None, will compute the mean of all elements in the input array.
        If axis is negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    if isinstance(axis, int):
        axis = [axis]
    return _ffi_api.max(data, axis, keepdims)


def min(
    data: Expr, axis: Optional[Union[int, List[int], Tuple[int]]] = None, keepdims: bool = False
) -> Expr:
    """Computes the min of array elements over given axes.

    Parameters
    ----------
    data : relax.Expr
        The input data

    axis : Optional[Union[int, List[int], Tuple[int]]]
        Axis or axes along which a mean operation is performed.
        The default, axis=None, will compute the mean of all elements in the input array.
        If axis is negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    if isinstance(axis, int):
        axis = [axis]
    return _ffi_api.min(data, axis, keepdims)
