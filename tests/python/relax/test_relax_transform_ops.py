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

from __future__ import annotations  # must import to defer parsing of annotations
import pytest
import numpy as np
import tvm
from tvm import relax
from tvm.error import DiagnosticError
from tvm.relax.testing import transform
from tvm.script._parser import relax as R
import tvm.testing

# Todo(ruihang): switch the unit tests from numpy-result comparison to structural equality check

target_str = "llvm --num-cores=16"
target = tvm.target.Target(target_str)
dev = tvm.device(target_str, 0)


def relax_build_and_run(f, inputs):
    f = f.with_attr("global_symbol", "default")
    mod = tvm.IRModule.from_expr(f)

    with tvm.transform.PassContext(opt_level=3):
        mod = relax.transform.Normalize()(mod)
        mod = transform.LowerWithRelayOpStrategyPass(target)(mod)
        ex = relax.vm.build(mod, target)
        vm = relax.VirtualMachine(ex, dev)
        return vm["default"](*inputs).numpy()


def test_transpose():
    dtype = "float32"
    input_shape = [1, 2, 3, 4]

    tensor_type = relax.DynTensorType(ndim=4, dtype="float32")
    x = relax.Var("x", input_shape, tensor_type)
    y = relax.op.transform.transpose(x, axes=[1, -1, 2, -4])
    f = relax.Function(
        params=[x], body=y, ret_type=tensor_type, ret_shape=relax.ShapeExpr([2, 4, 3, 1])
    )

    x_np = np.random.rand(*input_shape).astype(dtype)
    x_relax = tvm.nd.array(x_np, dev)

    res_np = np.transpose(x_np, axes=[1, -1, 2, -4])
    res_relax = relax_build_and_run(f, [x_relax])

    tvm.testing.assert_allclose(res_relax, res_np)


def test_transpose_none_arg():
    dtype = "float32"
    input_shape = [1, 2, 3, 4]

    tensor_type = relax.DynTensorType(ndim=4, dtype="float32")
    x = relax.Var("x", input_shape, tensor_type)
    y = relax.op.transform.transpose(x, axes=None)
    f = relax.Function(
        params=[x], body=y, ret_type=tensor_type, ret_shape=relax.ShapeExpr([2, 4, 3, 1])
    )

    x_np = np.random.rand(*input_shape).astype(dtype)
    x_relax = tvm.nd.array(x_np, dev)

    res_np = np.transpose(x_np, axes=None)
    res_relax = relax_build_and_run(f, [x_relax])

    tvm.testing.assert_allclose(res_relax, res_np)


def test_transpose_fail_on_duplicate_indices():
    x = relax.Var("x", [1, 2, 3, 4], relax.DynTensorType(ndim=4, dtype="float32"))
    bb = relax.BlockBuilder()
    with pytest.raises(DiagnosticError):
        with bb.function("main", [x]):
            gv = bb.emit(relax.op.transform.transpose(x, axes=[1, -1, 2, 3]))


def test_reshape():
    @R.function
    def expected(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor(None, "float32", ndim=2):
        gv: R.Tensor((8, 3), "float32") = R.reshape(x, (8, 3))
        return gv

    x = relax.Var("x", [1, 2, 3, 4], relax.DynTensorType(ndim=4, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        gv = bb.emit(relax.op.transform.reshape(x, newshape=(8, 3)))
        bb.emit_func_output(gv)

    expected = expected.with_attr("global_symbol", "main")
    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_reshape_infer_dim():
    @R.function
    def expected(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor(None, "float32", ndim=3):
        gv: R.Tensor((8, 1, 3), "float32") = R.reshape(x, (8, -1, 3))
        return gv

    x = relax.Var("x", [1, 2, 3, 4], relax.DynTensorType(ndim=4, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        gv = bb.emit(relax.op.transform.reshape(x, newshape=(8, -1, 3)))
        bb.emit_func_output(gv)

    expected = expected.with_attr("global_symbol", "main")
    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_reshape_fail_on_multiple_inference():
    input_shape = [1, 2, 3, 4]

    tensor_type = relax.DynTensorType(ndim=4, dtype="float32")
    return_type = relax.DynTensorType(ndim=4, dtype="float32")
    x = relax.Var("x", input_shape, tensor_type)
    y = relax.op.transform.reshape(x, newshape=(8, -1, 3, -1))
    f = relax.Function(
        params=[x], body=y, ret_type=return_type, ret_shape=relax.ShapeExpr([8, 1, 3, 1])
    )

    mod = tvm.IRModule.from_expr(f)
    with pytest.raises(DiagnosticError):
        mod = relax.transform.Normalize()(mod)


def test_expand_dims():
    @R.function
    def expected(x: R.Tensor((2, 3, 4), "float32")) -> R.Tensor(None, "float32", ndim=8):
        gv: R.Tensor((2, 1, 1, 1, 3, 1, 4, 1), "float32") = R.expand_dims(x, axis=[-1, 1, -6, 3, 5])
        return gv

    x = relax.Var("x", [2, 3, 4], relax.DynTensorType(ndim=3, dtype="float32"))

    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        gv = bb.emit(relax.op.transform.expand_dims(x, axis=[-1, 1, -6, 3, 5]))
        bb.emit_func_output(gv)

    expected = expected.with_attr("global_symbol", "main")
    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_squeeze():
    @R.function
    def expected(x: R.Tensor((2, 1, 3, 1, 1, 4), "float32")) -> R.Tensor(None, "float32", ndim=3):
        gv: R.Tensor((2, 3, 4), "float32") = R.squeeze(x)
        return gv

    x = relax.Var("x", [2, 1, 3, 1, 1, 4], relax.DynTensorType(ndim=6, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        gv = bb.emit(relax.op.transform.squeeze(x))
        bb.emit_func_output(gv)

    expected = expected.with_attr("global_symbol", "main")
    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_squeeze_with_indices():
    @R.function
    def expected(x: R.Tensor((2, 1, 3, 1, 1, 4), "float32")) -> R.Tensor(None, "float32", ndim=4):
        gv: R.Tensor((2, 3, 1, 4), "float32") = R.squeeze(x, axis=[3, -5])
        return gv

    x = relax.Var("x", [2, 1, 3, 1, 1, 4], relax.DynTensorType(ndim=6, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        gv = bb.emit(relax.op.transform.squeeze(x, axis=[3, -5]))
        bb.emit_func_output(gv)

    expected = expected.with_attr("global_symbol", "main")
    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_squeeze_unable_to_infer():
    @R.function
    def expected(
        x: R.Tensor((2, 1, 3, "a", 1, 4), "float32")
    ) -> R.Tensor(None, "float32", ndim=-1):
        gv: R.Tensor(None, "float32", ndim=-1) = R.squeeze(x)
        return gv

    a = tvm.tir.Var("a", "int64")
    x = relax.Var("x", [2, 1, 3, a, 1, 4], relax.DynTensorType(ndim=6, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        gv = bb.emit(relax.op.transform.squeeze(x))
        bb.emit_func_output(gv)

    expected = expected.with_attr("global_symbol", "main")
    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_squeeze_force_squeezing_with_indices():
    @R.function
    def expected(x: R.Tensor((2, 1, 3, "a", 1, 4), "float32")) -> R.Tensor(None, "float32", ndim=5):
        gv: R.Tensor((2, 1, 3, 1, 4), "float32") = R.squeeze(x, axis=3)
        return gv

    a = tvm.tir.Var("a", "int64")
    x = relax.Var("x", [2, 1, 3, a, 1, 4], relax.DynTensorType(ndim=6, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        gv = bb.emit(relax.op.transform.squeeze(x, axis=3))
        bb.emit_func_output(gv)

    expected = expected.with_attr("global_symbol", "main")
    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_squeeze_with_indices_fail_on_non_unit_dim():
    a = tvm.tir.Var("a", "int64")
    x = relax.Var("x", [2, 1, 3, a, 1, 4], relax.DynTensorType(ndim=6, dtype="float32"))
    bb = relax.BlockBuilder()
    with pytest.raises(DiagnosticError):
        with bb.function("main", [x]):
            gv = bb.emit(relax.op.transform.squeeze(x, axis=2))
            bb.emit_func_output(gv)


def test_concatenate():
    @R.function
    def expected(
        x1: R.Tensor((1, 2, 3), "float32"),
        x2: R.Tensor((1, 3, 3), "float32"),
        x3: R.Tensor((1, 4, 3), "float32"),
    ) -> R.Tensor(None, "float32", ndim=3):
        gv: R.Tensor((1, 9, 3), "float32") = R.concatenate((x1, x2, x3), axis=1)
        return gv

    x1 = relax.Var("x1", [1, 2, 3], relax.DynTensorType(ndim=3, dtype="float32"))
    x2 = relax.Var("x2", [1, 3, 3], relax.DynTensorType(ndim=3, dtype="float32"))
    x3 = relax.Var("x3", [1, 4, 3], relax.DynTensorType(ndim=3, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x1, x2, x3]):
        gv = bb.emit(relax.op.transform.concatenate((x1, x2, x3), axis=1))
        bb.emit_func_output(gv)

    expected = expected.with_attr("global_symbol", "main")
    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_concatenate_fail_on_incompatible_shape():
    x1 = relax.Var("x1", [1, 2, 3], relax.DynTensorType(ndim=3, dtype="float32"))
    x2 = relax.Var("x2", [2, 3, 3], relax.DynTensorType(ndim=3, dtype="float32"))
    x3 = relax.Var("x3", [1, 4, 3], relax.DynTensorType(ndim=3, dtype="float32"))
    bb = relax.BlockBuilder()
    with pytest.raises(DiagnosticError):
        with bb.function("main", [x1, x2, x3]):
            gv = bb.emit(relax.op.transform.concatenate((x1, x2, x3), axis=1))
            bb.emit_func_output(gv)


def test_concatenate_without_specified_axis():
    @R.function
    def expected(
        x1: R.Tensor((2,), "float32"), x2: R.Tensor((3,), "float32"), x3: R.Tensor((4,), "float32")
    ) -> R.Tensor(None, "float32", ndim=1):
        gv: R.Tensor((9,), "float32") = R.concatenate((x1, x2, x3), axis=None)
        return gv

    x1 = relax.Var("x1", [2], relax.DynTensorType(ndim=1, dtype="float32"))
    x2 = relax.Var("x2", [3], relax.DynTensorType(ndim=1, dtype="float32"))
    x3 = relax.Var("x3", [4], relax.DynTensorType(ndim=1, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x1, x2, x3]):
        gv = bb.emit(relax.op.transform.concatenate((x1, x2, x3), axis=None))
        bb.emit_func_output(gv)

    expected = expected.with_attr("global_symbol", "main")
    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


if __name__ == "__main__":
    test_transpose()
    test_transpose_none_arg()
    test_transpose_fail_on_duplicate_indices()
    test_reshape()
    test_reshape_infer_dim()
    test_reshape_fail_on_multiple_inference()
    test_expand_dims()
    test_squeeze()
    test_squeeze_with_indices()
    test_squeeze_unable_to_infer()
    test_squeeze_force_squeezing_with_indices()
    test_squeeze_with_indices_fail_on_non_unit_dim()
    test_concatenate()
    test_concatenate_fail_on_incompatible_shape()
    test_concatenate_without_specified_axis()
