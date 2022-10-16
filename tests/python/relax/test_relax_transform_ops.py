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
from tvm.script import relax as R, tir as T
import tvm.testing

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


if __name__ == "__main__":
    test_transpose()
    test_transpose_none_arg()
    test_transpose_fail_on_duplicate_indices()
