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
import pytest
import tvm
from tvm import relax as rx
from tvm.script import relax as R
from tvm.script import tir as T


@tvm.register_func("test.op.identity")
def identity_packed(a):
    return tvm.nd.array(a.asnumpy())


@T.prim_func
def identity_tir(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [54, 96])
    B = T.match_buffer(b, [54, 96])

    for i, j in T.grid(54, 96):
        with T.block("compute"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj]


def test_call_tir() -> None:
    v0 = rx.Var("v0", R.Tensor((54, 96), "float32"))
    v1 = rx.call_tir(rx.extern("test.op.identity"), [v0], R.Tensor((54, 96), "float32"))
    v1 = rx.call_tir(identity_tir, [v0], R.Tensor((54, 96), "float32"))


def test_call_tir_with_shape_var():
    shape_var = rx.Var("shape", R.Shape((32, 32)))
    call = rx.call_tir("extern_func", args=[], output_tensor_sinfo=R.Tensor(shape_var, "float32"))
    assert isinstance(call.args[2].struct_info, rx.struct_info.ShapeStructInfo)
    tvm.ir.assert_structural_equal(call.args[2].values, call.args[2].struct_info.values)


def test_call_tir_invalid_output_shape_1():
    with pytest.raises(ValueError):
        rx.call_tir("extern_func", args=[], output_tensor_sinfo=R.Tensor(ndim=2, dtype="float32"))


def test_call_tir_invalid_output_shape_2():
    with pytest.raises(TypeError):
        rx.call_tir("extern_func", args=[], output_tensor_sinfo=R.Shape((32, 32)))


def test_call_tir_invalid_output_dtype():
    with pytest.raises(ValueError):
        rx.call_tir("extern_func", args=[], output_tensor_sinfo=R.Tensor((32, 32)))


def test_call_tir_mixed_output_shape():
    shape_var = rx.Var("shape", R.Shape((32,)))
    call = rx.call_tir(
        "extern_func",
        args=[],
        output_tensor_sinfo=[R.Tensor(shape_var, "float32"), R.Tensor((64,), "int32")],
    )
    assert isinstance(call.args[2].struct_info, rx.struct_info.TupleStructInfo)

    shape_sinfo = call.args[2].struct_info
    for shape_field, sinfo_field in zip(call.args[2], shape_sinfo.fields, strict=True):
        assert isinstance(sinfo_field, rx.struct_info.ShapeStructInfo)
        tvm.ir.assert_structural_equal(shape_field.values, sinfo_field.values)


if __name__ == "__main__":
    pytest.main([__file__])
