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

import tvm.testing
from tvm import IRModule
from tvm.script.parser import ir as I, tir as T, relax as R


def _check(mod: IRModule):
    print(mod.script())


def test_simple_module():
    @I.ir_module
    class TestModule:
        @T.prim_func
        def tir_func(x: T.Buffer((128, 128), "float32"), y: T.Buffer((128, 128), "float32")):
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((128, 128), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv0 = R.call_tir(tir_func, x, (128, 128), dtype="float32")
            return gv0

    _check(TestModule)


def test_simple_func():
    @R.function
    def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor(None, "float32", ndim=2):
        gv0 = R.call_tir("extern_func", x, (128, 128), dtype="float32")
        return gv0

    _check(foo)


def test_symbolic_shape():
    @R.function
    def foo(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(None, "float32", ndim=2):
        m = T.var("int32", "m")
        n = T.var("int32", "n")
        gv0 = R.call_tir("extern_func", x, (m, n), dtype="float32")
        return gv0

    _check(foo)


def test_symbolic_shape_without_giving_names():
    @R.function
    def foo(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(None, "float32", ndim=2):
        m = T.var("int32")
        n = T.var("int32")
        gv0 = R.call_tir("extern_func", x, (m, n), dtype="float32")
        return gv0

    m0 = foo.params[0].shape[0]
    m1 = foo.body.blocks[0].bindings[0].value.args[2].values[0]
    assert m0 == m1
    _check(foo)


def test_directly_return():
    @R.function
    def foo(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(None, "float32", ndim=2):
        return x

    _check(foo)


def test_call_packed():
    @R.function
    def foo(x: R.Tensor((3, 4), "float32")) -> R.Tensor(None, "float32", ndim=2):
        z = R.call_packed("vm.builtin.copy", x, type_args=R.Tensor(None, "float32", ndim=2))
        return z

    _check(foo)


def test_relax_op():
    @R.function
    def foo(
        x: R.Tensor((4, 4), "float32"), w: R.Tensor((4, 4), "float32")
    ) -> R.Tensor(None, "float32", ndim=2):
        y = R.add(x, w)
        z = R.multiply(x, y)
        return z

    _check(foo)


def test_deduce_func_type():
    @R.function
    def foo(x: R.Tensor((3, 4), "float32")):
        z = R.call_packed("vm.builtin.copy", x, type_args=(R.Tensor(None, dtype="float32", ndim=2)))
        return z

    _check(foo)


def test_match_shape():
    @R.function
    def foo(x: R.Tensor(None, "float32")):
        m = T.var("int64")
        n = T.var("int64")
        R.match_shape(x, (n, m))
        return (n * T.int64(2), m * T.int64(3))

    _check(foo)


def test_match_shape_with_binding():
    @R.function
    def foo(x: R.Tensor(None, "float32")):
        m = T.var("int64")
        n = T.var("int64")
        y = R.match_shape(x, (n, m))
        return (n * 2, m * 3)

    _check(foo)


def test_builtin():
    @R.function
    def foo(x: R.Tensor(("m", "n"), "float32")):
        m = T.var("int64", "m")
        n = T.var("int64", "n")
        alloc = R.builtin.alloc_tensor((m, n), runtime_device_index=0, dtype="float32")
        _ = R.call_packed(
            "test.op.identity", x, alloc, type_args=(R.Tensor(ndim=2, dtype="float32"))
        )
        gv0 = alloc
        return gv0

    _check(foo)


@pytest.mark.xfail()
def test_error_report():
    @R.function
    def foo(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(None, dtype="float32", ndim=2):
        m = T.var("int64", "m")
        m = T.var("int64", "mm")
        n = T.var("int64", "n")
        z = R.call_packed("vm.builtin.copy", x, type_args=(R.Tensor(None, dtype="float32", ndim=2)))
        return z


def test_shadowing():
    @R.function
    def foo(
        x: R.Tensor((4, 4), "float32"), w: R.Tensor((4, 4), "float32")
    ) -> R.Tensor(None, "float32", ndim=2):
        y = R.add(x, w)
        z = R.multiply(x, y)
        y = R.add(x, y)
        y = z
        y = R.multiply(w, x)
        z = y
        return y

    _check(foo)


def test_dataflow_block():
    @R.function
    def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor(None, "float32", ndim=2):
        with R.dataflow():
            lv0 = R.call_tir("extern_func", x, (128, 128), dtype="float32")
            gv = R.call_tir("extern_func", lv0, (128, 128), dtype="float32")
            R.output(gv)
        return gv

    _check(foo)


def test_dataflow_block_advanced():
    @R.function
    def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor(None, "float32", ndim=2):
        gv0 = R.call_tir("extern_func", x, (128, 128), dtype="float32")
        gv1 = R.call_tir("extern_func", gv0, (128, 128), dtype="float32")
        with R.dataflow():
            m = T.var("int64")
            n = T.var("int64")
            lv0 = R.call_tir("extern_func", gv1, (128, 128), dtype="float32")
            lv1 = R.match_shape(lv0, (m, n))
            gv2 = R.call_tir("extern_func", lv0, (128, 128), dtype="float32")
            gv2 = R.call_tir("extern_func", gv2, (128, 128), dtype="float32")
            gv3 = R.match_shape(gv2, (m, n))
            gv3 = R.match_shape(lv0, (m, n))

            R.output(gv3, gv2)
        gv4 = R.call_tir("extern_func", gv2, (128, 128), dtype="float32")
        gv5 = R.call_tir("extern_func", gv4, (128, 128), dtype="float32")
        return gv5

    _check(foo)


@pytest.mark.xfail()
def test_dataflow_binding_after_output():
    @R.function
    def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor(None, "float32", ndim=2):
        with R.dataflow():
            gv = R.call_tir("extern_func", x, (128, 128), dtype="float32")
            R.output(gv)
            lv = R.call_tir("extern_func", gv, (128, 128), dtype="float32")
        return gv


@pytest.mark.xfail()
def test_dataflow_output_global_var():
    @R.function
    def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor(None, "float32", ndim=2):
        gv0 = R.call_tir("extern_func", x, (128, 128), dtype="float32")
        with R.dataflow():
            gv1 = R.call_tir("extern_func", gv0, (128, 128), dtype="float32")
            R.output(gv0, gv1)
        return gv1


@pytest.mark.xfail()
def test_dataflow_multiple_output():
    @R.function
    def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor(None, "float32", ndim=2):
        with R.dataflow():
            gv = R.call_tir("extern_func", x, (128, 128), dtype="float32")
            R.output(gv)
            R.output(gv)
        return gv


@pytest.mark.xfail()
def test_dataflow_output_outside_dataflow_block():
    @R.function
    def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor(None, "float32", ndim=2):
        gv = R.call_tir("extern_func", x, (128, 128), dtype="float32")
        R.output(gv)
        return gv


if __name__ == "__main__":
    tvm.testing.main()
