from bioc2ri.src.bioc2ri import pandas_plugin, base_plugin, numpy_plugin, biocpy_plugin, scipy_sparse_plugin

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple, Type
from rpy2.robjects import r, vectors as rv
# from rpy2.rinterface_lib.sexp import SexpS4
from rpy2.rinterface import SexpS4
eng = base_plugin.base_plugin()

from rpy2.robjects import r, vectors as rv
from rpy2.rinterface import NULL

# If you're in a notebook and base_plugin is already defined, just do:
eng = base_plugin.base_plugin()

def test_base_none_bool_int_float_str_complex_bytes_to_r():
    # None -> NULL
    r_null = eng.py2r(None)
    assert r_null is NULL

    # bool -> BoolSexpVector length 1
    r_bool = eng.py2r(True)
    print(r_bool)
    assert isinstance(r_bool, (rv.BoolSexpVector, rv.IntSexpVector))
    assert list(r_bool) == [True]

    # int -> IntSexpVector length 1 (32-bit range)
    r_int = eng.py2r(42)
    assert isinstance(r_int, rv.IntSexpVector)
    assert list(r_int) == [42]

    # float -> FloatSexpVector length 1
    r_float = eng.py2r(3.5)
    assert isinstance(r_float, rv.FloatSexpVector)
    assert list(r_float)[0] == 3.5

    # str -> StrSexpVector length 1
    r_str = eng.py2r("foo")
    assert isinstance(r_str, rv.StrSexpVector)
    assert list(r_str) == ["foo"]

    # complex -> ComplexSexpVector length 1
    r_cplx = eng.py2r(1+2j)
    assert isinstance(r_cplx, rv.ComplexSexpVector)
    assert complex(list(r_cplx)[0]) == 1+2j

    # bytes -> ByteSexpVector
    r_raw = eng.py2r(b"\x01\x02")
    assert isinstance(r_raw, rv.ByteSexpVector)
    assert list(r_raw) == [1, 2]
test_base_none_bool_int_float_str_complex_bytes_to_r()

def test_base_dict_list_tuple_to_r_list():
    # dict -> named ListVector
    obj = {"a": 1, "b": True}
    r_list = eng.py2r(obj)
    assert isinstance(r_list, rv.ListVector)
    assert list(r_list.names) == ["a", "b"]
    # values recursively converted
    _get = r["[["]
    a = _get(r_list, "a")
    b = _get(r_list, "b")
    assert isinstance(a, rv.IntSexpVector)
    assert isinstance(b, (rv.BoolSexpVector, rv.IntSexpVector))
    assert list(a) == [1]
    assert list(b) == [True]

    # list -> ListSexpVector (unnamed)
    obj_list = [1, "x", False]
    r_list2 = eng.py2r(obj_list)
    assert isinstance(r_list2, rv.ListSexpVector)
    assert (r_list2.names) == NULL

    # tuple -> ListSexpVector (unnamed)
    obj_tuple = (1, 2)
    r_list3 = eng.py2r(obj_tuple)
    assert isinstance(r_list3, rv.ListSexpVector)
    assert len(r_list3) == 2
test_base_dict_list_tuple_to_r_list()

import numpy as np
def test_base_r_int_with_na_to_python():
    r_vec = r("c(1L, NA_integer_, 3L)")
    py = eng.r2py(r_vec)
    # length>1 -> list of ints/None
    assert isinstance(py, list)
    assert py == [1, None, 3]
test_base_r_int_with_na_to_python()

def test_base_r_float_with_na_to_python():
    r_vec = r("c(1.5, NA_real_, 3.5)")
    py = eng.r2py(r_vec)
    print(py)
    assert isinstance(py, list)
    assert abs(py[0] - 1.5) < 1e-8
    assert np.isnan(py[1])
    assert abs(py[2] - 3.5) < 1e-8
test_base_r_float_with_na_to_python()

def test_base_r_logical_with_na_to_python():
    r_vec = r("c(TRUE, NA, FALSE)")
    py = eng.r2py(r_vec)
    assert isinstance(py, list)
    assert py == [True, None, False]
test_base_r_logical_with_na_to_python


def test_base_r_character_with_na_to_python():
    r_vec = r('c("a", NA_character_, "c")')
    py = eng.r2py(r_vec)
    assert isinstance(py, list)
    assert py == ["a", None, "c"]
test_base_r_character_with_na_to_python()

def test_base_r_scalar_vectors_to_python_scalars():
    r_i = r("1L")
    r_f = r("1.5")
    r_b = r("TRUE")
    r_s = r('"foo"')

    assert eng.r2py(r_i) == 1
    assert abs(eng.r2py(r_f) - 1.5) < 1e-8
    assert eng.r2py(r_b) is True
    assert eng.r2py(r_s) == "foo"

    # NAs -> None for length 1
    r_na_i = r("NA_integer_")
    r_na_f = r("NA_real_")
    r_na_b = r("NA")
    r_na_s = r("NA_character_")

    assert eng.r2py(r_na_i) is None
    assert np.isnan(eng.r2py(r_na_f))
    assert eng.r2py(r_na_b) is None
    assert eng.r2py(r_na_s) is None
test_base_r_scalar_vectors_to_python_scalars()

def test_base_r_named_list_to_dict():
    r_named = r("list(a = 1L, b = 2L)")
    py = eng.r2py(r_named)
    assert isinstance(py, dict)
    assert py["a"] == 1
    assert py["b"] == 2


def test_base_r_unnamed_list_to_list():
    r_unnamed = r("list(1L, 2L, 3L)")
    py = eng.r2py(r_unnamed)
    assert isinstance(py, list)
    assert py == [1, 2, 3]
test_base_r_named_list_to_dict()
test_base_r_unnamed_list_to_list()

test_base_none_bool_int_float_str_complex_bytes_to_r()
test_base_dict_list_tuple_to_r_list()
test_base_r_int_with_na_to_python()
test_base_r_float_with_na_to_python()
test_base_r_logical_with_na_to_python()
test_base_r_character_with_na_to_python()
test_base_r_scalar_vectors_to_python_scalars()
test_base_r_named_list_to_dict()
test_base_r_unnamed_list_to_list()