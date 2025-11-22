import numpy as np
from rpy2.robjects import r, vectors as rv
from rpy2.rinterface import NA_Logical, NA_Integer, NA_Real, NULLType

from bioc2ri.numpy_plugin import numpy_plugin

eng = numpy_plugin()

def test_numpy_scalars_to_r():
    # bool -> BoolSexpVector
    r_bool = eng.py2r(np.bool_(True))
    assert isinstance(r_bool, rv.BoolSexpVector)
    assert list(r_bool) == [True]

    # small int -> IntSexpVector
    r_i32 = eng.py2r(np.int32(42))
    assert isinstance(r_i32, rv.IntSexpVector)
    assert list(r_i32) == [42]

    # large int (out of 32-bit range) -> FloatSexpVector
    r_big = eng.py2r(np.int64(2**40))
    assert isinstance(r_big, rv.FloatSexpVector)
    assert float(list(r_big)[0]) == float(2**40)

    # float -> FloatSexpVector
    r_f = eng.py2r(np.float64(1.5))
    assert isinstance(r_f, rv.FloatSexpVector)
    assert float(list(r_f)[0]) == 1.5

    # complex -> ComplexSexpVector
    r_c = eng.py2r(np.complex128(1+2j))
    assert isinstance(r_c, rv.ComplexSexpVector)
    assert complex(list(r_c)[0]) == 1+2j

def test_numpy_1d_float_array_to_r_vector():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    r_vec = eng.py2r(arr)
    assert isinstance(r_vec, rv.FloatSexpVector)
    assert list(r_vec) == [1.0, 2.0, 3.0]


def test_numpy_2d_float_array_to_r_matrix():
    # shape (2, 3), row-major in NumPy
    arr = np.array([[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0]], dtype=np.float64)
    r_mat = eng.py2r(arr)

    # R side: should be a matrix / array with dim c(2, 3)
    dims = list(r["dim"](r_mat))
    assert dims == [2, 3]

    # Check column-major order: c(1,4,2,5,3,6)
    vals = list(r_mat)
    assert vals == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]


def test_r_float_vector_with_na_to_numpy():
    r_vec = r("c(1.5, NA_real_, 3.5)")
    arr = eng.r2py(r_vec)

    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float64
    assert arr.shape == (3,)
    assert np.isclose(arr[0], 1.5)
    assert np.isnan(arr[1])
    assert np.isclose(arr[2], 3.5)

def test_r_int_vector_no_na_to_numpy_int32():
    r_vec = r("c(1L, 2L, 3L)")
    arr = eng.r2py(r_vec)

    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.int32
    assert arr.shape == (3,)
    assert np.all(arr == np.array([1, 2, 3], dtype=np.int32))


def test_r_int_vector_with_na_to_numpy_float64():
    r_vec = r("c(1L, NA_integer_, 3L)")
    arr = eng.r2py(r_vec)

    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float64
    assert arr.shape == (3,)
    assert arr[0] == 1.0
    assert np.isnan(arr[1])
    assert arr[2] == 3.0

def test_r_logical_vector_no_na_to_numpy_bool():
    r_vec = r("c(TRUE, FALSE, TRUE)")
    arr = eng.r2py(r_vec)

    assert isinstance(arr, np.ndarray)
    assert arr.dtype == bool
    assert arr.shape == (3,)
    assert np.all(arr == np.array([True, False, True], dtype=bool))


def test_r_logical_vector_with_na_to_numpy_object():
    r_vec = r("c(TRUE, NA, FALSE)")
    arr = eng.r2py(r_vec)

    assert isinstance(arr, np.ndarray)
    assert arr.dtype == object
    assert arr.shape == (3,)
    assert arr[0] is True
    assert arr[1] is None
    assert arr[2] is False

def test_r_matrix_to_numpy_2d():
    # 2x3 matrix in R: matrix(1:6, nrow=2, ncol=3)
    r_mat = r("matrix(1:6, nrow = 2, ncol = 3)")
    arr = eng.r2py(r_mat)

    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 3)

    # In R: columns are (1,2), (3,4), (5,6)
    # So the NumPy array (Fortran reshape) should be:
    expected = np.array([[1, 3, 5],
                         [2, 4, 6]], dtype=np.float64)
    assert np.allclose(arr, expected)


test_numpy_scalars_to_r()
test_numpy_1d_float_array_to_r_vector()
test_numpy_2d_float_array_to_r_matrix()
test_r_float_vector_with_na_to_numpy()
test_r_int_vector_no_na_to_numpy_int32()
test_r_int_vector_with_na_to_numpy_float64()
test_r_logical_vector_no_na_to_numpy_bool()
test_r_logical_vector_with_na_to_numpy_object()
test_r_matrix_to_numpy_2d()