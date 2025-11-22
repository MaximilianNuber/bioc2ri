import numpy as np
from rpy2.robjects import r, vectors as rv
from rpy2.robjects.packages import importr

from biocframe import BiocFrame
from biocutils import BooleanList, IntegerList, FloatList, StringList

# from bioc2ri.biocpy_plugin import biocpy_plugin

eng = biocpy_plugin()
S4Vectors = importr("S4Vectors")

def test_booleanlist_to_r():
    bl = BooleanList([True, None, False])
    r_vec = eng.py2r(bl)

    assert isinstance(r_vec, rv.BoolSexpVector)

    vals = list(r_vec)
    is_na = list(r["is.na"](r_vec))

    assert vals[0] is True
    assert vals[2] is False
    assert is_na == [False, True, False]


def test_integerlist_to_r():
    il = IntegerList([1, None, 3])
    r_vec = eng.py2r(il)

    assert isinstance(r_vec, rv.IntSexpVector)

    vals = list(r_vec)
    is_na = list(r["is.na"](r_vec))

    assert vals[0] == 1
    assert vals[2] == 3
    assert is_na == [False, True, False]


def test_floatlist_to_r():
    fl = FloatList([1.5, None, 3.5])
    r_vec = eng.py2r(fl)

    assert isinstance(r_vec, rv.FloatSexpVector)

    vals = list(r_vec)
    is_na = list(r["is.na"](r_vec))

    assert np.isclose(vals[0], 1.5)
    assert is_na[1] is True
    assert np.isclose(vals[2], 3.5)


def test_stringlist_to_r():
    sl = StringList(["a", None, "c"])
    r_vec = eng.py2r(sl)

    assert isinstance(r_vec, rv.StrSexpVector)

    vals = list(r_vec)
    is_na = list(r["is.na"](r_vec))

    assert vals[0] == "a"
    assert vals[2] == "c"
    assert is_na == [False, True, False]


def test_r_logical_to_booleanlist():
    r_vec = r("c(TRUE, NA, FALSE)")
    py = eng.r2py(r_vec)

    assert isinstance(py, BooleanList)
    assert list(py) == [True, None, False]


def test_r_integer_to_integerlist():
    r_vec = r("c(1L, NA_integer_, 3L)")
    py = eng.r2py(r_vec)

    assert isinstance(py, IntegerList)
    assert list(py) == [1, None, 3]


def test_r_numeric_to_floatlist():
    r_vec = r("c(1.5, NA_real_, 3.5)")
    py = eng.r2py(r_vec)

    assert isinstance(py, FloatList)
    vals = list(py)
    assert np.isclose(vals[0], 1.5)
    assert vals[1] is None
    assert np.isclose(vals[2], 3.5)


def test_r_character_to_stringlist():
    r_vec = r('c("a", NA_character_, "c")')
    py = eng.r2py(r_vec)

    assert isinstance(py, StringList)
    assert list(py) == ["a", None, "c"]


def test_r_factor_to_stringlist():
    r_fac = r('factor(c("x", NA, "y"), levels = c("x", "y"))')
    py = eng.r2py(r_fac)

    assert isinstance(py, StringList)
    assert list(py) == ["x", None, "y"]


def test_biocframe_python_lists_to_s4_dataframe():
    bf = BiocFrame(
        {
            "ints":   [1, None, 3],
            "floats": [0.5, None, 1.5],
            "bools":  [True, None, False],
            "strs":   ["a", None, "c"],
        },
        row_names=["r1", "r2", "r3"],
    )

    r_df = eng.py2r(bf)

    # class is DataFrame/DFrame
    cls = list(r["class"](r_df))
    assert ("DataFrame" in cls) or ("DFrame" in cls)

    # colnames
    cn = list(r["names"](r_df))
    assert cn == ["ints", "floats", "bools", "strs"]

    # rownames
    rn = list(r["rownames"](r_df))
    assert rn == ["r1", "r2", "r3"]

    # columns: all atomic vectors, not list columns
    col_get = r["[["]  # S4Vectors DataFrame: x[[name]]

    ints   = col_get(r_df, "ints")
    floats = col_get(r_df, "floats")
    bools  = col_get(r_df, "bools")
    strs   = col_get(r_df, "strs")

    assert list(r["typeof"](ints))[0]   == "integer"
    assert list(r["typeof"](floats))[0] == "double"
    assert list(r["typeof"](bools))[0]  == "logical"
    assert list(r["typeof"](strs))[0]   == "character"

    # NA positions
    is_na_ints   = list(r["is.na"](ints))
    is_na_floats = list(r["is.na"](floats))
    is_na_bools  = list(r["is.na"](bools))
    is_na_strs   = list(r["is.na"](strs))

    assert is_na_ints   == [False, True, False]
    assert is_na_floats == [False, True, False]
    assert is_na_bools  == [False, True, False]
    assert is_na_strs   == [False, True, False]


def test_biocframe_with_biocutils_cols_to_s4_and_back():
    bf = BiocFrame(
        {
            "ints":   IntegerList([1, None, 3]),
            "floats": FloatList([0.5, None, 1.5]),
            "bools":  BooleanList([True, None, False]),
            "strs":   StringList(["a", None, "c"]),
        },
        row_names=["r1", "r2", "r3"],
    )

    r_df = eng.py2r(bf)

    cls = list(r["class"](r_df))
    assert ("DataFrame" in cls) or ("DFrame" in cls)

    col_get = r["[["]
    ints_r   = col_get(r_df, "ints")
    floats_r = col_get(r_df, "floats")
    bools_r  = col_get(r_df, "bools")
    strs_r   = col_get(r_df, "strs")

    assert list(r["typeof"](ints_r))[0]   == "integer"
    assert list(r["typeof"](floats_r))[0] == "double"
    assert list(r["typeof"](bools_r))[0]  == "logical"
    assert list(r["typeof"](strs_r))[0]   == "character"

    # roundtrip to Python
    bf2 = eng.r2py(r_df)
    print(bf2)

    assert isinstance(bf2, BiocFrame)
    print(type(bf2.row_names))
    print(bf2.row_names==["r1", "r2", "r3"])
    assert list(bf2.row_names) == ["r1", "r2", "r3"]
    assert list(bf2.column_names) == ["ints", "floats", "bools", "strs"]

    cols2 = {name: col for name, col in bf2._data.items()}

    assert isinstance(cols2["ints"], IntegerList)
    assert isinstance(cols2["floats"], FloatList)
    assert isinstance(cols2["bools"], BooleanList)
    assert isinstance(cols2["strs"], StringList)

    assert list(cols2["ints"])   == [1, None, 3]
    assert list(cols2["floats"]) == [0.5, None, 1.5]
    assert list(cols2["bools"])  == [True, None, False]
    assert list(cols2["strs"])   == ["a", None, "c"]

def test_biocframe_roundtrip_python_lists_to_biocutils_cols():
    bf = BiocFrame(
        {
            "ints":   [1, None, 3],
            "floats": [0.5, None, 1.5],
            "bools":  [True, None, False],
            "strs":   ["a", None, "c"],
        },
        row_names=["r1", "r2", "r3"],
    )

    r_df = eng.py2r(bf)
    bf2 = eng.r2py(r_df)

    assert isinstance(bf2, BiocFrame)
    assert list(bf2.row_names) == ["r1", "r2", "r3"]
    assert list(bf2.column_names) == ["ints", "floats", "bools", "strs"]

    cols2 = {name: col for name, col in bf2._data.items()}

    assert isinstance(cols2["ints"], IntegerList)
    assert isinstance(cols2["floats"], FloatList)
    assert isinstance(cols2["bools"], BooleanList)
    assert isinstance(cols2["strs"], StringList)

    assert list(cols2["ints"])   == [1, None, 3]
    assert list(cols2["floats"]) == [0.5, None, 1.5]
    assert list(cols2["bools"])  == [True, None, False]
    assert list(cols2["strs"])   == ["a", None, "c"]

def test_biocframe_heterogeneous_column_raises():
    bf = BiocFrame(
        {
            "bad": [1, "x", None],  # int + str mixed
        },
        row_names=["r1", "r2", "r3"],
    )

    try:
        eng.py2r(bf)
    except TypeError:
        pass
    else:
        raise AssertionError("Expected TypeError for heterogeneous BiocFrame column")


test_booleanlist_to_r()
test_integerlist_to_r()
test_floatlist_to_r()
test_stringlist_to_r()

test_r_logical_to_booleanlist()
test_r_integer_to_integerlist()
test_r_numeric_to_floatlist()
test_r_character_to_stringlist()
test_r_factor_to_stringlist()

test_biocframe_python_lists_to_s4_dataframe()
test_biocframe_with_biocutils_cols_to_s4_and_back()
test_biocframe_roundtrip_python_lists_to_biocutils_cols()
test_biocframe_heterogeneous_column_raises()