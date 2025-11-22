import numpy as np
import pandas as pd
from rpy2.robjects import r, vectors as rv
from rpy2.rinterface import NA_Real

#from bioc2ri.pandas_plugin import pandas_plugin

eng = pandas_plugin()

from bioc2ri.src.bioc2ri.rnames import set_rownames

def test_series_float_to_r_vector():
    s = pd.Series([1.0, np.nan, 3.0], name="x")
    r_vec = eng.py2r(s)

    # Should be a numeric (double) vector
    assert isinstance(r_vec, rv.FloatSexpVector)
    assert list(r["typeof"](r_vec))[0] == "double"

    vals = list(r_vec)
    is_na = list(r["is.na"](r_vec))

    assert np.isclose(vals[0], 1.0)
    assert is_na[1] is True        # np.nan -> NA_real_
    assert np.isclose(vals[2], 3.0)

def test_series_categorical_to_factor():
    cat = pd.Categorical(
        ["a", "b", "a", None],
        categories=["a", "b"],
        ordered=True,
    )
    s = pd.Series(cat, name="cat")

    r_fac = eng.py2r(s)

    # R side: should be a factor
    cls = list(r["class"](r_fac))
    assert "factor" in cls

    # levels should match
    levels = list(r["levels"](r_fac))
    assert levels == ["a", "b"]

    # values as character (with NA)
    as_char = r["as.character"](r_fac)
    vals = list(as_char)
    is_na = list(r["is.na"](r_fac))

    assert vals[0] == "a"
    assert vals[1] == "b"
    assert vals[2] == "a"
    assert is_na[3] is True

def test_dataframe_py2r_basic():
    df = pd.DataFrame(
        {
            "x": [1.0, np.nan, 3.0],
            "flag": [True, False, True],
            "cat": pd.Categorical(["a", "b", "a"], categories=["a", "b"], ordered=False),
        },
        index=["r1", "r2", "r3"],
    )

    r_df = eng.py2r(df)

    # data.frame class
    cls = list(r["class"](r_df))
    assert "data.frame" in cls

    # columns exist
    colnames = list(r["names"](r_df))
    assert colnames == ["x", "flag", "cat"]

    # rownames set
    rn = list(r["rownames"](r_df))
    print(rn)
    assert rn == ["r1", "r2", "r3"]

    # column types
    x = r_df.rx2("x")
    flag = r_df.rx2("flag")
    cat = r_df.rx2("cat")

    assert list(r["typeof"](x))[0] == "double"
    assert list(r["typeof"](flag))[0] == "logical"
    assert "factor" in list(r["class"](cat))

    # NA in x at second position
    is_na_x = list(r["is.na"](x))
    assert is_na_x == [False, True, False]

def test_r_dataframe_to_pandas():
    r_df = r('data.frame(x = c(1L, 2L, 3L), y = c(0.5, 1.5, 2.5), row.names = c("r1","r2","r3"))')

    df = eng.r2py(r_df)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["x", "y"]
    assert list(df.index) == ["r1", "r2", "r3"]

    # Values (dtypes may be float for x due to numpy rules, we just check numerics)
    assert np.allclose(df["x"].to_numpy(dtype=float), np.array([1, 2, 3], dtype=float))
    assert np.allclose(df["y"].to_numpy(dtype=float), np.array([0.5, 1.5, 2.5], dtype=float))

def test_dataframe_roundtrip():
    df = pd.DataFrame(
        {
            "x": [1.0, np.nan, 3.0],
            "flag": [True, False, True],
            "cat": pd.Categorical(["a", "b", "a"], categories=["a", "b"], ordered=False),
        },
        index=["r1", "r2", "r3"],
    )

    r_df = eng.py2r(df)
    df2 = eng.r2py(r_df)

    assert isinstance(df2, pd.DataFrame)
    assert list(df2.index) == ["r1", "r2", "r3"]
    assert list(df2.columns) == ["x", "flag", "cat"]

    # Numeric column
    x2 = df2["x"].to_numpy(dtype=float)
    assert np.isclose(x2[0], 1.0)
    assert np.isnan(x2[1])
    assert np.isclose(x2[2], 3.0)

    # Logical column
    flag2 = df2["flag"].to_numpy()
    # Might come back as {0,1} or bool depending on R->numpy; just interpret truthiness
    assert flag2.shape == (3,)
    assert bool(flag2[0]) is True
    assert bool(flag2[1]) is False
    assert bool(flag2[2]) is True

    # Categorical column: R factor -> numeric vector of codes by numpy_plugin,
    # then DataFrame sees ints; so we just check they’re 1/2-level-like.
    cat2 = df2["cat"].to_numpy()
    # Should be integer-ish codes with NA possibly as np.nan
    # We only check shape here; more detailed mapping can be done later if you
    # decide to reconstruct pandas.Categorical on the Python side.
    assert cat2.shape == (3,)


test_series_float_to_r_vector()
test_series_categorical_to_factor()
test_dataframe_py2r_basic()
test_r_dataframe_to_pandas()
test_dataframe_roundtrip()