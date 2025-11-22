# Build a small engine with only the "base" rules:
from bioc2ri import base_plugin, numpy_plugin, pandas_plugin, biocpy_plugin, scipy_sparse_plugin


def show_roundtrip(x):
    """Pretty-print a Python → R → Python roundtrip."""
    eng = base_plugin()
    r_obj = eng.py2r(x)
    back = eng.r2py(r_obj)

    print("Python → R → Python roundtrip")
    print("-" * 40)
    print("Python in :")
    pprint(x)
    print("  type    :", type(x).__name__)
    print()
    print("R object  :")
    # R class is often more informative than the Python wrapper type
    r_class = list(r["class"](r_obj))
    print("  repr    :", repr(r_obj))
    print("  R class :", r_class)
    print()
    print("Python out:")
    pprint(back)
    print("  type    :", type(back).__name__)
    print("=" * 40, "\n")


### numpy

from pprint import pprint

import numpy as np
from rpy2.robjects import r
from rpy2.rinterface import NULLType

# Build an engine with only the NumPy rules.


def _r_dims(obj):
    """Return dim(obj) as a Python list or None if dim is NULL."""
    dim = r["dim"](obj)
    if isinstance(dim, NULLType):
        return None
    d = list(dim)
    return d if len(d) else None


def show_np_scalar_roundtrip(x):
    """Pretty-print a NumPy scalar → R → Python roundtrip."""
    eng = numpy_plugin()
    r_obj = eng.py2r(x)
    back = eng.r2py(r_obj)

    print("NumPy scalar → R → Python")
    print("-" * 40)
    print("Python in :")
    print("  value   :", x)
    print("  type    :", type(x))
    print()
    print("R object  :")
    r_class = list(r["class"](r_obj))
    print("  repr    :", repr(r_obj))
    print("  R class :", r_class)
    print("  dim     :", _r_dims(r_obj))
    print()
    print("Python out:")
    print("  value   :", back)
    print("  type    :", type(back))
    print("=" * 40, "\n")


def show_ndarray_roundtrip(a: np.ndarray):
    """Pretty-print a NumPy ndarray → R → NumPy ndarray roundtrip."""
    eng = numpy_plugin()
    r_obj = eng.py2r(a)
    back = eng.r2py(r_obj)

    print("NumPy ndarray → R → NumPy ndarray")
    print("-" * 40)
    print("Python in :")
    print("  value   :")
    print(a)
    print("  dtype   :", a.dtype)
    print("  shape   :", a.shape)
    print()
    r_class = list(r["class"](r_obj))
    print("R object  :")
    print("  repr    :", repr(r_obj))
    print("  R class :", r_class)
    print("  dim     :", _r_dims(r_obj))
    print()
    print("Python out:")
    print("  value   :")
    print(back)
    print("  dtype   :", getattr(back, "dtype", None))
    print("  shape   :", getattr(back, "shape", None))
    print("=" * 40, "\n")


### pandas_plugin

from pprint import pprint

import numpy as np
import pandas as pd

# Build an engine with pandas + NumPy rules.
# (pandas_plugin() internally uses numpy_plugin for the actual array work.)


def show_series_roundtrip(s: pd.Series):
    """Pretty-print a pandas.Series → R → Python roundtrip."""
    eng = pandas_plugin()
    r_obj = eng.py2r(s)
    back = eng.r2py(r_obj)

    print("pandas.Series → R → Python")
    print("-" * 40)
    print("Python in :")
    print("  name    :", s.name)
    print("  dtype   :", s.dtype)
    print("  values  :")
    print(s.to_numpy())
    print()
    print("R object  :")
    r_class = list(r["class"](r_obj))
    print("  repr    :", repr(r_obj))
    print("  R class :", r_class)
    print()
    print("Python out (eng.r2py):")
    print("  value   :", back)
    print("  type    :", type(back))
    print("=" * 40, "\n")


def show_dataframe_roundtrip(df: pd.DataFrame):
    """Pretty-print pandas.DataFrame → R data.frame → pandas.DataFrame."""
    eng = pandas_plugin()
    r_obj = eng.py2r(df)
    back = eng.r2py(r_obj)

    print("pandas.DataFrame → R data.frame → pandas.DataFrame")
    print("-" * 40)
    print("Python in :")
    print("  shape   :", df.shape)
    print("  dtypes  :")
    print(df.dtypes)
    print("  data    :")
    print(df)
    print()
    print("R object  :")
    r_class = list(r["class"](r_obj))
    rn = list(r["rownames"](r_obj)) if r["rownames"](r_obj)[0] is not None else None
    print("  repr    :", repr(r_obj))
    print("  R class :", r_class)
    print("  rownames:", rn)
    print()
    print("Python out (eng.r2py):")
    print("  type    :", type(back))
    print("  shape   :", back.shape if isinstance(back, pd.DataFrame) else None)
    if isinstance(back, pd.DataFrame):
        print("  dtypes  :")
        print(back.dtypes)
        print("  data    :")
        print(back)
    else:
        pprint(back)
    print("=" * 40, "\n")


## scipy_sparse_plugin

from pprint import pprint

import numpy as np
import scipy.sparse as sp
from rpy2.robjects import r

# Engine with sparse rules


# small helper to inspect R sparse matrices
slot = r["slot"]


def _r_dim(s4):
    """Return Dim slot of an R Matrix S4 object as (nrow, ncol)."""
    try:
        d = list(slot(s4, "Dim"))
        return int(d[0]), int(d[1])
    except Exception:
        return None


def show_sparse_roundtrip(mat, label: str = ""):
    """Pretty-print SciPy sparse → R Matrix S4 → SciPy sparse roundtrip."""
    eng = scipy_sparse_plugin()
    r_obj = eng.py2r(mat)
    back = eng.r2py(r_obj)

    print("SciPy sparse → R Matrix → SciPy sparse")
    print("-" * 60)
    if label:
        print(f"Label      : {label}")
    print("Python in  :")
    print("  type     :", type(mat))
    print("  format   :", getattr(mat, "format", None))
    print("  shape    :", mat.shape)
    print("  nnz      :", mat.nnz)
    print("  dtype    :", mat.dtype)
    print()
    print("R object   :")
    r_class = list(r["class"](r_obj))
    print("  repr     :", repr(r_obj))
    print("  R class  :", r_class)
    print("  Dim      :", _r_dim(r_obj))
    print()
    print("Python out (eng.r2py):")
    print("  type     :", type(back))
    if sp.isspmatrix(back):
        print("  format   :", back.format)
        print("  shape    :", back.shape)
        print("  nnz      :", back.nnz)
        print("  dtype    :", back.dtype)
        # optional: compare to original
        same_shape = (back.shape == mat.shape)
        same_nnz = (back.nnz == mat.nnz)
        print("  same shape?:", same_shape)
        print("  same nnz?  :", same_nnz)
    else:
        pprint(back)
    print("=" * 60, "\n")


### biocpy_plugin

from pprint import pprint

from rpy2.robjects import r

from biocframe import BiocFrame
from biocutils import BooleanList, IntegerList, FloatList, StringList

# Engine with BiocPy/BiocUtils + NumPy rules



def show_biocframe_roundtrip(bf: BiocFrame, label: str = ""):
    """Pretty-print BiocFrame → S4Vectors::DataFrame → BiocFrame."""
    eng = biocpy_plugin()
    r_obj = eng.py2r(bf)
    back = eng.r2py(r_obj)

    print("BiocFrame → S4Vectors::DataFrame → BiocFrame")
    print("-" * 60)
    if label:
        print(f"Label         : {label}")
    print("Python in (BiocFrame):")
    print("  nrow        :", bf.shape[0])
    print("  ncol        :", bf.shape[1])
    print("  row_names   :", bf.row_names)
    print("  columns     :")
    for name, col in bf._data.items():
        print(f"    {name!r}: type={type(col).__name__}, len={len(col)}")
    print()
    print("R object (S4Vectors::DataFrame / DFrame):")
    r_class = list(r["class"](r_obj))
    rn = list(r["rownames"](r_obj)) if r["rownames"](r_obj)[0] is not None else None
    print("  repr        :", repr(r_obj))
    print("  R class     :", r_class)
    print("  rownames    :", rn)
    print()
    print("Python out (eng.r2py):")
    print("  type        :", type(back))
    if isinstance(back, BiocFrame):
        print("  nrow        :", back.shape[0])
        print("  ncol        :", back.shape[1])
        print("  row_names   :", back.row_names)
        print("  columns     :")
        for name, col in back._data.items():
            print(f"    {name!r}: type={type(col).__name__}, len={len(col)}")
    else:
        pprint(back)
    print("=" * 60, "\n")


def show_r_vector_to_biocutils_list(expr: str, label: str = ""):
    """Show how a raw R vector becomes a BiocUtils list."""
    eng = biocpy_plugin()
    r_vec = r(expr)
    py = eng.r2py(r_vec)

    print("R atomic vector → BiocUtils list")
    print("-" * 60)
    if label:
        print(f"Label         : {label}")
    print("R expression  :", expr)
    print("R class       :", list(r["class"](r_vec)))
    print("Python out    :")
    print("  value       :", py)
    print("  type        :", type(py).__name__)
    print("=" * 60, "\n")