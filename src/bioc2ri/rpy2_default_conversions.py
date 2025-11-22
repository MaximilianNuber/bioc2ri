from functools import singledispatch
from typing import Sequence, Union, Any, Tuple
# import types

from rpy2 import robjects as ro
from rpy2.robjects import vectors, methods, default_converter, conversion
from rpy2.robjects.vectors import ListVector
from rpy2.robjects.methods import RS4
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.conversion import get_conversion
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage as STAP
from anndata2ri import scipy2ri
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix
import os, tempfile
import h5py
from .exceptions import RPackageNotLoadedError

# from rpy_conversions.exceptions import RPackageNotLoadedError

def lazy_import_r_packages(packages: Union[str, Sequence[str]]) -> Tuple:
    """
    Lazily import one or more R packages via rpy2.robjects.packages.importr.
    Raises RPackageNotLoadedError on failure.
    """
    from rpy2.robjects.packages import importr

    if isinstance(packages, str):
        try:
            return importr(packages)
        except Exception as e:
            raise RPackageNotLoadedError(f"Could not load R pkg '{packages}': {e}") from e

    modules = []
    failures = []
    for pkg in packages:
        try:
            modules.append(importr(pkg))
        except Exception:
            modules.append(pkg)
            failures.append(pkg)
    if failures:
        raise RPackageNotLoadedError(f"Failed to load R packages: {', '.join(failures)}")
    return tuple(modules)


# ----------------------
# Python -> R converters
# ----------------------
@singledispatch
def _py_to_r(obj) -> ro.RObject:
    """Convert a generic Python object to an R object.

    Uses rpy2’s default conversion rules to map:
      - Atomic Python sequences (uniform lists/tuples) → R atomic vectors
      - Mixed‐type lists or dicts → R lists (ListVector)
      - Primitive types via their registered overloads

    Args:
        obj: Any Python object.

    Returns:
        ro.RObject: An R object representing `obj`.
    """
    with localconverter(default_converter):
        return conversion.py2rpy(obj)

@_py_to_r.register(type(None))
def _(obj) -> ro.NULL:    # noqa: F811
    """Convert Python None to R NULL.

    Args:
        obj (None): The Python None value.

    Returns:
        ro.NULL: The R NULL singleton.
    """
    return ro.NULL

@_py_to_r.register(bool)
def _(obj: bool) -> vectors.BoolVector:    # noqa: F811
    """Convert a Python boolean to a length‐1 R logical vector.

    Args:
        obj (bool): A Python boolean.

    Returns:
        BoolVector: An R logical vector of length 1 containing `obj`.
    """
    return vectors.BoolVector([obj])

@_py_to_r.register(int)
def _(obj: int) -> vectors.IntVector:    # noqa: F811
    """Convert a Python integer to a length‐1 R integer vector.

    Args:
        obj (int): A Python integer.

    Returns:
        IntVector: An R integer vector of length 1 containing `obj`.
    """
    return vectors.IntVector([obj])

@_py_to_r.register(float)
def _(obj: float) -> vectors.FloatVector:    # noqa: F811
    """Convert a Python float to a length‐1 R numeric vector.

    Args:
        obj (float): A Python float.

    Returns:
        FloatVector: An R numeric vector of length 1 containing `obj`.
    """
    return vectors.FloatVector([obj])

@_py_to_r.register(str)
def _(obj: str) -> vectors.StrVector:    # noqa: F811
    """Convert a Python string to a length‐1 R character vector.

    Args:
        obj (str): A Python string.

    Returns:
        StrVector: An R character vector of length 1 containing `obj`.
    """
    return vectors.StrVector([obj])

@_py_to_r.register(dict)
def _(obj: dict) -> ListVector:    # noqa: F811
    """Convert a Python dict to an R named list.

    Recursively applies `_py_to_r` to each value, and uses the dict keys
    as names in the resulting R list.

    Args:
        obj (dict): A mapping of keys to Python values.

    Returns:
        ListVector: An R named list with the same keys and converted values.
    """
    rdict = {str(k): _py_to_r(v) for k, v in obj.items()}
    return ListVector(rdict)

@_py_to_r.register(np.ndarray)
def _(obj: np.ndarray) -> ro.Matrix:    # noqa: F811
    """Convert a NumPy array to an R array or vector.

    Uses numpy2ri for seamless handling of:
      - 1D arrays → R atomic vectors
      - 2D arrays → R matrices
      - Higher‐dimensional arrays → R arrays

    Args:
        obj (np.ndarray): A NumPy array of any shape and dtype.

    Returns:
        ro.Matrix or Vector: The equivalent R array/matrix/vector.
    """
    with localconverter(default_converter + numpy2ri.converter):
        return numpy2ri.py2rpy(obj)

@_py_to_r.register(pd.DataFrame)
def _(obj: pd.DataFrame) -> ro.DataFrame:    # noqa: F811
    """Convert a pandas DataFrame to an R data.frame.

    Preserves column names, row names, and dtypes via pandas2ri.

    Args:
        obj (pd.DataFrame): A pandas DataFrame.

    Returns:
        ro.DataFrame: The equivalent R data.frame.
    """
    with localconverter(default_converter + pandas2ri.converter):
        return conversion.py2rpy(obj)

@_py_to_r.register(pd.Series)
def _(obj: pd.Series) -> vectors.Vector:    # noqa: F811
    """Convert a pandas Series to an R vector.

    Preserves the Series name as an R vector name if available.

    Args:
        obj (pd.Series): A pandas Series.

    Returns:
        Vector: The equivalent R atomic vector.
    """
    with localconverter(default_converter + pandas2ri.converter):
        return conversion.py2rpy(obj)

@_py_to_r.register(csr_matrix)
def _(obj: csr_matrix) -> ro.Matrix:    # noqa: F811
    """Convert a SciPy CSR sparse matrix to an R sparse matrix.

    Args:
        obj (csr_matrix): A SciPy CSR sparse matrix.

    Returns:
        ro.Matrix: An R dgCMatrix or similar sparse S4 object.
    """
    with localconverter(default_converter + scipy2ri.converter):
        return conversion.py2rpy(obj)

@_py_to_r.register(csc_matrix)
def _(obj: csc_matrix) -> ro.Matrix:    # noqa: F811
    """Convert a SciPy CSC sparse matrix to an R sparse matrix.

    Args:
        obj (csc_matrix): A SciPy CSC sparse matrix.

    Returns:
        ro.Matrix: An R dgCMatrix or similar sparse S4 object.
    """
    with localconverter(default_converter + scipy2ri.converter):
        return conversion.py2rpy(obj)

# ----------------------
# R -> Python converters
# ----------------------
@singledispatch
def _r_to_py(obj: ro.RObject): # noqa: F811
    """
    Generic R to Python conversion using rpy2's default converter.

    Args:
        obj: An R object of unknown type.

    Returns:
        Python equivalent of the R object via generic rpy2 conversion.
    """
    with localconverter(default_converter):
        return conversion.rpy2py(obj)

@_r_to_py.register(type(ro.NULL))
def _(obj) -> None: # noqa: F811
    """
    Convert R NULL to Python None.

    Args:
        obj: R NULL object.

    Returns:
        None
    """
    return None

@_r_to_py.register(vectors.BoolVector)
def _(obj): # noqa: F811
    """
    Convert an R logical vector to Python bool or list of bools.

    Args:
        obj: rpy2.robjects.vectors.BoolVector

    Returns:
        Single bool if length 1, else list of bools.
    """
    vals = list(obj)
    return bool(vals[0]) if len(vals) == 1 else [bool(x) for x in vals]

@_r_to_py.register(vectors.IntVector)
def _(obj): # noqa: F811
    """
    Convert an R integer vector to Python int or list of ints.

    Args:
        obj: rpy2.robjects.vectors.IntVector

    Returns:
        Single int if length 1, else list of ints.
    """
    vals = list(obj)
    return int(vals[0]) if len(vals) == 1 else vals

@_r_to_py.register(vectors.FloatVector)
def _(obj): # noqa: F811
    """
    Convert an R numeric (float) vector to Python float or list of floats.

    Args:
        obj: rpy2.robjects.vectors.FloatVector

    Returns:
        Single float if length 1, else list of floats.
    """
    vals = list(obj)
    return float(vals[0]) if len(vals) == 1 else vals

@_r_to_py.register(vectors.StrVector)
def _(obj): # noqa: F811
    """
    Convert an R string vector to Python str or list of str.

    Args:
        obj: rpy2.robjects.vectors.StrVector

    Returns:
        Single str if length 1, else list of str.
    """
    vals = list(obj)
    return str(vals[0]) if len(vals) == 1 else vals

@_r_to_py.register(ListVector)
def _(obj): # noqa: F811
    """
    Convert an R ListVector to Python dict or list.

    Args:
        obj: rpy2.robjects.vectors.ListVector

    Returns:
        dict if ListVector has names, else list.
    """
    items = [_r_to_py(obj[i]) for i in range(len(obj))]
    names = list(obj.names) if obj.names != ro.NULL else None
    return dict(zip(names, items)) if names else items

@_r_to_py.register(vectors.Matrix)
def _(obj: vectors.Matrix): # noqa: F811
    """
    Convert an R matrix to a NumPy array.

    Args:
        obj: rpy2.robjects.vectors.Matrix

    Returns:
        numpy.ndarray via numpy2ri converter.
    """
    with localconverter(default_converter + numpy2ri.converter):
        return conversion.rpy2py(obj)

@_r_to_py.register(ro.DataFrame)
def _(obj: ro.DataFrame): # noqa: F811
    """
    Convert an R DataFrame to a pandas.DataFrame.

    Args:
        obj: rpy2.robjects.DataFrame

    Returns:
        pandas.DataFrame via pandas2ri converter.
    """
    with localconverter(default_converter + pandas2ri.converter):
        return conversion.rpy2py(obj)

@_r_to_py.register(vectors.Vector)
def _(obj: vectors.Vector): # noqa: F811
    """
    Convert a generic R atomic vector to a Python list.

    Args:
        obj: rpy2.robjects.vectors.Vector

    Returns:
        list of elements.
    """
    return list(obj)

@_r_to_py.register(RS4)
def _(obj: RS4): # noqa: F811
    """
    Convert an R S4 object.

    Handles sparse Matrix S4 classes via scipy2ri; otherwise, returns
    a dict of slot values.

    Args:
        obj: rpy2.robjects.methods.RS4

    Returns:
        dict or sparse matrix conversion.
    """
    # Handle sparse S4 via scipy2ri
    cls = obj.rclass[0]
    if cls in ("dgCMatrix", "dgRMatrix"):
        with localconverter(default_converter + scipy2ri.converter):
            cv = get_conversion()
            return cv.rpy2py(obj)
    # Otherwise, map slots to dict
    return {name: _r_to_py(obj.slots[name]) for name in obj.slotnames()}


### Sparse CSC matrix conversion via HDF5 (for large matrices) ###

def write_csc_hdf5(X, path, rownames=None, colnames=None):
    X = X.tocsc(copy=False)
    # Ensure canonical form: no dupes, sorted row indices per column
    X.sum_duplicates()
    X.sort_indices()

    with h5py.File(path, "w") as f:
        g = f.create_group("csc")
        g.create_dataset("data",    data=X.data)                       # float64
        g.create_dataset("indices", data=X.indices.astype(np.int32))   # int32
        g.create_dataset("indptr",  data=X.indptr.astype(np.int32))    # int32
        g.create_dataset("shape",   data=np.array(X.shape, np.int64))
        if rownames is not None:
            g.create_dataset("rownames", data=np.asarray(rownames, dtype="S"))
        if colnames is not None:
            g.create_dataset("colnames", data=np.asarray(colnames, dtype="S"))

_r_code = r"""
read_csc_hdf5 <- function(path) {
  # Fast HDF5 -> dgCMatrix (expects datasets under group "csc")
  suppressPackageStartupMessages({
    library(rhdf5)
    library(Matrix)
  })
  ls <- rhdf5::h5ls(path, recursive = TRUE)
  has_rn <- any(ls$name == "rownames" & ls$group == "/csc")
  has_cn <- any(ls$name == "colnames" & ls$group == "/csc")

  data    <- rhdf5::h5read(path, "csc/data")
  indices <- as.integer(rhdf5::h5read(path, "csc/indices"))
  indptr  <- as.integer(rhdf5::h5read(path, "csc/indptr"))
  shape   <- as.integer(rhdf5::h5read(path, "csc/shape"))

  rn <- if (has_rn) as.character(rhdf5::h5read(path, "csc/rownames")) else NULL
  cn <- if (has_cn) as.character(rhdf5::h5read(path, "csc/colnames")) else NULL

  new("dgCMatrix",
      x  = as.numeric(data),
      i  = indices,
      p  = indptr,
      Dim = shape,
      Dimnames = list(rn, cn))
}
"""

cscio = STAP(_r_code, "cscio")

def csc_to_r_dgc_via_hdf5(
    X_csc: csc_matrix,
    *,
    rownames=None,
    colnames=None,
    filename: str = "matrix.h5",
):
    """
    One-shot handoff: SciPy CSC -> temp HDF5 -> R dgCMatrix (via STAP).
    Returns an rpy2 S4 object of class 'dgCMatrix'.
    """
    if not isinstance(X_csc, csc_matrix):
        X_csc = X_csc.tocsc(copy=False)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, filename)
        write_csc_hdf5(X_csc, path, rownames=rownames, colnames=colnames)
        r_mat = cscio.read_csc_hdf5(path)   # R dgCMatrix (rpy2 S4)
        # The R object is materialized in memory; deleting the temp file is fine.
        return r_mat
    



_r_write = r"""
write_csc_hdf5 <- function(M, path) {
  suppressPackageStartupMessages({
    library(Matrix)
    library(rhdf5)
  })
  # Coerce to dgCMatrix if needed
  if (!inherits(M, "dgCMatrix")) {
    if (inherits(M, "CsparseMatrix")) {
      M <- as(M, "dgCMatrix")
    } else {
      M <- as(M, "dgCMatrix")  # will error if incompatible
    }
  }
  if (file.exists(path)) {
    try(unlink(path), silent = TRUE)
  }
  rhdf5::h5createFile(path)
  rhdf5::h5createGroup(path, "csc")

  rhdf5::h5write(M@x,           path, "csc/data")                 # numeric (double)
  rhdf5::h5write(as.integer(M@i), path, "csc/indices")            # int32
  rhdf5::h5write(as.integer(M@p), path, "csc/indptr")             # int32
  rhdf5::h5write(as.integer(M@Dim), path, "csc/shape")            # (nrow, ncol)

  dn <- dimnames(M)
  if (!is.null(dn[[1]])) rhdf5::h5write(as.character(dn[[1]]), path, "csc/rownames")
  if (!is.null(dn[[2]])) rhdf5::h5write(as.character(dn[[2]]), path, "csc/colnames")

  invisible(TRUE)
}
"""
cscio_w = STAP(_r_write, "cscio_w")

def read_csc_hdf5_py(path, *, return_names=False):
    """
    Read an HDF5 file with group 'csc' (data, indices, indptr, shape, optional row/colnames)
    and return a SciPy csc_matrix. Optionally returns (X, rownames, colnames).
    """
    with h5py.File(path, "r") as f:
        g = f["csc"]
        data = g["data"][()]                            # float64
        indices = g["indices"][()].astype(np.int32)     # int32
        indptr  = g["indptr"][()].astype(np.int32)      # int32
        shape   = tuple(np.array(g["shape"][()], dtype=np.int64))  # (nrow, ncol)

        X = csc_matrix((data, indices, indptr), shape=shape, copy=False)

        if not return_names:
            return X

        def _read_names(key):
            if key in g:
                arr = g[key][()]
                # bytes -> str; return as Python list
                return np.array(arr, dtype=str).tolist()
            return None

        rn = _read_names("rownames")
        cn = _read_names("colnames")
        return X, rn, cn
    
def r_dgc_to_scipy_via_hdf5(r_mat, *, filename="matrix.h5", return_names=False):
    """
    Takes an R 'dgCMatrix' (rpy2 S4 object), writes it to a temp HDF5 via STAP,
    then reads it back as a SciPy csc_matrix. Optionally returns names.
    """
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, filename)
        # write on the R side (ensures dgCMatrix and canonical CSC)
        cscio_w.write_csc_hdf5(r_mat, path)
        return read_csc_hdf5_py(path, return_names=return_names)