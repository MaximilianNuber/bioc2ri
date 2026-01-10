# rnames.py
"""Utilities for getting and setting row/column/element names on R and Pandas objects.

This module provides a unified interface for working with names on both R objects
(via rpy2) and Pandas DataFrames. It handles the conversion between Python lists
and R character vectors, with optional length validation.

Functions:
    get_rownames: Get row names from R object or Pandas DataFrame.
    get_colnames: Get column names from R object or Pandas DataFrame.
    set_rownames: Set row names on R object or Pandas DataFrame.
    set_colnames: Set column names on R object or Pandas DataFrame.
    get_names: Get names attribute from R vector/list.
    set_names: Set names attribute on R vector/list.
"""
from typing import Optional, Sequence, Union

from rpy2.robjects import baseenv, r, vectors as rv
from rpy2.rinterface import NULL
from rpy2.rinterface_lib.sexp import NULLType, Sexp
from rpy2.robjects.packages import importr

__author__ = "MaximilianNuber"
__copyright__ = "MaximilianNuber"
__license__ = "MIT"

RObj = Union[Sexp, NULLType]

# Grab R replacement functions once
_rn_set = baseenv.__getitem__("rownames<-")
_cn_set = baseenv.__getitem__("colnames<-")

# For set_names
_rstats = importr("stats")
_setNames = _rstats.setNames
_names = baseenv.__getitem__("names")


def _to_charvec_or_null(names: Optional[Sequence[object]]) -> Union[rv.StrSexpVector, type(NULL)]:
    """Converts a Python sequence to an R character vector or NULL.

    Args:
        names: A sequence of objects to convert to strings, or None.

    Returns:
        An R StrSexpVector if names is not None, otherwise R NULL.
    """
    if names is None:
        return NULL
    return rv.StrSexpVector([str(x) for x in names])


def _len_or_none(x: RObj) -> Optional[int]:
    """Extracts the number of rows from a dimensioned R object.

    For matrices, DataFrames, etc., this returns dim(x)[0] (nrow).
    For objects without dimensions, returns None.

    Args:
        x: An R object.

    Returns:
        The number of rows, or None if the object has no dimensions.
    """
    d = r["dim"](x)
    if isinstance(d, NULLType):
        return None
    d = list(d)
    return int(d[0]) if d else None


# ---------------------------
#         GETTERS
# ---------------------------

def get_rownames(x: Union[RObj, "pandas.DataFrame"]) -> Optional[list]:
    """Returns row names from an R object or Pandas DataFrame.

    Args:
        x: An R object (Sexp) or a Pandas DataFrame.

    Returns:
        A list of row names as strings, or None if no row names are set.
    """
    if isinstance(x, (Sexp, NULLType)):
        rn = r["rownames"](x)
        if isinstance(rn, NULLType):
            return None
        return [str(v) for v in rn]
    # Pandas path
    try:
        if hasattr(x, "index"):
            return [str(v) for v in x.index.tolist()]
    except Exception:
        pass
    return None


def get_colnames(x: Union[RObj, "pandas.DataFrame"]) -> Optional[list]:
    """Returns column names from an R object or Pandas DataFrame.

    Args:
        x: An R object (Sexp) or a Pandas DataFrame.

    Returns:
        A list of column names as strings, or None if no column names are set.
    """
    if isinstance(x, (Sexp, NULLType)):
        cn = r["colnames"](x)
        if isinstance(cn, NULLType):
            return None
        return [str(v) for v in cn]
    try:
        if hasattr(x, "columns"):
            return [str(v) for v in x.columns.tolist()]
    except Exception:
        pass
    return None


# ---------------------------
#         SETTERS
# ---------------------------

def set_rownames(
    x: Union[RObj, "pandas.DataFrame"],
    names: Optional[Sequence[object]],
    *,
    strict_len: bool = False,
) -> Union[RObj, "pandas.DataFrame"]:
    """Sets row names on an R object or Pandas DataFrame.

    For R objects, uses the R replacement function `rownames<-`.
    For Pandas DataFrames, sets the index.

    Args:
        x: An R object (Sexp) or a Pandas DataFrame.
        names: A sequence of names to set, or None to clear names.
        strict_len: If True, raises ValueError if len(names) != nrow.

    Returns:
        The modified object.

    Raises:
        ValueError: If strict_len is True and lengths don't match.
        TypeError: If x is neither an R object nor a Pandas DataFrame.
    """
    if isinstance(x, (Sexp, NULLType)):
        if strict_len:
            n = _len_or_none(x)
            if n is not None and names is not None and n != len(names):
                raise ValueError(f"Row name length {len(names)} != nrow {n}")
        return _rn_set(x, _to_charvec_or_null(names))

    # Pandas path
    try:
        if hasattr(x, "index"):
            if names is None:
                x.index = range(len(x))
            else:
                if strict_len and len(names) != len(x):
                    raise ValueError(f"Row name length {len(names)} != nrow {len(x)}")
                x.index = [str(v) for v in names]
            return x
    except Exception:
        pass
    raise TypeError("set_rownames expects an R object (Sexp) or a pandas DataFrame.")


def set_colnames(
    x: Union[RObj, "pandas.DataFrame"],
    names: Optional[Sequence[object]],
    *,
    strict_len: bool = False,
) -> Union[RObj, "pandas.DataFrame"]:
    """Sets column names on an R object or Pandas DataFrame.

    For R objects, uses the R replacement function `colnames<-`.
    For Pandas DataFrames, sets the columns attribute.

    Args:
        x: An R object (Sexp) or a Pandas DataFrame.
        names: A sequence of names to set, or None to clear names.
        strict_len: If True, raises ValueError if len(names) != ncol.

    Returns:
        The modified object.

    Raises:
        ValueError: If strict_len is True and lengths don't match.
        TypeError: If x is neither an R object nor a Pandas DataFrame.
    """
    if isinstance(x, (Sexp, NULLType)):
        if strict_len:
            d = r["dim"](x)
            if not isinstance(d, NULLType) and len(list(d)) >= 2 and names is not None:
                ncol = int(list(d)[1])
                if ncol != len(names):
                    raise ValueError(f"Column name length {len(names)} != ncol {ncol}")
        return _cn_set(x, _to_charvec_or_null(names))

    try:
        if hasattr(x, "columns"):
            if names is None:
                x.columns = list(range(len(x.columns)))
            else:
                if strict_len and len(names) != len(x.columns):
                    raise ValueError(f"Column name length {len(names)} != ncol {len(x.columns)}")
                x.columns = [str(v) for v in names]
            return x
    except Exception:
        pass
    raise TypeError("set_colnames expects an R object (Sexp) or a pandas DataFrame.")


# ---------------------------
#    GENERIC NAMES (vectors)
# ---------------------------

def get_names(x: RObj) -> Optional[list]:
    """Returns names(x) as a Python list of strings.

    Args:
        x: An R object (Sexp).

    Returns:
        A list of names as strings, or None if no names are set.

    Raises:
        TypeError: If x is not an R object.
    """
    if not isinstance(x, (Sexp, NULLType)):
        raise TypeError("get_names expects an rpy2 R object (Sexp or NULL).")
    nm = _names(x)
    if isinstance(nm, NULLType):
        return None
    return [str(v) for v in nm]


def set_names(
    x: RObj,
    names: Optional[Sequence[object]],
    *,
    strict_len: bool = False,
) -> RObj:
    """Sets names(x) using R's setNames().

    Args:
        x: An R object (Sexp).
        names: A sequence of names to set, or None to remove names.
        strict_len: If True, raises ValueError if len(names) != length(x).

    Returns:
        The R object with names set.

    Raises:
        ValueError: If strict_len is True and lengths don't match.
        TypeError: If x is not an R object.
    """
    if not isinstance(x, (Sexp, NULLType)):
        raise TypeError("set_names expects an rpy2 R object (Sexp or NULL).")

    if names is None:
        nm = NULL
    else:
        nm_list = [str(v) for v in names]
        if strict_len:
            n = int(r["length"](x)[0])
            if n != len(nm_list):
                raise ValueError(f"names length {len(nm_list)} != object length {n}")
        nm = rv.StrSexpVector(nm_list)

    return _setNames(x, nm)