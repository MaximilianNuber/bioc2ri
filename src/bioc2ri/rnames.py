# rnames.py
from typing import Iterable, Optional, Sequence, Union
from rpy2.robjects import r, baseenv, vectors as rv
from rpy2.rinterface_lib.sexp import Sexp, NULLType
from rpy2.rinterface import NULL
import rpy2

RObj = Union[Sexp, NULLType]

# Grab R replacement functions once
_rn_set = baseenv.__getitem__("rownames<-")   # function(x, value) in R
_cn_set = baseenv.__getitem__("colnames<-")   # function(x, value) in R

def _to_charvec_or_null(names: Optional[Sequence[object]]):
    if names is None:
        return NULL
    # R expects character; coerce to str and build StrSexpVector
    return rv.StrSexpVector([str(x) for x in names])

def _len_or_none(x: RObj) -> Optional[int]:
    # For matrices/DataFrames/SE/SCE this equals nrow/ncol when applied to dim parts.
    d = r["dim"](x)
    if isinstance(d, NULLType):
        return None
    d = list(d)
    return d[0] if d else None

# ---------------------------
#         GETTERS
# ---------------------------

def get_rownames(x: Union[RObj, "pandas.DataFrame"]) -> Optional[list]:
    """Return a list of row names or None if absent."""
    # R side
    if isinstance(x, (Sexp, NULLType)):
        rn = r["rownames"](x)
        if isinstance(rn, NULLType):
            return None
        return [str(v) for v in rn]
    # Pandas (optional)
    try:
        import pandas as pd  # noqa
        if hasattr(x, "index") and not isinstance(x, (Sexp, NULLType)):
            return [str(v) for v in x.index.tolist()]
    except Exception:
        pass
    return None

def get_colnames(x: Union[RObj, "pandas.DataFrame"]) -> Optional[list]:
    """Return a list of column names or None if absent."""
    if isinstance(x, (Sexp, NULLType)):
        cn = r["colnames"](x)
        if isinstance(cn, NULLType):
            return None
        return [str(v) for v in cn]
    try:
        import pandas as pd  # noqa
        if hasattr(x, "columns") and not isinstance(x, (Sexp, NULLType)):
            return [str(v) for v in x.columns.tolist()]
    except Exception:
        pass
    return None

# ---------------------------
#         SETTERS
# ---------------------------

def set_rownames(x: Union[RObj, "pandas.DataFrame"],
                 names: Optional[Sequence[object]],
                 *, strict_len: bool = False):
    """
    Set row names. On R objects, modifies and returns the object (R replacement function).
    On pandas, sets DataFrame.index.
    If strict_len=True, raises if length mismatches nrow.
    """
    if isinstance(x, (Sexp, NULLType)):
        if strict_len:
            n = _len_or_none(x)
            if n is not None and names is not None and n != len(names):
                raise ValueError(f"Row name length {len(names)} != nrow {n}")
        return _rn_set(x, _to_charvec_or_null(names))  # returns modified R object

    # Pandas path
    try:
        import pandas as pd  # noqa
        if hasattr(x, "set_index") and hasattr(x, "index"):
            if names is None:
                # reset to RangeIndex
                x.index = range(len(x))
            else:
                if strict_len and len(names) != len(x):
                    raise ValueError(f"Row name length {len(names)} != nrow {len(x)}")
                x.index = [str(v) for v in names]
            return x
    except Exception:
        pass
    raise TypeError("set_rownames expects an R object (Sexp) or a pandas DataFrame.")

def set_colnames(x: Union[RObj, "pandas.DataFrame"],
                 names: Optional[Sequence[object]],
                 *, strict_len: bool = False):
    """
    Set column names. On R objects, modifies and returns the object (R replacement function).
    On pandas, sets DataFrame.columns.
    If strict_len=True, raises if length mismatches ncol.
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
        import pandas as pd  # noqa
        if hasattr(x, "columns") and not isinstance(x, (Sexp, NULLType)):
            if names is None:
                # Default numbered columns
                x.columns = list(range(len(x.columns)))
            else:
                if strict_len and len(names) != len(x.columns):
                    raise ValueError(f"Column name length {len(names)} != ncol {len(x.columns)}")
                x.columns = [str(v) for v in names]
            return x
    except Exception:
        pass
    raise TypeError("set_colnames expects an R object (Sexp) or a pandas DataFrame.")



from typing import Optional, Sequence, Union
from rpy2.robjects import baseenv, r, vectors as rv
from rpy2.rinterface_lib.sexp import Sexp, NULLType
from rpy2.rinterface import NULL
from rpy2.robjects.packages import importr

RObj = Union[Sexp, NULLType]
rstats = importr("stats")
_setNames = rstats.setNames  # R::setNames(object, nm)
_names    = baseenv.__getitem__("names")     # R::names(object)

def get_names(x: RObj) -> Optional[list]:
    """Return names(x) as a Python list of str, or None if names are absent."""
    if not isinstance(x, (Sexp, NULLType)):
        raise TypeError("get_names expects an rpy2 R object (Sexp or NULL).")
    nm = _names(x)
    if isinstance(nm, NULLType):
        return None
    return [str(v) for v in nm]

def set_names(x: RObj,
              names: Optional[Sequence[object]],
              *,
              strict_len: bool = False):
    """
    Set names(x) using R's setNames(). Returns the modified R object.
    - names=None removes names.
    - strict_len=True enforces len(names) == length(x) when names is not None.
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

    # setNames returns the object with names set
    return _setNames(x, nm)