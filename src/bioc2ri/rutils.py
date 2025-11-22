from typing import Union, Iterable, Optional
from rpy2.robjects import r, baseenv
from rpy2.rinterface import Sexp, NULLType

__author__ = "MaximilianNuber"
__copyright__ = "MaximilianNuber"
__license__ = "MIT"

RObj = Union[Sexp, NULLType]

def is_r(obj) -> bool:
    return isinstance(obj, (Sexp, NULLType))

def r_class(x: RObj) -> Iterable[str]:
    if isinstance(x, NULLType):
        return ("NULL",)
    return tuple(r["class"](x))

def r_len(x: RObj) -> int:
    if isinstance(x, NULLType):
        return 0
    return int(r["length"](x)[0])

def r_dim(x: RObj) -> Optional[tuple]:
    if isinstance(x, NULLType):
        return None
    d = r["dim"](x)
    # d can be NULL
    from rpy2.rinterface_lib.sexp import NULLType as _NULL
    if isinstance(d, _NULL):
        return None
    return tuple(int(v) for v in d)

def r_names(x: RObj) -> Optional[list]:
    if isinstance(x, NULLType):
        return None
    n = r["names"](x)
    from rpy2.rinterface_lib.sexp import NULLType as _NULL
    if isinstance(n, _NULL):
        return None
    return [str(v) for v in n]

def r_print(x: RObj):
    if not is_r(x):
        raise TypeError("r_print expects an rpy2 R object (Sexp or NULL).")
    baseenv["print"](x)

def r_str(x: RObj):
    r["str"](x)

def r_summary(x: RObj):
    baseenv["print"](baseenv["summary"](x))