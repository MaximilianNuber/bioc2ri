from functools import cache
from .engine import Engine

__author__ = "MaximilianNuber"
__copyright__ = "MaximilianNuber"
__license__ = "MIT"

@cache
def numpy_plugin():
    import numpy as np
    from rpy2.robjects import vectors as rv, r
    from rpy2.rinterface import NA_Logical, NA_Integer, NA_Real, NULLType

    eng = Engine()

    # ---------- helpers ----------
    def _dims(x):
        rdim = r["dim"](x)
        if isinstance(rdim, NULLType):
            return None
        d = list(rdim)
        return d if len(d) else None

    def _reshape(arr, dims):
        return arr.reshape(dims, order="F") if dims else arr

    # ---------- Python -> R: NumPy scalars ----------
    @eng.register_py(np.bool_)
    def _(e, x): return rv.BoolSexpVector([bool(x)])

    @eng.register_py(np.integer)
    def _(e, x):
        xi = int(x)
        if -(2**31) <= xi < 2**31:
            return rv.IntSexpVector([xi])
        return rv.FloatSexpVector([float(xi)])

    @eng.register_py(np.floating)
    def _(e, x): return rv.FloatSexpVector([float(x)])

    @eng.register_py(np.complexfloating)
    def _(e, x): return rv.ComplexSexpVector([complex(x)])

    # ---------- Python -> R: np.ndarray ----------
    @eng.register_py(np.ndarray)
    def _(e, a: "np.ndarray"):
        k = a.dtype.kind
        vec = None

        if k in ("f",):  # float32/64 -> double
            vec = rv.FloatSexpVector(a.ravel(order="F").astype("float64", copy=False).tolist())

        elif k in ("i", "u"):  # signed/unsigned ints
            # If any value outside 32-bit range -> promote to double
            amin = a.min() if a.size else 0
            amax = a.max() if a.size else 0
            if (k == "i" and (amin < -(2**31) or amax >= 2**31)) or k == "u" or a.dtype.itemsize > 4:
                vec = rv.FloatSexpVector(a.ravel(order="F").astype("float64", copy=False).tolist())
            else:
                vec = rv.IntSexpVector(a.ravel(order="F").astype("int32", copy=False).tolist())

        elif k == "b":  # bool
            vec = rv.BoolSexpVector(a.ravel(order="F").tolist())

        elif k == "c":  # complex64/128
            vec = rv.ComplexSexpVector(a.ravel(order="F").astype(np.complex128, copy=False).tolist())

        # ---------- string / object -> character ----------
        elif k in ("U", "S", "O"):
            # U: unicode, S: bytes, O: Python objects (assume string-like)
            flat = []
            for v in a.ravel(order="F"):
                if v is None:
                    flat.append("")  # or NA handling here if you want
                else:
                    flat.append(str(v))
            vec = rv.StrSexpVector(flat)

        else:
            raise TypeError(f"Unsupported NumPy dtype kind {k} ({a.dtype})")

        if a.ndim <= 1:
            return vec
        return r["array"](vec, dim=rv.IntSexpVector(list(a.shape)))

    # ---------- R -> Python: numeric/bool vectors & arrays ----------
    @eng.register_r(rv.FloatSexpVector)
    def _(e, x):
        # Float: NA_real_ -> np.nan
        data = [np.nan if (v is NA_Real) else float(v) for v in x]
        arr = np.asarray(data, dtype=np.float64, order="F")
        dims = _dims(x)
        return _reshape(arr, dims)

    @eng.register_r(rv.IntSexpVector)
    def _(e, x):
        # If any NA -> float64 with np.nan, else int32
        has_na = any(v is NA_Integer for v in x)
        if has_na:
            data = [np.nan if (v is NA_Integer) else int(v) for v in x]
            arr = np.asarray(data, dtype=np.float64, order="F")
        else:
            arr = np.asarray(list(x), dtype=np.int32, order="F")
        return _reshape(arr, _dims(x))

    @eng.register_r(rv.BoolSexpVector)
    def _(e, x):
        # If any NA -> object array with None, else bool
        has_na = any(v is NA_Logical for v in x)
        if has_na:
            data = [None if (v is NA_Logical) else bool(v) for v in x]
            arr = np.asarray(data, dtype=object, order="F")
        else:
            arr = np.asarray(list(x), dtype=bool, order="F")
        return _reshape(arr, _dims(x))

    @eng.register_r(rv.ComplexSexpVector)
    def _(e, x):
        # R has no distinct complex NA singleton; NA_complex_ is just NA with type complex.
        data = [complex(v) for v in x]  # you'd need to decide how to handle NA here
        arr = np.asarray(data, dtype=np.complex128, order="F")
        return _reshape(arr, _dims(x))

    return eng