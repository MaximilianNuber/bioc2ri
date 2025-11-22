# biocpy_plugin.py

from __future__ import annotations

from typing import Any

from .engine import Engine
from .numpy_plugin import numpy_plugin
from .rnames import get_rownames, set_rownames

__author__ = "MaximilianNuber"
__copyright__ = "MaximilianNuber"
__license__ = "MIT"


def biocpy_plugin() -> Engine:
    """
    Plugin for BiocPy / BiocUtils / S4Vectors:

    py -> R
    ------
    - BooleanList / IntegerList / FloatList / StringList
      -> R logical/integer/double/character vectors (None -> NA_*).

    - BiocFrame -> S4Vectors::DataFrame:
        * NumPy arrays via numpy_plugin.
        * BiocUtils atomic lists passed through as above.
        * Plain Python lists are inspected, wrapped into BiocUtils lists,
          then converted. No R lists for BiocFrame columns.

    R -> py
    ------
    - R logical/integer/double/character vectors
      -> BooleanList / IntegerList / FloatList / StringList
        (factor -> StringList via as.character).

    - S4Vectors::DataFrame / DFrame -> BiocFrame with BiocUtils lists as columns.
    """
    import numpy as np
    from collections.abc import Sequence as SeqABC
    from .engine import Engine
    from .numpy_plugin import numpy_plugin
    from .rnames import get_rownames, set_rownames

    from biocframe import BiocFrame
    from biocutils import (
        BooleanList,
        IntegerList,
        FloatList,
        StringList,
        is_list_of_type,
    )

    from rpy2.robjects import r, vectors as rv
    from rpy2.rinterface import (
        NA_Logical,
        NA_Integer,
        NA_Real,
        NA_Character,
    )
    from rpy2.robjects.packages import importr

    S4Vectors = importr("S4Vectors")
    RMethods  = importr("methods")
    
    # Grab the S4 rownames<- method once
    rownames_set = RMethods.getMethod("rownames<-", )
    

    eng = Engine()
    np_eng = numpy_plugin()  # internal NumPy engine for arrays

    # -------------------------------------------------
    # Helpers: NA-aware construction of R atomic types
    # -------------------------------------------------

    def _bool_seq_to_r(seq) -> rv.BoolSexpVector:
        return rv.BoolSexpVector(
            [NA_Logical if v is None else bool(v) for v in seq]
        )

    def _int_seq_to_r(seq) -> rv.IntSexpVector:
        out = []
        for v in seq:
            out.append(NA_Integer if v is None else int(v))
        return rv.IntSexpVector(out)

    def _float_seq_to_r(seq) -> rv.FloatSexpVector:
        out = []
        for v in seq:
            out.append(NA_Real if v is None else float(v))
        return rv.FloatSexpVector(out)

    def _str_seq_to_r(seq) -> rv.StrSexpVector:
        out = []
        for v in seq:
            out.append(NA_Character if v is None else str(v))
        return rv.StrSexpVector(out)

    # -------------------------------------------------
    # BiocUtils atomic lists -> R atomic vectors
    # -------------------------------------------------

    @eng.register_py(BooleanList)
    def _(e, x: BooleanList):
        return _bool_seq_to_r(x)

    @eng.register_py(IntegerList)
    def _(e, x: IntegerList):
        return _int_seq_to_r(x)

    @eng.register_py(FloatList)
    def _(e, x: FloatList):
        return _float_seq_to_r(x)

    @eng.register_py(StringList)
    def _(e, x: StringList):
        return _str_seq_to_r(x)

    # -------------------------------------------------
    # Upgrade plain Python lists to BiocUtils typed lists
    # -------------------------------------------------

    def _as_biocutils_list(col: SeqABC) -> BooleanList | IntegerList | FloatList | StringList:
        """
        Take a generic Python sequence (usually list) and return a typed
        BiocUtils list:
    
        - BooleanList  for all-bool / bool-or-None
        - IntegerList  for all-int / int-or-None
        - FloatList    for numeric mixtures with actual floats
        - StringList   for all-str / str-or-None
    
        Raises TypeError if the column is genuinely heterogeneous.
        """
        if isinstance(col, (str, bytes)):
            raise TypeError("String column should not be treated as a generic sequence.")
    
        # ignore Nones for type inference
        non_none = [v for v in col if v is not None]
    
        # all None -> punt to StringList for now (could be refined later)
        if not non_none:
            return StringList([None] * len(col))
    
        # Bool / bool-or-None
        if all(isinstance(v, bool) for v in non_none):
            return BooleanList(col)
    
        # Pure ints (no floats), possibly with None
        if all(isinstance(v, int) and not isinstance(v, bool) for v in non_none):
            return IntegerList(col)
    
        # Numeric mix with actual floats -> FloatList
        if all(isinstance(v, (int, float)) for v in non_none):
            return FloatList(col)
    
        # Strings / None
        if all(isinstance(v, str) for v in non_none):
            return StringList(col)
    
        raise TypeError(f"Cannot infer BiocUtils list type for heterogeneous column: {col!r}")


    # -------------------------------------------------
    # BiocFrame -> S4Vectors::DataFrame
    # -------------------------------------------------

    @eng.register_py(BiocFrame)
    def _(e, bf: BiocFrame):
        cols_r: dict[str, Any] = {}

        for name, col in bf._data.items():
            key = str(name)

            # 1) Already a BiocUtils typed list
            if isinstance(col, (BooleanList, IntegerList, FloatList, StringList)):
                cols_r[key] = e.py2r(col)
                continue

            # 2) NumPy arrays: use numpy_plugin
            if isinstance(col, np.ndarray):
                cols_r[key] = np_eng.py2r(col)
                continue

            # 3) Generic Python sequences (most common: plain list)
            if isinstance(col, SeqABC) and not isinstance(col, (str, bytes)):
                typed = _as_biocutils_list(col)
                cols_r[key] = e.py2r(typed)
                continue

            # 4) Everything else: treat as a single string value
            typed = StringList([str(col)])
            cols_r[key] = e.py2r(typed)

        r_df = S4Vectors.DataFrame(**cols_r)

        rn = bf.row_names
        if rn is not None:
            rn_vec = rv.StrSexpVector([str(x) for x in rn])
            # S4 generic call: rownames(r_df) <- rn_vec
            r_df = rownames_set(r_df, rn_vec)

        return r_df

    # -------------------------------------------------
    # R atomic vectors -> BiocUtils lists
    # -------------------------------------------------

    def _r_vec_to_py_list(x, coerce):
        """Use R is.na to detect NA, map NA -> None, and coerce value otherwise."""
        na_mask = list(r["is.na"](x))
        out = []
        for v, is_na in zip(list(x), na_mask):
            if is_na:
                out.append(None)
            else:
                out.append(coerce(v))
        return out

    @eng.register_r(rv.BoolSexpVector)
    def _(e, x):
        vals = _r_vec_to_py_list(x, bool)
        return BooleanList(vals)

    @eng.register_r(rv.StrSexpVector)
    def _(e, x):
        vals = _r_vec_to_py_list(x, str)
        return StringList(vals)

    @eng.register_r(rv.FloatSexpVector)
    def _(e, x):
        # plain numeric vector
        vals = _r_vec_to_py_list(x, float)
        return FloatList(vals)

    @eng.register_r(rv.IntSexpVector)
    def _(e, x):
        # Special-case factors: class(x) includes "factor"
        classes = list(r["class"](x))
        if "factor" in classes:
            as_char = r["as.character"](x)
            vals = _r_vec_to_py_list(as_char, str)
            return StringList(vals)
        # otherwise integer vector
        vals = _r_vec_to_py_list(x, int)
        return IntegerList(vals)

    # -------------------------------------------------
    # S4Vectors::DataFrame / DFrame -> BiocFrame
    # -------------------------------------------------

    @eng.register_s4("DataFrame")
    @eng.register_s4("DFrame")
    def _(e, x):
        # S4Vectors::DataFrame / DFrame
        names = list(x.names)
        cols_py: dict[str, Any] = {}
    
        col_get = r["[["]  # R's `[[` method, dispatches correctly for DataFrame
    
        for n in names:
            # equivalent to x[[n]] in R
            col_r = col_get(x, n)
            cols_py[str(n)] = e.r2py(col_r)
    
        rn = get_rownames(x)
        return BiocFrame(cols_py, row_names=rn)

    return eng