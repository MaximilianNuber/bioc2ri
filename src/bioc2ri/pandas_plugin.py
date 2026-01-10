# pandas_plugin.py
from functools import cache 

__author__ = "MaximilianNuber"
__copyright__ = "MaximilianNuber"
__license__ = "MIT"

@cache
def pandas_plugin():
    """Creates and returns an Engine with Pandas conversion rules.

    Handles conversion between Pandas types (Series, DataFrame) and R types (vectors, data.frames).
    Delegates element-wise conversion to the NumPy engine.

    Features:
    - pd.Series -> R vector (or factor if categorical)
    - pd.DataFrame -> R data.frame
    - R data.frame -> pd.DataFrame
    - Handling of row names/indexing where appropriate.

    Returns:
        Engine: An Engine instance with Pandas rules registered.
    """
    import numpy as np
    import pandas as pd
    from pandas import CategoricalDtype

    from rpy2.robjects import r, baseenv, vectors as rv
    from rpy2.robjects.vectors import DataFrame as RDataFrame

    from .engine import Engine
    from .rnames import get_rownames, set_rownames
    from .numpy_plugin import numpy_plugin  # internal

    eng = Engine()
    np_eng = numpy_plugin()  # private NumPy engine

    df_ctor   = baseenv["data.frame"]
    factor_fn = baseenv["factor"]

    # ---------- helpers ----------

    # ---------- helpers ----------

    def _series_to_factor(s: "pd.Series"):
        """Converts a categorical Pandas Series to an R factor."""
        cat = s.astype("string")
        vals = [None if pd.isna(v) else str(v) for v in cat]
        vec = rv.StrSexpVector(vals)
        levels = [str(l) for l in s.cat.categories]
        r_levels = rv.StrSexpVector(levels)
        ordered = bool(getattr(s.cat, "ordered", False))
        return factor_fn(vec, levels=r_levels, ordered=ordered)

    # ---------- Python -> R: Series ----------

    @eng.register_py(pd.Series)
    def _(e, s: "pd.Series"):
        """Converts Pandas Series to R vector or factor.

        If the Series is categorical, returns an R factor.
        Otherwise, converts to NumPy array and delegates to the internal NumPy engine.
        """
        if isinstance(s.dtype, CategoricalDtype):
            return _series_to_factor(s)
        arr = s.to_numpy(copy=False)
        return np_eng.py2r(arr)   # use NumPy engine

    # ---------- Python -> R: DataFrame ----------

    @eng.register_py(pd.DataFrame)
    def _(e, df: "pd.DataFrame"):
        """Converts Pandas DataFrame to R data.frame.

        Recursively converts columns using registered rules.
        Preserves the index as R row names if present.
        """
        cols = {str(name): e.py2r(df[name]) for name in df.columns}
        r_df = df_ctor(**cols)
        if df.index is not None:
            r_df = set_rownames(r_df, df.index.tolist(), strict_len=False)
        return r_df

    # ---------- R -> Python: vectors via NumPy engine ----------

    @eng.register_r(rv.FloatSexpVector)
    def _(e, x):
        """Converts R numeric vector to NumPy/Pandas compatible type via NumPy engine."""
        return np_eng.r2py(x)

    @eng.register_r(rv.IntSexpVector)
    def _(e, x):
        """Converts R integer vector to NumPy/Pandas compatible type via NumPy engine."""
        return np_eng.r2py(x)

    @eng.register_r(rv.BoolSexpVector)
    def _(e, x):
        """Converts R logical vector to NumPy/Pandas compatible type via NumPy engine."""
        return np_eng.r2py(x)

    @eng.register_r(rv.StrSexpVector)
    def _(e, x):
        """Converts R character vector to NumPy/Pandas compatible type via NumPy engine."""
        return np_eng.r2py(x)

    # ---------- R -> Python: DataFrame -> pandas.DataFrame ----------

    @eng.register_r(RDataFrame)
    def _(e, x: RDataFrame):
        """Converts R data.frame to Pandas DataFrame.

        Columns are converted using the internal NumPy engine (r2py).
        Row names are preserved as the DataFrame index.
        """
        import pandas as pd
        cols = {}
        for name in list(x.names):
            col_r = x.rx2(name)
            cols[str(name)] = np_eng.r2py(col_r)
        df = pd.DataFrame(cols)
        rn = get_rownames(x)
        if rn is not None and len(rn) == len(df):
            df.index = rn
        return df

    return eng