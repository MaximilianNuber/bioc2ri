# --- scipy_sparse_plugin.py ---
from functools import cache

@cache
def scipy_sparse_plugin():
    import numpy as np
    import scipy.sparse as sp
    from rpy2.robjects import vectors as rv, r
    from rpy2.robjects.packages import importr
    from .engine import Engine

    Matrix = importr("Matrix")  # requires R package Matrix
    eng = Engine()

    # --------- helpers ---------
    slot = r["slot"]  # slot(x, "name")
    def _dims(s4):
        d = list(slot(s4, "Dim"))
        return int(d[0]), int(d[1])

    # --------- Python -> R ---------
    @eng.register_py(sp.csc_matrix)
    def _(e, x: "sp.csc_matrix"):
        i = rv.IntSexpVector(x.indices.astype("int32", copy=False))
        p = rv.IntSexpVector(x.indptr.astype("int32", copy=False))
        dims = rv.IntSexpVector([int(x.shape[0]), int(x.shape[1])])

        if x.dtype == np.bool_:
            xv = rv.BoolSexpVector(x.data.astype(bool, copy=False))
        else:
            xv = rv.FloatSexpVector(x.data.astype("float64", copy=False))

        # index1=False => 0-based (matches SciPy)
        return Matrix.sparseMatrix(i=i, p=p, x=xv, dims=dims, index1=False, repr="C")

    @eng.register_py(sp.csr_matrix)
    def _(e, x: "sp.csr_matrix"):
        j = rv.IntSexpVector(x.indices.astype("int32", copy=False))
        p = rv.IntSexpVector(x.indptr.astype("int32", copy=False))
        dims = rv.IntSexpVector([int(x.shape[0]), int(x.shape[1])])

        if x.dtype == np.bool_:
            xv = rv.BoolSexpVector(x.data.astype(bool, copy=False))
        else:
            xv = rv.FloatSexpVector(x.data.astype("float64", copy=False))

        # Build row-compressed directly (dgR*/lgR*)
        return Matrix.sparseMatrix(j=j, p=p, x=xv, dims=dims, index1=False, repr="R")

    # --------- R -> Python (C: column-compressed) ---------
    @eng.register_s4("dgCMatrix")
    def _(e, s4):
        i = np.asarray(list(slot(s4, "i")), dtype=np.int32)
        p = np.asarray(list(slot(s4, "p")), dtype=np.int32)
        x = np.asarray(list(slot(s4, "x")), dtype=np.float64)
        m, n = _dims(s4)
        return sp.csc_matrix((x, i, p), shape=(m, n))

    @eng.register_s4("lgCMatrix")
    def _(e, s4):
        i = np.asarray(list(slot(s4, "i")), dtype=np.int32)
        p = np.asarray(list(slot(s4, "p")), dtype=np.int32)
        # logicals stored as TRUEs (no NAs here) -> just ones
        x = np.ones_like(i, dtype=bool)
        m, n = _dims(s4)
        return sp.csc_matrix((x.astype(bool), i, p), shape=(m, n), dtype=bool)

    @eng.register_s4("ngCMatrix")  # pattern ("n") class: implicit ones
    def _(e, s4):
        i = np.asarray(list(slot(s4, "i")), dtype=np.int32)
        p = np.asarray(list(slot(s4, "p")), dtype=np.int32)
        x = np.ones_like(i, dtype=np.float64)
        m, n = _dims(s4)
        return sp.csc_matrix((x, i, p), shape=(m, n))

    # --------- R -> Python (R: row-compressed) ---------
    @eng.register_s4("dgRMatrix")
    def _(e, s4):
        j = np.asarray(list(slot(s4, "j")), dtype=np.int32)
        p = np.asarray(list(slot(s4, "p")), dtype=np.int32)
        x = np.asarray(list(slot(s4, "x")), dtype=np.float64)
        m, n = _dims(s4)
        return sp.csr_matrix((x, j, p), shape=(m, n))

    @eng.register_s4("lgRMatrix")
    def _(e, s4):
        j = np.asarray(list(slot(s4, "j")), dtype=np.int32)
        p = np.asarray(list(slot(s4, "p")), dtype=np.int32)
        x = np.ones_like(j, dtype=bool)
        m, n = _dims(s4)
        return sp.csr_matrix((x, j, p), shape=(m, n), dtype=bool)

    @eng.register_s4("ngRMatrix")
    def _(e, s4):
        j = np.asarray(list(slot(s4, "j")), dtype=np.int32)
        p = np.asarray(list(slot(s4, "p")), dtype=np.int32)
        x = np.ones_like(j, dtype=np.float64)
        m, n = _dims(s4)
        return sp.csr_matrix((x, j, p), shape=(m, n))

    return eng