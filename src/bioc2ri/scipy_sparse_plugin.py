# --- scipy_sparse_plugin.py ---
from functools import cache

__author__ = "MaximilianNuber"
__copyright__ = "MaximilianNuber"
__license__ = "MIT"

@cache
def scipy_sparse_plugin():
    """Creates and returns an Engine with SciPy Sparse conversion rules.

    Facilitates conversion of sparse matrices between Python (SciPy) and R (Matrix package).

    Conversions:
    - scipy.sparse.csc_matrix -> R dgCMatrix (or lgCMatrix/ngCMatrix via import defaults)
    - scipy.sparse.csr_matrix -> R dgRMatrix (or lgRMatrix/ngRMatrix)
    - R s4 classes (dgCMatrix, lgCMatrix, ngCMatrix) -> scipy.sparse.csc_matrix
    - R s4 classes (dgRMatrix, lgRMatrix, ngRMatrix) -> scipy.sparse.csr_matrix

    Returns:
        Engine: An Engine instance with SciPy Sparse rules registered.
    """
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
        """Extracts dimensions from an S4 matrix object."""
        d = list(slot(s4, "Dim"))
        return int(d[0]), int(d[1])

    # --------- Python -> R ---------
    @eng.register_py(sp.csc_matrix)
    def _(e, x: "sp.csc_matrix"):
        """Converts SciPy CSC matrix to R sparseMatrix (column-compressed)."""
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
        """Converts SciPy CSR matrix to R sparseMatrix (row-compressed)."""
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
        """Converts R dgCMatrix (double, column-compressed) to SciPy CSC matrix."""
        i = np.asarray(list(slot(s4, "i")), dtype=np.int32)
        p = np.asarray(list(slot(s4, "p")), dtype=np.int32)
        x = np.asarray(list(slot(s4, "x")), dtype=np.float64)
        m, n = _dims(s4)
        return sp.csc_matrix((x, i, p), shape=(m, n))

    @eng.register_s4("lgCMatrix")
    def _(e, s4):
        """Converts R lgCMatrix (logical, column-compressed) to SciPy CSC matrix."""
        i = np.asarray(list(slot(s4, "i")), dtype=np.int32)
        p = np.asarray(list(slot(s4, "p")), dtype=np.int32)
        # logicals stored as TRUEs (no NAs here) -> just ones
        x = np.ones_like(i, dtype=bool)
        m, n = _dims(s4)
        return sp.csc_matrix((x.astype(bool), i, p), shape=(m, n), dtype=bool)

    @eng.register_s4("ngCMatrix")  # pattern ("n") class: implicit ones
    def _(e, s4):
        """Converts R ngCMatrix (pattern, column-compressed) to SciPy CSC matrix (float ones)."""
        i = np.asarray(list(slot(s4, "i")), dtype=np.int32)
        p = np.asarray(list(slot(s4, "p")), dtype=np.int32)
        x = np.ones_like(i, dtype=np.float64)
        m, n = _dims(s4)
        return sp.csc_matrix((x, i, p), shape=(m, n))

    # --------- R -> Python (R: row-compressed) ---------
    @eng.register_s4("dgRMatrix")
    def _(e, s4):
        """Converts R dgRMatrix (double, row-compressed) to SciPy CSR matrix."""
        j = np.asarray(list(slot(s4, "j")), dtype=np.int32)
        p = np.asarray(list(slot(s4, "p")), dtype=np.int32)
        x = np.asarray(list(slot(s4, "x")), dtype=np.float64)
        m, n = _dims(s4)
        return sp.csr_matrix((x, j, p), shape=(m, n))

    @eng.register_s4("lgRMatrix")
    def _(e, s4):
        """Converts R lgRMatrix (logical, row-compressed) to SciPy CSR matrix."""
        j = np.asarray(list(slot(s4, "j")), dtype=np.int32)
        p = np.asarray(list(slot(s4, "p")), dtype=np.int32)
        x = np.ones_like(j, dtype=bool)
        m, n = _dims(s4)
        return sp.csr_matrix((x, j, p), shape=(m, n), dtype=bool)

    @eng.register_s4("ngRMatrix")
    def _(e, s4):
        """Converts R ngRMatrix (pattern, row-compressed) to SciPy CSR matrix (float ones)."""
        j = np.asarray(list(slot(s4, "j")), dtype=np.int32)
        p = np.asarray(list(slot(s4, "p")), dtype=np.int32)
        x = np.ones_like(j, dtype=np.float64)
        m, n = _dims(s4)
        return sp.csr_matrix((x, j, p), shape=(m, n))

    return eng