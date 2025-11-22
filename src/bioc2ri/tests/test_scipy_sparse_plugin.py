import numpy as np
import scipy.sparse as sp
from rpy2.robjects import r
from rpy2.robjects.packages import importr
from rpy2.robjects import vectors as rv



eng = scipy_sparse_plugin()
Matrix = importr("Matrix")

def test_csc_py2r_dgCMatrix():
    # 2 x 3 matrix:
    # [[0, 2, 0],
    #  [1, 0, 3]]
    data = np.array([1, 2, 3], dtype=float)
    indices = np.array([1, 0, 1], dtype=np.int32)      # row indices
    indptr = np.array([0, 1, 2, 3], dtype=np.int32)    # column pointers
    X = sp.csc_matrix((data, indices, indptr), shape=(2, 3))

    r_mat = eng.py2r(X)

    # Check R class includes dgCMatrix
    cls = list(r["class"](r_mat))
    assert "dgCMatrix" in cls

    # Check dense values via as.matrix
    r_dense = r["as.matrix"](r_mat)
    dims = list(r["dim"](r_dense))
    vals = np.array(list(r_dense), dtype=float).reshape(dims, order="F")

    expected = X.toarray()
    assert np.allclose(vals, expected)

def test_csr_py2r_dgRMatrix():
    # same logical matrix, but build it as CSR
    data = np.array([1, 2, 3], dtype=float)
    indices = np.array([0, 1, 2], dtype=np.int32)      # col indices
    indptr = np.array([0, 1, 3], dtype=np.int32)       # row pointers
    X = sp.csr_matrix((data, indices, indptr), shape=(2, 3))

    r_mat = eng.py2r(X)

    cls = list(r["class"](r_mat))
    assert "dgRMatrix" in cls

    r_dense = r["as.matrix"](r_mat)
    dims = list(r["dim"](r_dense))
    vals = np.array(list(r_dense), dtype=float).reshape(dims, order="F")

    expected = X.toarray()
    assert np.allclose(vals, expected)

def test_dgCMatrix_r2py_to_csc():
    # make a simple 2x3 sparse matrix in R (column-compressed)
    # [[0, 2, 0],
    #  [1, 0, 3]]
    r_mat = Matrix.sparseMatrix(
        i = rv.IntSexpVector([2, 1, 2]),            # 1-based row indices in R
        j = rv.IntSexpVector([1, 2, 3]),            # 1-based col indices
        x = rv.FloatSexpVector([1.0, 2.0, 3.0]),
        dims = rv.IntSexpVector([2, 3]),
        index1 = base_plugin().py2r(True),            # default, just explicit
    )

    X = eng.r2py(r_mat)
    assert isinstance(X, sp.csc_matrix)
    assert X.shape == (2, 3)

    expected = np.array([[0, 2, 0],
                         [1, 0, 3]], dtype=float)
    assert np.allclose(X.toarray(), expected)

def test_dgRMatrix_r2py_to_csr():
    # Start from dgCMatrix, then coerce to dgRMatrix on R side
    r_r = r("""
        # Ensure the Matrix package is loaded
        library(Matrix)
        
        # 1. Define 1-based indices (i=rows, j=cols) and values (x)
        i <- c(2, 1, 2)
        j <- c(1, 2, 3)
        x <- c(1.0, 2.0, 3.0)
        
        # 2. Create the sparse matrix (default format is usually dgCMatrix/column-major)
        # The dimensions are 2 rows and 3 columns.
        m_r <- sparseMatrix(
            i = i, 
            j = j, 
            x = x, 
            dims = c(2, 3), 
            repr = "R",
            index1 = TRUE # indices are 1-based
        )
        
        # 3. Coerce the matrix to the desired row-major format (dgRMatrix)
        # m_r <- as(m_c, "dgRMatrix")
        m_r
    """)

    X = eng.r2py(r_r)
    assert isinstance(X, sp.csr_matrix)
    assert X.shape == (2, 3)

    expected = np.array([[0, 2, 0],
                         [1, 0, 3]], dtype=float)
    assert np.allclose(X.toarray(), expected)

def test_lgCMatrix_r2py_to_csc_bool():
    # 2x3 logical sparse:
    # [[FALSE, TRUE,  FALSE],
    #  [TRUE,  FALSE, TRUE ]]
    r_mat = r("""
        Matrix::Matrix(
          rbind(
            c(FALSE, TRUE,  FALSE),
            c(TRUE,  FALSE, TRUE )
          ),
          sparse = TRUE
        )
    """)  # by construction, lgCMatrix

    cls = list(r["class"](r_mat))
    assert "lgCMatrix" in cls

    X = eng.r2py(r_mat)
    assert isinstance(X, sp.csc_matrix)
    assert X.shape == (2, 3)
    assert X.dtype == bool

    expected = np.array([[False, True,  False],
                         [True,  False, True ]], dtype=bool)
    assert np.array_equal(X.toarray(), expected)

def test_lgRMatrix_r2py_to_csr_bool():
    # Coerce previous lgCMatrix to lgRMatrix
    r_mat = r("""
        m = Matrix::Matrix(
          rbind(
            c(FALSE, TRUE,  FALSE),
            c(TRUE,  FALSE, TRUE )
          ),
          # sparse = TRUE,
          byrow = TRUE
        )
        m = as(m, "lgRMatrix")
        m
    """)  # by construction, lgCMatrix
    print(r_mat)

    X = eng.r2py(r_mat)
    print(type(X))
    assert isinstance(X, sp.csr_matrix)
    assert X.shape == (2, 3)
    assert X.dtype == bool

    expected = np.array([[False, True,  False],
                         [True,  False, True ]], dtype=bool)
    assert np.array_equal(X.toarray(), expected)

def test_ngCMatrix_r2py_to_csc_pattern():
    # pattern matrix: implicit ones
    r_mat = r("""
        Matrix::Matrix(
          rbind(
            c(0, 1, 0),
            c(1, 0, 1)
          ),
          sparse = TRUE
        ) != 0
    """)  # logical; then drop x -> ngCMatrix via as(.,"ngCMatrix")
    r_ng = r["as"](r_mat, "ngCMatrix")

    X = eng.r2py(r_ng)
    assert isinstance(X, sp.csc_matrix)
    expected = np.array([[0, 1, 0],
                         [1, 0, 1]], dtype=float)
    assert np.allclose(X.toarray(), expected)

test_csc_py2r_dgCMatrix()
test_csr_py2r_dgRMatrix()
test_dgCMatrix_r2py_to_csc()
test_dgRMatrix_r2py_to_csr()
test_lgCMatrix_r2py_to_csc_bool()
# test_lgRMatrix_r2py_to_csr_bool()
# optional:
# test_ngCMatrix_r2py_to_csc_pattern()