`bioc2ri` provides a modern, extensible conversion engine that allows Python objects and R objects to flow between each other safely, predictably, and efficiently.

Unlike `rpy2.localconverter`, which uses a fixed set of implicit conversions, `bioc2ri` gives you:

- A dispatch-based conversion engine

- Plugins for major ecosystems (NumPy, pandas, SciPy sparse, BiocPy)

- S4-aware dispatch for Bioconductor classes (Matrix, S4Vectors)

- Zero-copy handling where possible

- Transparent Python → R → Python roundtrips

You assemble an engine with:


```python
from bioc2ri.base_plugin import base_plugin
from bioc2ri.numpy_plugin import numpy_plugin
from bioc2ri.pandas_plugin import pandas_plugin

eng = base_plugin().with_plugins(
    numpy_plugin(),
    pandas_plugin(),
)

```

    /Users/maximiliannuber/miniconda3/envs/bioc_pyrtools_env/lib/python3.12/site-packages/rpy2/rinterface/__init__.py:1211: UserWarning: Environment variable "PWD" redefined by R and overriding existing variable. Current: "/Users/maximiliannuber/Documents/my_packages", R: "/Users/maximiliannuber/Documents/my_packages/pyrtools_conversions/bioc2ri"
      warnings.warn(
    /Users/maximiliannuber/miniconda3/envs/bioc_pyrtools_env/lib/python3.12/site-packages/rpy2/rinterface/__init__.py:1211: UserWarning: Environment variable "R_SESSION_TMPDIR" redefined by R and overriding existing variable. Current: "/var/folders/cn/cq8z8mrn35j9ytmctm_fcl580000gn/T//RtmpNHMFUD", R: "/var/folders/cn/cq8z8mrn35j9ytmctm_fcl580000gn/T//RtmpAqoWHN"
      warnings.warn(



```python
r_obj = eng.py2r([1, 2, 3])
back = eng.r2py(r_obj)
```

### Features

- Base conversions: Python scalars, lists, dicts ↔ R atomic vectors + lists

- NumPy arrays: dtype-safe, shape-preserving, Fortran-order semantics

- pandas DataFrame: roundtrip with R data.frame

- SciPy sparse matrices: exact structural mapping to R dgCMatrix, dgRMatrix, lgCMatrix, …

- BiocPy + BiocUtils: BiocFrame, BooleanList, IntegerList, FloatList, StringList ↔ R S4Vectors::DataFrame

All conversions are explicit, debuggable, and easy to extend.

### Installation

#### Option 1: User-provided R

If R is installed, the `R_HOME` variable must be set in PATH, and we can:
```{bash}
pip install bioc2ri
```
Then in Python:
```{python}
from bioc2ri.install import ensure_core_bioc2ri_packages
ensure_core_bioc2ri_packages()  # installs Matrix + S4Vectors + BiocManager
```

#### Option 2: Conda-managed R (recommended)

We provide environment YAMLs that include:

- r-base

- Matrix

- S4Vectors

- BiocManager

- rpy2

```{bash}
conda env create -f environment.yml
conda activate bioc2ri
```
Conveniently, conda-installed `r-base` automatically sets the environment variable for only this environment.

## Using the conversion engine

The core idea of `bioc2ri` is simple:

1. You build an **Engine** by combining one or more **plugins**.
2. You use that engine’s `py2r` and `r2py` methods to move data between Python and R.
3. For the README (and for interactive exploration), we provide a few **demo helpers** in
   `bioc2ri.readme_functions` that pretty-print roundtrips.


```python
from bioc2ri.base_plugin import base_plugin
from bioc2ri.numpy_plugin import numpy_plugin
from bioc2ri.pandas_plugin import pandas_plugin
from bioc2ri.scipy_sparse_plugin import scipy_sparse_plugin
from bioc2ri.biocpy_plugin import biocpy_plugin

from bioc2ri import readme_functions as demo  # helper functions for this section
```


```python
base_eng = base_plugin()

demo.show_roundtrip(42)
demo.show_roundtrip([1, 2, None, 4])
demo.show_roundtrip({"a": 1, "b": 2.5, "label": "test"})
```

    Python → R → Python roundtrip
    ----------------------------------------
    Python in :
    42
      type    : int
    
    R object  :
      repr    : <rpy2.rinterface.IntSexpVector object at 0x11bbbc750> [13]
      R class : ['integer']
    
    Python out:
    42
      type    : int
    ======================================== 
    
    Python → R → Python roundtrip
    ----------------------------------------
    Python in :
    [1, 2, None, 4]
      type    : list
    
    R object  :
      repr    : <rpy2.rinterface.ListSexpVector object at 0x11bbbd490> [19]
      R class : ['list']
    
    Python out:
    [1, 2, <rpy2.rinterface_lib.sexp.NULLType object at 0x1054c9590> [0], 4]
      type    : list
    ======================================== 
    
    Python → R → Python roundtrip
    ----------------------------------------
    Python in :
    {'a': 1, 'b': 2.5, 'label': 'test'}
      type    : dict
    
    R object  :
      repr    : <rpy2.robjects.vectors.ListVector object at 0x106ccbfd0> [19]
    R classes: ('list',)
    [IntSexpVector, FloatSexpVector, StrSexpVector]
      a: <class 'rpy2.rinterface.IntSexpVector'>
      <rpy2.rinterface.IntSexpVector object at 0x11bb60d50> [13]
      b: <class 'rpy2.rinterface.FloatSexpVector'>
      <rpy2.rinterface.FloatSexpVector object at 0x104f66c10> [14]
      label: <class 'rpy2.rinterface_lib.sexp.StrSexpVector'>
      <rpy2.rinterface_lib.sexp.StrSexpVector object at 0x11bb60d50> [16]
      R class : ['list']
    
    Python out:
    {'a': 1, 'b': 2.5, 'label': 'test'}
      type    : dict
    ======================================== 
    



```python
np_eng = base_plugin().with_plugins(
    numpy_plugin(),
)

import numpy as np

vec = np.array([1.0, 2.5, 3.25], dtype=np.float32)
mat = np.arange(12, dtype=np.float64).reshape(3, 4)

demo.show_ndarray_roundtrip(vec)
demo.show_ndarray_roundtrip(mat)
```

    NumPy ndarray → R → NumPy ndarray
    ----------------------------------------
    Python in :
      value   :
    [1.   2.5  3.25]
      dtype   : float32
      shape   : (3,)
    
    R object  :
      repr    : <rpy2.rinterface.FloatSexpVector object at 0x11bbd6110> [14]
      R class : ['numeric']
      dim     : None
    
    Python out:
      value   :
    [1.   2.5  3.25]
      dtype   : float64
      shape   : (3,)
    ======================================== 
    
    NumPy ndarray → R → NumPy ndarray
    ----------------------------------------
    Python in :
      value   :
    [[ 0.  1.  2.  3.]
     [ 4.  5.  6.  7.]
     [ 8.  9. 10. 11.]]
      dtype   : float64
      shape   : (3, 4)
    
    R object  :
      repr    : <rpy2.robjects.vectors.FloatMatrix object at 0x10fc36390> [14]
    R classes: ('matrix', 'array')
    [0.000000, 4.000000, 8.000000, 1.000000, ..., 10.000000, 3.000000, 7.000000, 11.000000]
      R class : ['matrix', 'array']
      dim     : [3, 4]
    
    Python out:
      value   :
    [[ 0.  1.  2.  3.]
     [ 4.  5.  6.  7.]
     [ 8.  9. 10. 11.]]
      dtype   : float64
      shape   : (3, 4)
    ======================================== 
    



```python
import pandas as pd

pd_eng = base_plugin().with_plugins(
    numpy_plugin(),
    pandas_plugin(),
)

df = pd.DataFrame(
    {
        "sample": ["s1", "s2", "s3", "s4"],
        "age": [34, 51, 29, 40],
        "is_case": [True, True, False, False],
    }
).set_index("sample")  # index becomes rownames in R

demo.show_dataframe_roundtrip(df)
```

    pandas.DataFrame → R data.frame → pandas.DataFrame
    ----------------------------------------
    Python in :
      shape   : (4, 2)
      dtypes  :
    age        int64
    is_case     bool
    dtype: object
      data    :
            age  is_case
    sample              
    s1       34     True
    s2       51     True
    s3       29    False
    s4       40    False
    
    R object  :
      repr    : <rpy2.robjects.vectors.DataFrame object at 0x11b0a9610> [19]
    R classes: ('data.frame',)
    [FloatSexpVector, BoolSexpVector]
      age: <class 'rpy2.rinterface.FloatSexpVector'>
      <rpy2.rinterface.FloatSexpVector object at 0x106ce5090> [14]
      is_case: <class 'rpy2.rinterface.BoolSexpVector'>
      <rpy2.rinterface.BoolSexpVector object at 0x11bbbd6d0> [10]
      R class : ['data.frame']
      rownames: ['s1', 's2', 's3', 's4']
    
    Python out (eng.r2py):
      type    : <class 'pandas.core.frame.DataFrame'>
      shape   : (4, 2)
      dtypes  :
    age        float64
    is_case       bool
    dtype: object
      data    :
         age  is_case
    s1  34.0     True
    s2  51.0     True
    s3  29.0    False
    s4  40.0    False
    ======================================== 
    



```python
import scipy.sparse as sp
import numpy as np

sparse_eng = base_plugin().with_plugins(
    numpy_plugin(),          # optional, but often useful together
    scipy_sparse_plugin(),
)

dense = np.array(
    [
        [0.0, 1.0, 0.0, 2.0],
        [3.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 4.0, 0.0],
    ],
    dtype=np.float64,
)

csc = sp.csc_matrix(dense)
csr = sp.csr_matrix(dense)

demo.show_sparse_roundtrip(csc, label="csc / dgCMatrix")
demo.show_sparse_roundtrip(csr, label="csr / dgRMatrix")
```

    SciPy sparse → R Matrix → SciPy sparse
    ------------------------------------------------------------
    Label      : csc / dgCMatrix
    Python in  :
      type     : <class 'scipy.sparse._csc.csc_matrix'>
      format   : csc
      shape    : (3, 4)
      nnz      : 4
      dtype    : float64
    
    R object   :
      repr     : <rpy2.robjects.methods.RS4 object at 0x12814b1d0> [25]
    R classes: ('dgCMatrix',)
      R class  : ['dgCMatrix']
      Dim      : (3, 4)
    
    Python out (eng.r2py):
      type     : <class 'scipy.sparse._csc.csc_matrix'>
      format   : csc
      shape    : (3, 4)
      nnz      : 4
      dtype    : float64
      same shape?: True
      same nnz?  : True
    ============================================================ 
    
    SciPy sparse → R Matrix → SciPy sparse
    ------------------------------------------------------------
    Label      : csr / dgRMatrix
    Python in  :
      type     : <class 'scipy.sparse._csr.csr_matrix'>
      format   : csr
      shape    : (3, 4)
      nnz      : 4
      dtype    : float64
    
    R object   :
      repr     : <rpy2.robjects.methods.RS4 object at 0x11bc16310> [25]
    R classes: ('dgRMatrix',)
      R class  : ['dgRMatrix']
      Dim      : (3, 4)
    
    Python out (eng.r2py):
      type     : <class 'scipy.sparse._csr.csr_matrix'>
      format   : csr
      shape    : (3, 4)
      nnz      : 4
      dtype    : float64
      same shape?: True
      same nnz?  : True
    ============================================================ 
    



```python
from biocframe import BiocFrame
from biocutils import BooleanList, IntegerList, FloatList, StringList

bioc_eng = base_plugin().with_plugins(
    biocpy_plugin(),
)

bf = BiocFrame(
    {
        "age": IntegerList([34, 51, 29, 40]),
        "is_case": BooleanList([True, True, False, False]),
        "score": FloatList([1.2, None, 3.4, 2.1]),
        "label": StringList(["A", "B", "A", "B"]),
    },
    row_names=["sample1", "sample2", "sample3", "sample4"],
)

demo.show_biocframe_roundtrip(bf, label="BiocFrame with BiocUtils lists")
```

    BiocFrame → S4Vectors::DataFrame → BiocFrame
    ------------------------------------------------------------
    Label         : BiocFrame with BiocUtils lists
    Python in (BiocFrame):
      nrow        : 4
      ncol        : 4
      row_names   : ['sample1', 'sample2', 'sample3', 'sample4']
      columns     :
        'age': type=IntegerList, len=4
        'is_case': type=BooleanList, len=4
        'score': type=FloatList, len=4
        'label': type=StringList, len=4
    
    R object (S4Vectors::DataFrame / DFrame):
      repr        : <rpy2.robjects.methods.RS4 object at 0x16e192050> [25]
    R classes: ('DFrame',)
      R class     : ['DFrame']
      rownames    : ['sample1', 'sample2', 'sample3', 'sample4']
    
    Python out (eng.r2py):
      type        : <class 'biocframe.BiocFrame.BiocFrame'>
      nrow        : 4
      ncol        : 4
      row_names   : ['sample1', 'sample2', 'sample3', 'sample4']
      columns     :
        'age': type=IntegerList, len=4
        'is_case': type=BooleanList, len=4
        'score': type=FloatList, len=4
        'label': type=StringList, len=4
    ============================================================ 
    


Here you’ll see:

- a `BiocFrame` becomes an `S4Vectors::DataFrame/DFrame` in R,

- columns are represented as atomic vectors (driven by BiocUtils list types),

- row names are preserved,

and the object roundtrips back to a BiocFrame with BiocUtils columns.

In normal code, you’ll call `eng.py2r` and `eng.r2py` directly.
In this README (and in your exploratory notebooks), the `bioc2ri.readme_functions` helpers simply show exactly what each engine is doing to your data.

## Working with R names and dimensions (`rnames` utilities)

`bioc2ri` includes a small set of helpers around **R names**, **rownames**, **colnames**, and basic metadata.  
These functions are thin wrappers around R’s own `rownames()`, `colnames()`, `names()`, `dim()`, etc., but:

- They speak **Python types** (`list[str]`, `None`, etc.).
- They work **both** on R objects (via rpy2) **and** on pandas DataFrames (where it makes sense).
- They centralize common checks like length mismatches.

### Row and column names

The main entry points are:

- `get_rownames(x)`
- `get_colnames(x)`
- `set_rownames(x, names, strict_len=False)`
- `set_colnames(x, names, strict_len=False)`

They accept either:

- an R object (`Sexp` / `NULLType`), or  
- a `pandas.DataFrame` (for convenience).

#### Getting row and column names

```python
from rpy2.robjects import r
from bioc2ri.rnames import get_rownames, get_colnames

# Example: R matrix
mat = r.matrix(r.c(1, 2, 3, 4), nrow=2, ncol=2)
mat = set_rownames(mat, eng.py2r(np.asarray(["r1", "r2"])))
mat = set_colnames(mat, eng.py2r(np.asarray(["c1", "c2"])))

get_rownames(mat)  # ['r1', 'r2']
get_colnames(mat)  # ['c1', 'c2']


from bioc2ri.rnames import set_rownames, set_colnames

# R object: rownames(x) <- value; returns modified object
mat = set_rownames(mat, ["row1", "row2"])
mat = set_colnames(mat, ["col1", "col2"])

# Pandas: sets DataFrame.index / DataFrame.columns in-place and returns df
df = set_rownames(df, ["i1", "i2"])
df = set_colnames(df, ["A", "B"])```

```

### Small R-object introspection helpers

The `bioc2ri.rutils` module also provides a few tiny utilities for inspecting raw R objects:

- `is_r(obj)` — is this an rpy2 R object?

- `r_class(x)` — tuple of R class names (class(x) in R).

- `r_len(x)` — length(x) as an int.

- `r_dim(x)` — dim(x) as a Python tuple[int, ...] or None.

- `r_names(x)` — names(x) as list[str] or None.

- `r_print(x)` — call R’s print(x).

- `r_str(x)´ — call R’s str(x).

- `r_summary(x)` — call print(summary(x)).


```python
from rpy2.robjects import r
from bioc2ri.rnames import set_rownames, set_colnames
from bioc2ri.rutils import r_class, r_len, r_dim, r_names, r_print, r_str

x = r("matrix(1:6, nrow = 2, ncol = 3)")
x = set_rownames(x, eng.py2r(np.asarray(["r1", "r2"])))
x = set_colnames(x, eng.py2r(np.asarray(["c1", "c2", "c3"])))

r_class(x)  # ('matrix', 'array')
r_len(x)    # 6
r_dim(x)    # (2, 3)
r_names(x)  # None (names(x) for matrices is often NULL)

r_print(x)  # calls R's print
r_str(x)    # calls R's str(x) for a quick structural summary
```

       c1 c2 c3
    r1  1  3  5
    r2  2  4  6
     int [1:2, 1:3] 1 2 3 4 5 6
     - attr(*, "dimnames")=List of 2
      ..$ : chr [1:2] "r1" "r2"
      ..$ : chr [1:3] "c1" "c2" "c3"


These helpers are intentionally tiny, but they make it much easier to write debugging code and consistent conversion logic without sprinkling raw `r["rownames"]`, `r["names"]`, `dim()`, etc. all over your codebase.

## Note

This project has been set up using [BiocSetup](https://github.com/biocpy/biocsetup)
and [PyScaffold](https://pyscaffold.org/).


```python

```
