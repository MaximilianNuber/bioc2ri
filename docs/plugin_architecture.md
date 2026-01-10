# Plugin Architecture in `bioc2ri`

This document explains the architecture of the conversion engine and the role of the default plugins provided with `bioc2ri`.

## 1. The Core Engine (`engine.py`)

The heart of the system is the `Engine` class defined in `src/bioc2ri/engine.py`. It manages the conversion rules between Python objects and R objects (via `rpy2`).

### Mechanisms
- **Registry**: The `Engine` maintains three registries:
    - `py_rules`: A list of rules for converting Python objects to R (`py2r`).
    - `r_rules`: A list of rules for converting R objects to Python (`r2py`).
    - `s4_rules`: A dictionary mapping R S4 class names to conversion functions.
- **Dispatch**:
    - `py2r(x)`: Iterates through `py_rules` and applies the first rule where `isinstance(x, type)` is true.
    - `r2py(x)`: 
        - First checks if `x` is an S4 object (`SexpS4`). If so, it looks up the object's R class in `s4_rules`.
        - If not S4 (or no S4 rule is found), it iterates through `r_rules` and applies the first match.
- **Composition**: The `with_plugins(*plugins)` method allows combining multiple engines. Rules from the new plugins are prepended to the existing rules, meaning **plugins take precedence** over the base rules.

## 2. Default Plugins

`bioc2ri` comes with three main plugins that build upon each other.

### A. `base_plugin` (`base_plugin.py`)
This provides the fundamental conversions for Python built-in types.

**Python -> R:**
- **Scalars**: `int`, `float`, `bool`, `str`, `complex`, `bytes` → R atomic vectors of length 1.
    - *Note*: Integers outside the 32-bit range are promoted to doubles.
    - `None` → `NULL`.
- **Collections**: `list`, `tuple` → R `list`.
- **Dictionaries**: `dict` → R named `list`.

**R -> Python:**
- **Atomic Vectors**: Converted to Python scalars (if length 1) or Python lists.
    - `NA` values are converted to `None`.
- **Lists**: Converted to Python lists or dictionaries (if the R list has names).

### B. `numpy_plugin` (`numpy_plugin.py`)
Matches conversions between NumPy arrays and R arrays/vectors.

**Python -> R:**
- **Arrays**: `np.ndarray` → R vector (1D) or R array (nD).
    - `float` → `double`
    - `int` → `integer` (or `double` if overflow)
    - `bool` → `logical`
    - `complex` → `complex`
    - `str`/`object` → `character`
    - *Note*: Uses column-major (Fortran) order for flattening to match R's memory layout.

**R -> Python:**
- **Vectors/Arrays**: R atomic vectors → `np.ndarray`.
    - Handles `NA` conversion:
        - Float `NA` → `np.nan`
        - Integer `NA` → Promotes array to float and uses `np.nan`.
        - Logical `NA` → Promotes array to object and uses `None`.
    - Reshapes the output to match R dimensions (using column-major order).

### C. `biocpy_plugin` (`biocpy_plugin.py`)
Provides integration with [BiocPy](https://github.com/biocpy) ecosystem types (`biocutils`, `biocframe`).

**Python -> R:**
- **BiocUtils Lists**: `BooleanList`, `IntegerList`, `FloatList`, `StringList` → R atomic vectors (preserving `NA` via `None`).
- **BiocFrame** → `S4Vectors::DataFrame` (or `DFrame`).
    - Columns are converted recursively:
        - `BiocUtils` lists → R vectors.
        - `np.ndarray` → R arrays/vectors (via `numpy_plugin`).
        - Plain Python lists → Inspected and upgraded to typed `BiocUtils` lists if possible, then converted.
    - Row names are preserved.

**R -> Python:**
- **Atomic Vectors**: Converted to specific `BiocUtils` lists (e.g., R integer vector → `IntegerList`).
    - *Note*: R factors are converted to `StringList`.
- **S4 DataFrames**: `S4Vectors::DataFrame` / `DFrame` → `BiocFrame`.
    - Columns are extracted and converted back to Python (recursively invoking the vector -> BiocUtils list rules).
    - Row names are preserved.

## 3. How to Use

Typically, you don't instantiate these plugins manually. The main `rpy2_default_conversions` or `lazy_r_env` module likely composes these into a single usable engine.

To create a new plugin:
1. Define a function that returns an `Engine` instance.
2. Use `@eng.register_py(Type)` or `@eng.register_r(Type)` to define conversion rules.
3. Combine it with the existing engine using `base_engine.with_plugins(my_plugin())`.
