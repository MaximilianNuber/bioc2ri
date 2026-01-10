# --- base_plugin.py --- 
from functools import cache 
from .engine import Engine

__author__ = "MaximilianNuber"
__copyright__ = "MaximilianNuber"
__license__ = "MIT"

@cache 
def base_plugin(): 
    """Creates and returns the base Engine with default conversion rules.
    
    This plugin handles fundamental Python types (int, float, str, bool, list, dict, set)
    and their R counterparts (integer, numeric, character, logical, list).
    
    Returns:
        Engine: An Engine instance with base rules registered.
    """
    from rpy2.robjects import vectors as rv 
    from rpy2.robjects import vectors as rv  # Note: duplicate import reserved for potential cleanup
    from rpy2.rinterface import NULL, NA_Logical, NA_Integer, NA_Real, NA_Character 
    import rpy2 
    
    eng = Engine() 
    
    
    # ---------- helpers ---------- 
    def _r_scalar_to_py(x): 
        """Map a single R atomic element to a Python scalar with NA->None.""" 
        # rpy2 vectors yield Python values when indexed 
        v = x[0] 
        if v is NA_Logical or v is NA_Integer or v is NA_Real or v is NA_Character: 
            return None 
        return v 
        
    def _r_vec_to_py_list(x): 
        out = [] 
        for v in x: 
            if v is NA_Logical or v is NA_Integer or v is NA_Real or v is NA_Character: 
                out.append(None) 
            else: 
                out.append(v) 
        return out 
        
    # ---------- Python -> R (scalars) ---------- 
        
    @eng.register_py(type(None)) 
    
    @eng.register_py(type(None)) 
    def _(e, x): 
        """Converts Python None to R NULL."""
        return NULL 
        
    @eng.register_py(bool) 
    @eng.register_py(bool) 
    def _(e, x): 
        """Converts Python bool to R logical vector."""
        return rv.BoolSexpVector([bool(x)]) 
        
    @eng.register_py(int) 
    @eng.register_py(int) 
    def _(e, x): 
        """Converts Python int to R integer or numeric (float) vector.
        
        Promotes to float if the integer is out of R's 32-bit integer range.
        """
        try: 
            if -(2**31) <= x < 2**31: 
                return rv.IntSexpVector([int(x)]) 
            else: 
                return rv.FloatSexpVector([float(x)]) 
        except Exception: 
            return rv.FloatSexpVector([float(x)]) 
            
    @eng.register_py(float) 
    @eng.register_py(float) 
    def _(e, x): 
        """Converts Python float to R numeric vector."""
        return rv.FloatSexpVector([float(x)]) 
        
    @eng.register_py(str) 
    @eng.register_py(str) 
    def _(e, x): 
        """Converts Python str to R character vector."""
        return rv.StrSexpVector([x]) 
        
    @eng.register_py(complex) 
    @eng.register_py(complex) 
    def _(e, x): 
        """Converts Python complex to R complex vector."""
        return rv.ComplexSexpVector([x]) 
        
    @eng.register_py(bytes) 
    @eng.register_py(bytes) 
    def _(e, x): 
        """Converts Python bytes to R raw vector."""
        return rv.ByteSexpVector(x) 
        
    # ---------- R -> Python (atomic vectors) ---------- 
    @eng.register_r(rv.BoolSexpVector) 
    def _(e, x): 
        """Converts R logical vector to Python bool or list of bools/None."""
        return _r_scalar_to_py(x) if len(x) == 1 else _r_vec_to_py_list(x) 
        
    @eng.register_r(rv.IntSexpVector) 
    @eng.register_r(rv.IntSexpVector) 
    def _(e, x): 
        """Converts R integer vector to Python int or list of ints/None."""
        new = _r_scalar_to_py(x) if len(x) == 1 else _r_vec_to_py_list(x) 
        return new 
        
    @eng.register_r(rv.FloatSexpVector) 
    @eng.register_r(rv.FloatSexpVector) 
    def _(e, x): 
        """Converts R numeric vector to Python float or list of floats/None."""
        return _r_scalar_to_py(x) if len(x) == 1 else _r_vec_to_py_list(x) 
        
    @eng.register_r(rv.StrSexpVector) 
    @eng.register_r(rv.StrSexpVector) 
    def _(e, x): 
        """Converts R character vector to Python str or list of strs/None."""
        return _r_scalar_to_py(x) if len(x) == 1 else _r_vec_to_py_list(x) 
        
    @eng.register_r(rv.ComplexSexpVector) 
    @eng.register_r(rv.ComplexSexpVector) 
    def _(e, x): 
        """Converts R complex vector to Python complex or list of complex."""
        return x[0] if len(x) == 1 else list(x) 
        
    @eng.register_r(rv.ByteSexpVector) 
    @eng.register_r(rv.ByteSexpVector) 
    def _(e, x): 
        """Converts R raw vector to Python bytes."""
        return bytes(x) if len(x) else b"" 
        
    # ---------- Python -> R (collections) ---------- 
    @eng.register_py(dict) 
    def _(e, x): 
        """Converts Python dict to R list (named list)."""
        return rv.ListVector({str(k): e.py2r(v) for k, v in x.items()}) 
        
    @eng.register_py(list) 
    @eng.register_py(tuple) 
    def _(e, x): 
        """Converts Python list/tuple to R list."""
        return rv.ListSexpVector([e.py2r(v) for v in x]) 
        
    # ---------- R -> Python (collections) ----------     
    @eng.register_r(rv.ListSexpVector) 
    def _(e, x): 
        """Converts R list to Python list or dict (if all elements are named)."""
        if isinstance(x.names, rpy2.rinterface_lib.sexp.NULLType): 
            vals = [e.r2py(elt) for elt in x] 
            return vals 
        names = list(x.names) if x.names is not None else [None] * len(x) 
        
        vals = [e.r2py(elt) for elt in x] 
        clean = all(n and n != "" for n in names) and len(set(names)) == len(names) 
        
        return {n: v for n, v in zip(names, vals)} if clean else vals 
        
    return eng 