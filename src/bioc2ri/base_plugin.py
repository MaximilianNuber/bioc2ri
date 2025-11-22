# --- base_plugin.py --- 
from functools import cache 
from .engine import Engine

__author__ = "MaximilianNuber"
__copyright__ = "MaximilianNuber"
__license__ = "MIT"

@cache 
def base_plugin(): 
    from rpy2.robjects import vectors as rv 
    from rpy2.robjects import vectors as rv 
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
    
    def _(e, x): 
        # None -> NULL 
        return NULL 
        
    @eng.register_py(bool) 
    def _(e, x): 
        return rv.BoolSexpVector([bool(x)]) 
        
    @eng.register_py(int) 
    def _(e, x): 
        # R integers are 32-bit. If you expect big ints, either: 
        # - promote to double, or 
        # - raise in strict mode. Here we promote if out-of-range. 
        try: 
            if -(2**31) <= x < 2**31: 
                return rv.IntSexpVector([int(x)]) 
            else: 
                return rv.FloatSexpVector([float(x)]) 
        except Exception: 
            return rv.FloatSexpVector([float(x)]) 
            
    @eng.register_py(float) 
    def _(e, x): 
        return rv.FloatSexpVector([float(x)]) 
        
    @eng.register_py(str) 
    def _(e, x): 
        return rv.StrSexpVector([x]) 
        
    @eng.register_py(complex) 
    def _(e, x): 
        return rv.ComplexSexpVector([x]) 
        
    @eng.register_py(bytes) 
    def _(e, x): 
        # R 'raw' vector 
        return rv.ByteSexpVector(x) 
        
    # ---------- R -> Python (atomic vectors) ---------- 
    @eng.register_r(rv.BoolSexpVector) 
    def _(e, x): 
        return _r_scalar_to_py(x) if len(x) == 1 else _r_vec_to_py_list(x) 
        
    @eng.register_r(rv.IntSexpVector) 
    def _(e, x): 
        new = _r_scalar_to_py(x) if len(x) == 1 else _r_vec_to_py_list(x) 
        return new 
        
    @eng.register_r(rv.FloatSexpVector) 
    def _(e, x): 
        return _r_scalar_to_py(x) if len(x) == 1 else _r_vec_to_py_list(x) 
        
    @eng.register_r(rv.StrSexpVector) 
    def _(e, x): 
        return _r_scalar_to_py(x) if len(x) == 1 else _r_vec_to_py_list(x) 
        
    @eng.register_r(rv.ComplexSexpVector) 
    def _(e, x): 
        # rpy2 yields Python complex numbers 
        return x[0] if len(x) == 1 else list(x) 
        
    @eng.register_r(rv.ByteSexpVector) 
    def _(e, x): 
        # raw vector -> bytes 
        return bytes(x) if len(x) else b"" 
        
    # ---------- Python -> R (collections) ---------- 
    @eng.register_py(dict) 
    def _(e, x): 
        return rv.ListVector({str(k): e.py2r(v) for k, v in x.items()}) 
        
    @eng.register_py(list) 
    @eng.register_py(tuple) 
    def _(e, x): 
        return rv.ListSexpVector([e.py2r(v) for v in x]) 
        
    # ---------- R -> Python (collections) ----------     
    @eng.register_r(rv.ListSexpVector) 
    def _(e, x): 
        if isinstance(x.names, rpy2.rinterface_lib.sexp.NULLType): 
            vals = [e.r2py(elt) for elt in x] 
            return vals 
        names = list(x.names) if x.names is not None else [None] * len(x) 
        
        vals = [e.r2py(elt) for elt in x] 
        clean = all(n and n != "" for n in names) and len(set(names)) == len(names) 
        
        return {n: v for n, v in zip(names, vals)} if clean else vals 
        
    return eng 