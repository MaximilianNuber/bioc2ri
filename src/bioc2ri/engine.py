# --- engine.py ---
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple, Type
from rpy2.robjects import r, vectors as rv
# from rpy2.rinterface_lib.sexp import SexpS4
from rpy2.rinterface import SexpS4

__author__ = "MaximilianNuber"
__copyright__ = "MaximilianNuber"
__license__ = "MIT"

PyRule = Tuple[Type, Callable[['Engine', Any], Any]]
RRule  = Tuple[Type, Callable[['Engine', Any], Any]]

@dataclass
class Engine:
    py_rules: List[PyRule] = field(default_factory=list)
    r_rules:  List[RRule]  = field(default_factory=list)
    s4_rules: Dict[str, Callable[['Engine', Any], Any]] = field(default_factory=dict)

    # --- Decorator-style registration ---
    def register_py(self, typ: Type):
        def deco(fn: Callable[['Engine', Any], Any]):
            self.py_rules.insert(0, (typ, fn))
            return fn
        return deco

    def register_r(self, typ: Type):
        def deco(fn: Callable[['Engine', Any], Any]):
            self.r_rules.insert(0, (typ, fn))
            return fn
        return deco

    def register_s4(self, r_class: str):
        def deco(fn: Callable[['Engine', Any], Any]):
            self.s4_rules[r_class] = fn
            return fn
        return deco

    # --- Programmatic registration (optional) ---
    def add_py(self, typ: Type, fn: Callable[['Engine', Any], Any]):
        self.py_rules.insert(0, (typ, fn))

    def add_r(self, typ: Type, fn: Callable[['Engine', Any], Any]):
        self.r_rules.insert(0, (typ, fn))

    def add_s4(self, r_class: str, fn: Callable[['Engine', Any], Any]):
        self.s4_rules[r_class] = fn

    # --- Conversion ---
    def py2r(self, x: Any):
        for typ, fn in self.py_rules:
            if isinstance(x, typ):
                return fn(self, x)
        raise TypeError(f"No py2r rule for {type(x)}")

    def r2py(self, x: Any):
        if isinstance(x, SexpS4):
            for cls in tuple(r["class"](x)):
                if cls in self.s4_rules:
                    return self.s4_rules[cls](self, x)
            return x  # unknown S4
        for typ, fn in self.r_rules:
            if isinstance(x, typ):
                return fn(self, x)
        return x  # fallback: leave as rpy2 wrapper

    def with_plugins(self, *plugins: 'Engine'):
        new = Engine(self.py_rules.copy(), self.r_rules.copy(), self.s4_rules.copy())
        for p in plugins:
            new.py_rules  = p.py_rules  + new.py_rules
            new.r_rules   = p.r_rules   + new.r_rules
            new.s4_rules.update(p.s4_rules)
        return new