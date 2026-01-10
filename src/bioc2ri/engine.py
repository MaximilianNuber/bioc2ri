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
    """Core conversion engine for bidirectional Python-R interoperability.

    Manages conversion rules between Python objects and R objects (including S4 classes),
    providing registration mechanisms and conversion dispatch.

    Attributes:
        py_rules: List of (type, handler) tuples for Python-to-R conversion.
        r_rules: List of (type, handler) tuples for R-to-Python conversion.
        s4_rules: Dictionary mapping R class names to handler functions for S4 objects.
    """
    py_rules: List[PyRule] = field(default_factory=list)
    r_rules:  List[RRule]  = field(default_factory=list)
    s4_rules: Dict[str, Callable[['Engine', Any], Any]] = field(default_factory=dict)

    # --- Decorator-style registration ---
    def register_py(self, typ: Type):
        """Decorator to register a Python-to-R conversion rule.

        The decorated function should accept (engine, obj) and return the R object.
        Rules registered later take precedence (LIFO).

        Args:
            typ: The Python type to handle.

        Returns:
            A decorator that registers the function.
        """
        def deco(fn: Callable[['Engine', Any], Any]):
            self.py_rules.insert(0, (typ, fn))
            return fn
        return deco

    def register_r(self, typ: Type):
        """Decorator to register an R-to-Python conversion rule.

        The decorated function should accept (engine, obj) and return the Python object.
        Rules registered later take precedence (LIFO).

        Args:
            typ: The R object type (e.g., from rpy2) to handle.

        Returns:
            A decorator that registers the function.
        """
        def deco(fn: Callable[['Engine', Any], Any]):
            self.r_rules.insert(0, (typ, fn))
            return fn
        return deco

    def register_s4(self, r_class: str):
        """Decorator to register an S4 class conversion rule.

        The decorated function should accept (engine, obj) and return the Python object.

        Args:
            r_class: The name of the R S4 class to handle.

        Returns:
            A decorator that registers the function.
        """
        def deco(fn: Callable[['Engine', Any], Any]):
            self.s4_rules[r_class] = fn
            return fn
        return deco

    # --- Programmatic registration (optional) ---
    def add_py(self, typ: Type, fn: Callable[['Engine', Any], Any]):
        """Directly adds a Python-to-R conversion rule.

        Args:
            typ: The Python type to handle.
            fn: Conversion function taking (engine, object).
        """
        self.py_rules.insert(0, (typ, fn))

    def add_r(self, typ: Type, fn: Callable[['Engine', Any], Any]):
        """Directly adds an R-to-Python conversion rule.

        Args:
            typ: The R object type to handle.
            fn: Conversion function taking (engine, object).
        """
        self.r_rules.insert(0, (typ, fn))

    def add_s4(self, r_class: str, fn: Callable[['Engine', Any], Any]):
        """Directly adds an S4 conversion rule.

        Args:
            r_class: The name of the R S4 class.
            fn: Conversion function taking (engine, object).
        """
        self.s4_rules[r_class] = fn

    # --- Conversion ---
    def py2r(self, x: Any):
        """Converts a Python object to its R representation.

        Iterates through registered Python rules in LIFO order to find a matching type.

        Args:
            x: The Python object to convert.

        Returns:
            The converted R object.

        Raises:
            TypeError: If no suitable rule is found.
        """
        for typ, fn in self.py_rules:
            if isinstance(x, typ):
                return fn(self, x)
        raise TypeError(f"No py2r rule for {type(x)}")

    def r2py(self, x: Any):
        """Converts an R object to its Python representation.

        Checks first if the object is an S4 instance with a registered handler.
        Otherwise, iterates through registered R rules in LIFO order.

        Args:
            x: The R object to convert.

        Returns:
            The converted Python object, or the original object if no rule matches.
        """
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
        """Creates a new Engine incorporating rules from other Engine instances (plugins).

        The new engine inherits rules from this instance, and then prepends/overwrites
        with rules from the provided plugins.

        Args:
            *plugins: Other Engine instances to merge rules from.

        Returns:
            A new Engine instance with combined rules.
        """
        new = Engine(self.py_rules.copy(), self.r_rules.copy(), self.s4_rules.copy())
        for p in plugins:
            new.py_rules  = p.py_rules  + new.py_rules
            new.r_rules   = p.r_rules   + new.r_rules
            new.s4_rules.update(p.s4_rules)
        return new