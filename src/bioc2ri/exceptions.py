__author__ = "MaximilianNuber"
__copyright__ = "MaximilianNuber"
__license__ = "MIT"

class RFunctionNotFoundError(Exception):
    """Raised when an R function cannot be found or loaded."""
    pass

class RPackageNotLoadedError(Exception):
    """Raised when an R package fails to load."""
    pass