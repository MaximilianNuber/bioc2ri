from .lazy_r_env import get_r_environment

__author__ = "MaximilianNuber"
__copyright__ = "MaximilianNuber"
__license__ = "MIT"

renv = get_r_environment()
r = renv.ro.r


def ensure_core_bioc2ri_packages() -> None:
    """
    Ensure that the core R packages needed by bioc2ri are installed:

      - BiocManager  (CRAN)
      - S4Vectors    (Bioconductor; provides DFrame)
      - Matrix       (CRAN; provides sparse matrix classes)

    Safe to call multiple times (idempotent).
    """
    r(
        """
        local({
          # Make sure CRAN repo is usable
          repos <- getOption("repos")
          if (is.null(repos) || is.na(repos["CRAN"]) || repos["CRAN"] == "@CRAN@") {
            repos["CRAN"] <- "https://cloud.r-project.org"
            options(repos = repos)
          }

          # 1) BiocManager from CRAN
          if (!requireNamespace("BiocManager", quietly = TRUE)) {
            install.packages("BiocManager")
          }

          # 2) S4Vectors (for DFrame) via BiocManager
          if (!requireNamespace("S4Vectors", quietly = TRUE)) {
            BiocManager::install("S4Vectors", update = FALSE, ask = FALSE)
          }

          # 3) Matrix from CRAN
          if (!requireNamespace("Matrix", quietly = TRUE)) {
            install.packages("Matrix")
          }
        })
        """
    )