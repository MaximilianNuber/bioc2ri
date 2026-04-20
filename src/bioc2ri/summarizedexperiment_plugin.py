# --- bioc_experiment_plugin.py ---
from __future__ import annotations
from typing import Any
from bioc2ri.lazy_r_env import get_r_environment
from bioc2ri.scipy_sparse_plugin import scipy_sparse_plugin

__author__ = "MaximilianNuber"
__license__ = "MIT"

# --- bioc_experiment_plugin.py ---
from typing import Any
from bioc2ri.lazy_r_env import get_r_environment
from bioc2ri.rnames import set_rownames, set_colnames, get_rownames, get_colnames, get_names

__author__ = "MaximilianNuber"
__license__ = "MIT"

def bioc_experiment_plugin() -> "Engine":
    # Python imports (Internal for laziness)
    from summarizedexperiment import SummarizedExperiment, RangedSummarizedExperiment
    from singlecellexperiment import SingleCellExperiment
    from bioc2ri.engine import Engine
    from bioc2ri.biocpy_plugin import biocpy_plugin
    from bioc2ri.bioc_ranges_plugin import bioc_ranges_plugin
    from bioc2ri.numpy_plugin import numpy_plugin
    
    from rpy2.rinterface import NULL as r_NULL
    
    eng = Engine()
    renv = get_r_environment()
    r = renv.ro.r
    
    # R Package Imports
    se_pkg = renv.importr("SummarizedExperiment")
    sce_pkg = renv.importr("SingleCellExperiment")
    s4_pkg = renv.importr("S4Vectors")

    # -------------------------------------------------
    # Python -> R
    # -------------------------------------------------
    @eng.register_py(SummarizedExperiment)
    @eng.register_py(RangedSummarizedExperiment)
    @eng.register_py(SingleCellExperiment)
    def _(e: Engine, x: SummarizedExperiment):
        import numpy as np
        from scipy.sparse import issparse
        from bioc2ri.numpy_plugin import numpy_plugin
        from bioc2ri.scipy_sparse_plugin import scipy_sparse_plugin
        from biocutils import StringList

        # 1. Convert Assays individually
        assay_keys = list(x.get_assays().keys())
        r_assays_dict = {}
        for k in assay_keys:
            v = x.get_assay(k)
            if isinstance(v, np.ndarray):
                r_assays_dict[k] = numpy_plugin().py2r(v)
            elif issparse(v):
                r_assays_dict[k] = scipy_sparse_plugin().py2r(v)
            else:
                r_assays_dict[k] = e.py2r(v)

        # 2. Build the SimpleList and FORCE names using R
        # r_assays = s4_pkg.SimpleList(*r_assays_list)
        # r_assays = r["names<-"](r_assays, value=biocpy_plugin().py2r(StringList(assay_keys)))
        r_assays = renv.ListVector(r_assays_dict)

        # 3. colData Conversion
        py_col_data = x.get_column_data()
        r_col_data = biocpy_plugin().py2r(py_col_data) if py_col_data is not None else r_NULL

        # 4. Construct base SE
        # Note: We pass r_assays as a positional argument or named 'assays'
        r_obj = se_pkg.SummarizedExperiment(assays=r_assays, colData=r_col_data)

        # 5. Handle Row-side Strategy
        py_row_ranges = getattr(x, "get_row_ranges", lambda: None)()
        py_row_data = x.get_row_data()
        
        has_ranges = py_row_ranges is not None and len(py_row_ranges) > 0
        has_data = py_row_data is not None and py_row_data.shape[1] > 0

        if isinstance(x, SingleCellExperiment):
            r_obj = sce_pkg.SingleCellExperiment(r_obj)

        if has_ranges:
            r_ranges = bioc_ranges_plugin().py2r(py_row_ranges)
            if has_data:
                r_mcols = biocpy_plugin().py2r(py_row_data)
                r_ranges = r["mcols<-"](r_ranges, value=r_mcols)
            # Explicitly set rowRanges using replacement function
            # r_obj = r["rowRanges<-"](r_obj, value=r_ranges)
            r_obj = se_pkg.SummarizedExperiment(assays=r_assays, colData=r_col_data, rowRanges=r_ranges)
            
        elif has_data:
            # r_obj = r["rowData<-"](r_obj, value=biocpy_plugin().py2r(py_row_data))
            
            r_obj = se_pkg.SummarizedExperiment(assays=r_assays, colData=r_col_data, rowData=rowData)

        # 6. Set Names (Syncing via rnames.py)
        rn, cn = x.get_row_names(), x.get_column_names()
        if rn is not None:
            r_obj = set_rownames(r_obj, list(rn))
        if cn is not None:
            r_obj = set_colnames(r_obj, list(cn))

        # 7. SingleCellExperiment Specifics
        if isinstance(x, SingleCellExperiment):
            if has_ranges:
                r_ranges = bioc_ranges_plugin().py2r(py_row_ranges)
                if has_data:
                    r_mcols = biocpy_plugin().py2r(py_row_data)
                    r_ranges = r["mcols<-"](r_ranges, value=r_mcols)
                # Explicitly set rowRanges using replacement function
                # r_obj = r["rowRanges<-"](r_obj, value=r_ranges)
                r_obj = sce_pkg.SingleCellExperiment(assays=r_assays, colData=r_col_data, rowRanges=r_ranges)
                
            elif has_data:
                # r_obj = r["rowData<-"](r_obj, value=biocpy_plugin().py2r(py_row_data))
                
                r_obj = sce_pkg.SingleCellExperiment(assays=r_assays, colData=r_col_data, rowData=rowData)
                    
            
            # Reduced Dims: Use names replacement to be certain
            red_dims_py = x.get_reduced_dims()
            if red_dims_py:
                red_keys = list(red_dims_py.keys())
                r_red_list = []
                for k in red_keys:
                    v = red_dims_py[k]
                    r_red_list.append(numpy_plugin().py2r(v) if isinstance(v, np.ndarray) else e.py2r(v))
                
                r_red_slist = s4_pkg.SimpleList(*r_red_list)
                r_red_slist = r["names<-"](r_red_slist, value=biocpy_plugin().py2r(StringList(red_keys)))
                r_obj = r["reducedDims<-"](r_obj, value=r_red_slist)
            
            # AltExps: Use names replacement
            alt_exps_py = x.get_alternative_experiments()
            if alt_exps_py:
                alt_keys = list(alt_exps_py.keys())
                r_alt_list = [e.py2r(alt_exps_py[k]) for k in alt_keys]
                r_alt_slist = s4_pkg.SimpleList(*r_alt_list)
                r_alt_slist = r["names<-"](r_alt_slist, value=biocpy_plugin().py2r(StringList(alt_keys)))
                r_obj = r["altExps<-"](r_obj, value=r_alt_slist)

        return r_obj
    
    # @eng.register_py(SummarizedExperiment)
    # @eng.register_py(RangedSummarizedExperiment)
    # @eng.register_py(SingleCellExperiment)
    # def _(e: Engine, x: SummarizedExperiment):
    #     import numpy as np
    #     from scipy.sparse import issparse
    #     from bioc2ri.numpy_plugin import numpy_plugin
    #     from bioc2ri.scipy_sparse_plugin import scipy_sparse_plugin

    #     # 1. Convert Assays individually based on type
    #     r_assays_dict = {}
    #     for k, v in x.get_assays().items():
    #         if isinstance(v, np.ndarray):
    #             # Use the dense numpy plugin
    #             r_assays_dict[k] = numpy_plugin().py2r(v)
    #         elif issparse(v):
    #             # Use the sparse matrix plugin (typically becomes a dgCMatrix in R)
    #             r_assays_dict[k] = scipy_sparse_plugin().py2r(v)
    #         else:
    #             # Fallback: if it's a custom array type, try the engine 'e' 
    #             # or raise a clear error.
    #             try:
    #                 r_assays_dict[k] = e.py2r(v)
    #             except Exception:
    #                 raise TypeError(f"Assay '{k}' has an unsupported type: {type(v)}")

    #     r_assays = s4_pkg.SimpleList(**r_assays_dict)

    #     # 2. colData (Python check)
    #     py_col_data = x.get_column_data()
    #     r_col_data = biocpy_plugin().py2r(py_col_data) if py_col_data is not None else r_NULL

    #     # 3. Row-side Strategy (Python-side checks)
    #     py_row_ranges = getattr(x, "get_row_ranges", lambda: None)()
    #     py_row_data = x.get_row_data()
        
    #     se_kwargs = {"assays": r_assays, "colData": r_col_data}

    #     has_ranges = py_row_ranges is not None and len(py_row_ranges) > 0
    #     has_data = py_row_data is not None and py_row_data.shape[1] > 0

    #     if has_ranges:
    #         r_ranges = bioc_ranges_plugin().py2r(py_row_ranges)
    #         if has_data:
    #             # Sync rowData into the mcols of the ranges
    #             r_mcols = biocpy_plugin().py2r(py_row_data)
    #             r_ranges = r["mcols<-"](r_ranges, value=r_mcols)
    #         se_kwargs["rowRanges"] = r_ranges
    #     elif has_data:
    #         se_kwargs["rowData"] = biocpy_plugin().py2r(py_row_data)

    #     # 4. Construct
    #     r_obj = se_pkg.SummarizedExperiment(**se_kwargs)

    #     # 5. Set Names (CRITICAL for "parallel" validation)
    #     # SE row names come from the rowData/rowRanges index
    #     # SE col names come from the colData index
    #     rn = x.get_row_names()
    #     cn = x.get_column_names()
        
    #     if rn is not None:
    #         r_obj = set_rownames(r_obj, rn)
    #     if cn is not None:
    #         r_obj = set_colnames(r_obj, cn)

    #     # 6. SingleCellExperiment Specifics
    #     # if isinstance(x, SingleCellExperiment):
    #     #     r_obj = sce_pkg.SingleCellExperiment(r_obj)
            
    #     #     red_dims_py = x.get_reduced_dims()
    #     #     if red_dims_py:
    #     #         r_red_dims_dict = {k: numpy_plugin().py2r(v) for k, v in red_dims_py.items()}
    #     #         r_obj = sce_pkg.reducedDims_set(r_obj, value=s4_pkg.SimpleList(**r_red_dims_dict))
            
    #     #     alt_exps_py = x.get_alt_exps()
    #     #     if alt_exps_py:
    #     #         # Recursion on the Experiment Engine itself
    #     #         r_alts_dict = {k: e.py2r(v) for k, v in alt_exps_py.items()}
    #     #         r_obj = sce_pkg.altExps_set(r_obj, value=s4_pkg.SimpleList(**r_alts_dict))
        
    #     # 6. SingleCellExperiment Specifics
    #     if isinstance(x, SingleCellExperiment):
    #         r_obj = sce_pkg.SingleCellExperiment(r_obj)
            
    #         # Reduced Dims: Per-item type checking loop
    #         red_dims_py = x.get_reduced_dims()
    #         if red_dims_py:
    #             r_red_dims_dict = {}
    #             for k, v in red_dims_py.items():
    #                 if isinstance(v, np.ndarray):
    #                     r_red_dims_dict[k] = numpy_plugin().py2r(v)
    #                 elif issparse(v):
    #                     r_red_dims_dict[k] = scipy_sparse_plugin().py2r(v)
    #                 else:
    #                     # Fallback for custom objects
    #                     r_red_dims_dict[k] = e.py2r(v)
                
    #             # Use R replacement function to set the whole list
    #             r_obj = r["reducedDims<-"](r_obj, value=s4_pkg.SimpleList(**r_red_dims_dict))
            
    #         # AltExps: Recursive call on the experiment engine
    #         alt_exps_py = x.get_alternative_experiments()
    #         if alt_exps_py:
    #             r_alts_dict = {k: e.py2r(v) for k, v in alt_exps_py.items()}
    #             r_obj = r["altExps<-"](r_obj, value=s4_pkg.SimpleList(**r_alts_dict))

    #     return r_obj

    # -------------------------------------------------
    # R -> Python
    # -------------------------------------------------
    # @eng.register_s4("SummarizedExperiment")
    # @eng.register_s4("RangedSummarizedExperiment")
    # @eng.register_s4("SingleCellExperiment")
    # def _(e: Engine, x: Any):
    #     r_classes = list(r["class"](x))
        
    #     # 1. Assays by index (Same as before)
    #     r_assays = se_pkg.assays(x)
    #     # assay_names = get_names(se_pkg.assayNames(x))
    #     assay_names = get_names(se_pkg.assays(x))
    #     assays_py = {}
    #     def get_r_type(r_inst, obj):
    #         """Identifies the R matrix type for plugin routing."""
    #         if r_inst["is"](obj, "sparseMatrix")[0]:
    #             return "sparse"
    #         if r_inst["is.matrix"](obj)[0]:
    #             return "dense_matrix"
    #         if r_inst["is"](obj, "denseMatrix")[0] or r_inst["is.matrix"](obj)[0]:
    #             return "dense"
    #         if r_inst["is.array"](obj)[0]:
    #             return "array"
    #         return "unknown"
    #     for i in range(1, int(r["length"](r_assays)[0]) + 1):
    #         r_mat = r["[["](r_assays, i)
    #         key = assay_names[i-1] if (assay_names and assay_names[i-1]) else str(i)
    #         if get_r_type(r, r_mat) == "sparse":
    #             assays_py[key] = scipy_sparse_plugin().r2py(r_mat)
    #         else:
    #             assays_py[key] = numpy_plugin().r2py(r_mat)

    #     # 2. Metadata Sanitizer Helper
    #     def _sanitize_metadata(r_df):
    #         if isinstance(r_df, type(r_NULL)):
    #             return None
    #         # Use standalone plugin
    #         py_df = biocpy_plugin().r2py(r_df)
            
    #         # CHECK: If BiocFrame has 0 rows but R says it has >0 rows,
    #         # it was a 0-column R DataFrame. Return None so Python 
    #         # SE constructor can generate a correctly-sized placeholder.
    #         r_nrow = int(r["nrow"](r_df)[0])
    #         if py_df is not None and py_df.shape[0] == 0 and r_nrow > 0:
    #             return None
    #         return py_df

    #     col_data_py = _sanitize_metadata(se_pkg.colData(x))
    #     row_data_py = _sanitize_metadata(se_pkg.rowData(x))
        
    #     # Get names via rnames.py
    #     rn, cn = get_rownames(x), get_colnames(x)
        
    #     # # 3. Row-side Ranges
    #     # r_row_ranges = r["rowRanges"](x)
    #     # is_ranged = not isinstance(r_row_ranges, type(r_NULL))
    #     # row_ranges_py = bioc_ranges_plugin().r2py(r_row_ranges) if is_ranged else None
        
    #     # 3. Row-side logic (Inside the r2py registration)
    #     r_row_ranges = r["rowRanges"](x)
        
    #     # Check if it's a GRangesList / CompressedGRangesList
    #     if not isinstance(r_row_ranges, type(r_NULL)) and r["is"](r_row_ranges, "GRangesList")[0]:
    #         # Collapse the hierarchical list into a single representative range per row.
    #         # This ensures it's now a standard 'GRanges' object.
    #         r_row_ranges = r["range"](r_row_ranges)
            
    #     is_ranged = not isinstance(r_row_ranges, type(r_NULL))
        
    #     # Now bioc_ranges_plugin().r2py() will recognize it!
    #     row_ranges_py = bioc_ranges_plugin().r2py(r_row_ranges) if is_ranged else None
        
    #     # 4. Constructor Dispatch
    #     if "SingleCellExperiment" in r_classes:
    #         # Reduced Dims
    #         r_reds = sce_pkg.reducedDims(x)
    #         red_names = get_names(r["names"](r_reds))
    #         red_dims_py = {}
    #         for i in range(1, int(r["length"](r_reds)[0]) + 1):
    #             r_mat = r["[["](r_reds, i)
    #             key = red_names[i-1] if (red_names and red_names[i-1]) else str(i)
    #             red_dims_py[key] = numpy_plugin().r2py(r_mat)
                
    #         # AltExps (Recursive call)
    #         r_alts = sce_pkg.altExps(x)
    #         alt_names = get_names(r["names"](r_alts))
    #         alt_exps_py = {alt_names[i-1]: e.r2py(r["[["](r_alts, i)) for i in range(1, int(r["length"](r_alts)[0]) + 1)}
            
    #         print(type(assays_py))
    #         for assay_name, assay_data in assays_py.items():
    #             print(f"Assay '{assay_name}': {type(assay_data)}")
    #             print(r["class"](assay_data))
    #         print(type(row_ranges_py))
    #         # print(r["class"](row_ranges_py))
    #         print(type(row_data_py))
    #         print(type(col_data_py))
    #         print(type(red_dims_py))
    #         print(type(alt_exps_py))
    #         print(type(rn))
    #         print(type(cn))
            

    #         return SingleCellExperiment(
    #             assays=assays_py,
    #             row_ranges=row_ranges_py,
    #             row_data=row_data_py,
    #             column_data=col_data_py,
    #             row_names=rn,
    #             column_names=cn,
    #             reduced_dims=red_dims_py,
    #             alternative_experiments=alt_exps_py
    #         )
            
    #     if is_ranged:
    #         return RangedSummarizedExperiment(
    #             assays=assays_py,
    #             row_ranges=row_ranges_py,
    #             row_data=row_data_py,
    #             column_data=col_data_py,
    #             row_names=rn,
    #             column_names=cn
    #         )
            
    #     return SummarizedExperiment(
    #         assays=assays_py,
    #         row_data=row_data_py,
    #         column_data=col_data_py,
    #         row_names=rn,
    #         column_names=cn
    #     )
    @eng.register_s4("SummarizedExperiment")
    @eng.register_s4("RangedSummarizedExperiment")
    @eng.register_s4("SingleCellExperiment")
    def _(e: Engine, x: Any):
        from bioc2ri.numpy_plugin import numpy_plugin
        from bioc2ri.biocpy_plugin import biocpy_plugin
        from bioc2ri.bioc_ranges_plugin import bioc_ranges_plugin
        from bioc2ri.scipy_sparse_plugin import scipy_sparse_plugin
        import numpy as np

        r_classes = list(r["class"](x))
        
        # 1. Access Assays correctly
        r_assays = se_pkg.assays(x)
        n_assays = int(r["length"](r_assays)[0])
        # Use assayNames() for best results in R
        assay_names = get_names(r_assays)
        
        assays_py = {}
        for i in range(1, n_assays + 1):
            r_mat = r["[["](r_assays, i)
            key = assay_names[i-1] if (assay_names and assay_names[i-1]) else str(i)
            
            # Identify the R matrix type
            if r["is"](r_mat, "sparseMatrix")[0]:
                assays_py[key] = scipy_sparse_plugin().r2py(r_mat)
            elif r["is"](r_mat, "denseMatrix")[0] or r["is.matrix"](r_mat)[0]:
                assays_py[key] = numpy_plugin().r2py(r_mat)
            else:
                # FALLBACK: If an assay is NOT a matrix (like your RSE-in-assay case),
                # we try a standard conversion, but warn that SE expects matrices.
                res = e.r2py(r_mat)
                if hasattr(res, "__class__") and "RS4" in str(type(res)):
                    # Last resort: Force to numpy array to provide a .shape
                    res = np.array(res)
                assays_py[key] = res

        # 2. Metadata components
        col_data_py = biocpy_plugin().r2py(se_pkg.colData(x))
        # Ensure col_data is not a 0-row ghost
        if col_data_py is not None and col_data_py.shape[0] == 0 and int(r["nrow"](x)[0]) > 0:
            col_data_py = None
            
        rn, cn = get_rownames(x), get_colnames(x)
        
        # 3. Row-side logic (Hierarchical or Flat)
        r_row_ranges = r["rowRanges"](x)
        row_ranges_py = None
        if not isinstance(r_row_ranges, type(r_NULL)):
            # bioc_ranges_plugin now handles CompressedGenomicRangesList!
            row_ranges_py = bioc_ranges_plugin().r2py(r_row_ranges)
            
        row_data_py = biocpy_plugin().r2py(se_pkg.rowData(x))
        if row_data_py is not None and row_data_py.shape[0] == 0 and int(r["nrow"](x)[0]) > 0:
            row_data_py = None

        # 4. SCE Specifics
        if "SingleCellExperiment" in r_classes:
            # Reduced Dims
            r_reds = sce_pkg.reducedDims(x)
            red_names = get_names(r["names"](r_reds))
            red_dims_py = {
                (red_names[i-1] if red_names else str(i)): numpy_plugin().r2py(r["[["](r_reds, i))
                for i in range(1, int(r["length"](r_reds)[0]) + 1)
            }
                
            # AltExps (Recursive call)
            r_alts = sce_pkg.altExps(x)
            alt_names = get_names(r["names"](r_alts))
            alt_exps_py = {
                alt_names[i-1]: e.r2py(r["[["](r_alts, i)) 
                for i in range(1, int(r["length"](r_alts)[0]) + 1)
            }
            
            
            return SingleCellExperiment(
                assays=assays_py,
                row_ranges=row_ranges_py,
                row_data=row_data_py,
                column_data=col_data_py,
                row_names=rn,
                column_names=cn,
                reduced_dims=red_dims_py,
                alternative_experiments=alt_exps_py # Use the full constructor name
            )
            
        if not isinstance(r_row_ranges, type(r_NULL)):
            return RangedSummarizedExperiment(
                assays=assays_py, row_ranges=row_ranges_py, row_data=row_data_py,
                column_data=col_data_py, row_names=rn, column_names=cn
            )
            
        return SummarizedExperiment(
            assays=assays_py, row_data=row_data_py, column_data=col_data_py, 
            row_names=rn, column_names=cn
        )
        
    return eng
    # @eng.register_s4("SummarizedExperiment")
    # @eng.register_s4("RangedSummarizedExperiment")
    # @eng.register_s4("SingleCellExperiment")
    # def _(e: Engine, x: Any):
    #     r_classes = list(r["class"](x))
        
    #     ####
    #     r_assays = se_pkg.assays(x)
    #     n_assays = int(r["length"](r_assays)[0])
        
    #     # Get names as a list (can be None or contain empty strings)
    #     assay_names_list = get_names(se_pkg.assayNames(x))
    #     assays_py = {}
        
    #     def get_r_type(r, x):
    #         # 1. Check for Sparse (using the virtual class 'sparseMatrix')
    #         if r["is"](x, "sparseMatrix")[0]:
    #             return "sparse"
            
    #         # 2. Check for Dense Matrix
    #         if r["is.matrix"](x)[0]:
    #             return "dense_matrix"
            
    #         # 3. Check for Array (covers 3D+ data)
    #         if r["is.array"](x)[0]:
    #             return "array"
            
    #         return "unknown"
    #     def get_r_type(r_inst, obj):
    #         """Identifies the R matrix type for plugin routing."""
    #         if r_inst["is"](obj, "sparseMatrix")[0]:
    #             return "sparse"
    #         if r_inst["is.matrix"](obj)[0]:
    #             return "dense_matrix"
    #         if r_inst["is.array"](obj)[0]:
    #             return "array"
    #         return "unknown"

    #     for i in range(1, n_assays + 1):
    #         # Index by number (1-based for R) to ensure we hit every assay
    #         r_assay = r["[["](r_assays, i)
            
    #         # Determine the dictionary key:
    #         # Use name if it exists and isn't empty, otherwise use the index string.
    #         key = str(i) 
    #         if assay_names_list is not None:
    #             name_candidate = assay_names_list[i-1]
    #             # Filter out None, empty strings, or R 'NA' strings
    #             if name_candidate and name_candidate not in ("", "NA"):
    #                 key = str(name_candidate)

    #         # Route to the appropriate standalone plugin
    #         if get_r_type(r, r_assay) == "sparse":
    #             assays_py[key] = scipy_sparse_plugin().r2py(r_assay)
    #         else:
    #             assays_py[key] = numpy_plugin().r2py(r_assay)
        
    #     col_data_py = biocpy_plugin().r2py(se_pkg.colData(x))
        
    #     # Determine names using rnames.py
    #     rn = get_rownames(x)
    #     cn = get_colnames(x)
        
    #     r_row_ranges = renv.ro.r["rowRanges"](x)
    #     is_ranged = not isinstance(r_row_ranges, type(r_NULL))
        
    #     row_ranges_py = bioc_ranges_plugin().r2py(r_row_ranges) if is_ranged else None
    #     row_data_py = biocpy_plugin().r2py(se_pkg.rowData(x))
        
    #     if "SingleCellExperiment" in r_classes:
    #         red_dims_py = {str(k): numpy_plugin().r2py(v) for k, v in sce_pkg.reducedDims(x).items()}
    #         alt_exps_py = {str(k): e.r2py(v) for k, v in sce_pkg.altExps(x).items()}
            
    #         return SingleCellExperiment(
    #             assays=assays_py, row_ranges=row_ranges_py, column_data=col_data_py,
    #             reduced_dims=red_dims_py, alt_exps=alt_exps_py, column_names=cn
    #         )
            
    #     if is_ranged:
    #         return RangedSummarizedExperiment(
    #             assays=assays_py, row_ranges=row_ranges_py, column_data=col_data_py, column_names=cn
    #         )
            
    #     return SummarizedExperiment(
    #         assays=assays_py, row_data=row_data_py, column_data=col_data_py, column_names=cn
    #     )

    # return eng


# def bioc_experiment_plugin() -> "Engine":
#     # Python imports
#     from summarizedexperiment import SummarizedExperiment, RangedSummarizedExperiment
#     from singlecellexperiment import SingleCellExperiment
#     from bioc2ri.engine import Engine
#     from bioc2ri.biocpy_plugin import biocpy_plugin
#     from bioc2ri.bioc_ranges_plugin import bioc_ranges_plugin
#     from bioc2ri.numpy_plugin import numpy_plugin
    
#     # Correct way to handle R NULL in rpy2
#     from rpy2.rinterface import NULL as r_NULL
    
#     eng = Engine()
#     renv = get_r_environment()
#     r = renv.ro.r
    
#     # Individual engines for specific component types
#     bioc_eng = biocpy_plugin()
#     ranges_eng = bioc_ranges_plugin()
#     np_eng = numpy_plugin()
    
#     # R Package Imports
#     se_pkg = renv.importr("SummarizedExperiment")
#     sce_pkg = renv.importr("SingleCellExperiment")
#     s4_pkg = renv.importr("S4Vectors")

#     # -------------------------------------------------
#     # Python -> R (SE / RSE / SCE)
#     # -------------------------------------------------
#     @eng.register_py(SummarizedExperiment)
#     @eng.register_py(RangedSummarizedExperiment)
#     @eng.register_py(SingleCellExperiment)
#     def _(e: Engine, x: SummarizedExperiment):
#         # 1. Convert Assays (Dict of matrices -> SimpleList of matrices)
#         # We use np_eng because assays are typically NumPy-backed matrices
#         assays_py = x.get_assays()
#         r_assays_dict = {k: np_eng.py2r(v) for k, v in assays_py.items()}
#         r_assays = s4_pkg.SimpleList(**r_assays_dict)
        
#         # 2. colData (BiocFrame -> S4 DataFrame)
#         r_col_data = bioc_eng.py2r(x.get_column_data())
        
#         # 3. rowData or rowRanges
#         r_row_ranges = r_NULL
#         r_row_data = r_NULL
        
#         if isinstance(x, (RangedSummarizedExperiment, SingleCellExperiment)):
#             # Use ranges_eng specifically for GRanges
#             r_row_ranges = ranges_eng.py2r(x.get_row_ranges())
#         else:
#             r_row_data = bioc_eng.py2r(x.get_row_data())

#         r.ro.r["print"]()
#         # 4. Construct the base SE or RSE
#         if r_row_ranges is not r_NULL:
#             r_obj = se_pkg.SummarizedExperiment(
#                 assays=r_assays,
#                 colData=r_col_data,
#                 rowRanges=r_row_ranges
#             )
#         else:
#             r_obj = se_pkg.SummarizedExperiment(
#                 assays=r_assays,
#                 colData=r_col_data,
#                 rowData=r_row_data
#             )

#         # 5. Handle SingleCellExperiment Specifics
#         if isinstance(x, SingleCellExperiment):
#             # Upcast to SCE in R
#             r_obj = sce_pkg.SingleCellExperiment(r_obj)
            
#             # Reduced Dims (Dict of arrays -> SimpleList)
#             red_dims_py = x.get_reduced_dims()
#             if red_dims_py:
#                 r_red_dims_dict = {k: np_eng.py2r(v) for k, v in red_dims_py.items()}
#                 r_red_dims = s4_pkg.SimpleList(**r_red_dims_dict)
#                 r_obj = sce_pkg.reducedDims_set(r_obj, value=r_red_dims)
            
#             # AltExps (Dict of SEs -> SimpleList)
#             alt_exps_py = x.get_alt_exps()
#             if alt_exps_py:
#                 # Recursive call using 'e' (the experiment engine)
#                 r_alts_dict = {k: e.py2r(v) for k, v in alt_exps_py.items()}
#                 r_alts = s4_pkg.SimpleList(**r_alts_dict)
#                 r_obj = sce_pkg.altExps_set(r_obj, value=r_alts)

#         # 6. Metadata (Dict/List -> list)
#         # We use 'e' here if base_plugin rules are registered, otherwise handle simply
#         meta_py = x.get_metadata()
#         if meta_py:
#             try:
#                 r_meta = e.py2r(meta_py)
#                 r_obj = s4_pkg.metadata_set(r_obj, value=r_meta)
#             except:
#                 pass # Fallback if no list/dict rules found
            
#         return r_obj

#     # -------------------------------------------------
#     # R -> Python (SE / RSE / SCE)
#     # -------------------------------------------------
#     @eng.register_s4("SummarizedExperiment")
#     @eng.register_s4("RangedSummarizedExperiment")
#     @eng.register_s4("SingleCellExperiment")
#     def _(e: Engine, x: Any):
#         r_classes = list(r["class"](x))
        
#         # 1. Convert Assays
#         r_assays = se_pkg.assays(x)
#         # SimpleList -> Dict of NumPy arrays
#         assays_py = {str(k): np_eng.r2py(v) for k, v in r_assays.items()}
        
#         # 2. colData
#         col_data_py = bioc_eng.r2py(se_pkg.colData(x))
        
#         # 3. rowData / rowRanges
#         # Use R generics to determine if it's ranged
#         r_row_ranges = se_pkg.rowRanges(x)
#         is_ranged = not isinstance(r_row_ranges, type(r_NULL))
        
#         row_ranges_py = ranges_eng.r2py(r_row_ranges) if is_ranged else None
#         row_data_py = bioc_eng.r2py(se_pkg.rowData(x))
        
#         # 4. Metadata
#         metadata_py = e.r2py(s4_pkg.metadata(x))

#         # 5. Handle SingleCellExperiment
#         if "SingleCellExperiment" in r_classes:
#             r_red_dims = sce_pkg.reducedDims(x)
#             red_dims_py = {str(k): np_eng.r2py(v) for k, v in r_red_dims.items()}
            
#             r_alt_exps = sce_pkg.altExps(x)
#             # Recursive call using 'e' for nested SEs
#             alt_exps_py = {str(k): e.r2py(v) for k, v in r_alt_exps.items()}
            
#             return SingleCellExperiment(
#                 assays=assays_py,
#                 row_ranges=row_ranges_py,
#                 col_data=col_data_py,
#                 metadata=metadata_py,
#                 reduced_dims=red_dims_py,
#                 alt_exps=alt_exps_py
#             )
            
#         if is_ranged:
#             return RangedSummarizedExperiment(
#                 assays=assays_py,
#                 row_ranges=row_ranges_py,
#                 col_data=col_data_py,
#                 metadata=metadata_py
#             )
            
#         return SummarizedExperiment(
#             assays=assays_py,
#             row_data=row_data_py,
#             col_data=col_data_py,
#             metadata=metadata_py
#         )

#     return eng