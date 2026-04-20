# --- bioc_ranges_plugin.py ---
from __future__ import annotations
from typing import Any
from bioc2ri.lazy_r_env import get_r_environment

__author__ = "MaximilianNuber"
__license__ = "MIT"

def bioc_ranges_plugin() -> "Engine":
    from iranges import IRanges
    from genomicranges import GenomicRanges
    from bioc2ri.engine import Engine
    from bioc2ri.numpy_plugin import numpy_plugin
    from bioc2ri import biocpy_plugin
    # We use these to ensure lists become R atomic vectors
    from biocutils import StringList
    
    from genomicranges import GenomicRanges, CompressedGenomicRangesList
    from compressed_lists import Partitioning
       
    # Correct way to get NULL in rpy2
    from rpy2.rinterface import NULL as r_NULL
    
    eng = Engine()
    renv = get_r_environment()
    r = renv.ro.r
    np_eng = numpy_plugin()
    
    ir_pkg = renv.importr("IRanges")
    gr_pkg = renv.importr("GenomicRanges")
    s4_pkg = renv.importr("S4Vectors")

    def _ensure_atomic_r(e: Engine, x: Any):
        """Ensures sequences become R atomic vectors, using the provided engine."""
        if x is None:
            return r_NULL
        if hasattr(x, "dtype"): # NumPy array
            return np_eng.py2r(x)
        if isinstance(x, list):
            # We wrap in StringList so the biocpy rules (which must be 
            # present in 'e') convert it to a character vector.
            return e.py2r(StringList(x))
        return e.py2r(x)

    # -------------------------------------------------
    # IRanges: Python -> R
    # -------------------------------------------------
    @eng.register_py(IRanges)
    def _(e: Engine, x: IRanges):
        r_start = np_eng.py2r(x.get_start())
        r_width = np_eng.py2r(x.get_width())
        
        # Use 'e' (the composite engine) for atomic conversion
        r_names = _ensure_atomic_r(biocpy_plugin(), x.get_names())

        r_ir = ir_pkg.IRanges(start=r_start, width=r_width, names=r_names)

        # Handle metadata columns
        mcols_py = x.get_mcols()
        if mcols_py is not None and len(mcols_py.get_column_names()) > 0:
            r_ir = r["mcols<-"](r_ir, value=biocpy_plugin().py2r(mcols_py))
            
        return r_ir

    # -------------------------------------------------
    # GenomicRanges: Python -> R
    # -------------------------------------------------
    @eng.register_py(GenomicRanges)
    def _(e: Engine, x: GenomicRanges):
        r_seqnames = _ensure_atomic_r(biocpy_plugin(), x.get_seqnames())
        ###
        # r_strand = _ensure_atomic_r(biocpy_plugin(), x.get_strand())
        
        # --- STRAND FIX ---
        # Decode BiocPy int8 codes [0, 1, 2] back to ["+", "*", "-"]
        # --- STRAND DECODING ---
        # BiocPy strands are codes: 0 (+), 1 (-), 2 (*)
        # Note: If your gr.get_strand() showed [0, 0] for ["+", "-"], 
        strand_map = ["+", "-", "*"]
        codes = x.get_strand()
        strand_strings = [strand_map[c] for c in codes]
        r_strand = biocpy_plugin().py2r(StringList(strand_strings))
        ###
        
        # This triggers the IRanges rule registered above
        r_ranges = e.py2r(x.get_ranges()) 
        
        # The R source shows that GRanges@ranges cannot have mcols.
        # If the IRanges we just converted has them, R will throw an error.
        if not isinstance(r["mcols"](r_ranges), type(r_NULL)):
             r_ranges = r["mcols<-"](r_ranges, value=r_NULL)

        r_gr = gr_pkg.GRanges(
            seqnames=r_seqnames,
            ranges=r_ranges,
            strand=r_strand
        )
        
        mcols_py = x.get_mcols()
        if mcols_py is not None and len(mcols_py.get_column_names()) > 0:
            r_gr = r["mcols<-"](r_gr, value=biocpy_plugin().py2r(mcols_py))
            
        return r_gr

    # ... [R -> Python methods] ...
    # @eng.register_s4("IRanges")
    # def _(e: Engine, x: Any):
    #     start = np_eng.r2py(r["start"](x))
    #     width = np_eng.r2py(r["width"](x))
    #     r_names = r["names"](x)
    #     names = list(r_names) if not isinstance(r_names, type(r_NULL)) else None
    #     r_mcols = s4_pkg.mcols(x)
    #     mcols = e.r2py(r_mcols) if not isinstance(r_mcols, type(r_NULL)) and r["ncol"](r_mcols)[0] > 0 else None
    #     return IRanges(start=start, width=width, names=names, mcols=mcols)
    @eng.register_s4("IRanges")
    @eng.register_s4("NormalIRanges")
    def _(e: Engine, x: Any):
        # Coordinates stay with numpy engine
        start = np_eng.r2py(r["start"](x))
        width = np_eng.r2py(r["width"](x))
        
        # Strings/Factors MUST use the composite engine 'e'
        r_names = r["names"](x)
        names = list(e.r2py(r_names)) if not isinstance(r_names, type(r_NULL)) else None
        
        r_mcols = s4_pkg.mcols(x)
        mcols = biocpy_plugin().r2py(r_mcols) if not isinstance(r_mcols, type(r_NULL)) and r["ncol"](r_mcols)[0] > 0 else None
        
        return IRanges(start=start, width=width, names=names, mcols=mcols)

    # @eng.register_s4("GRanges")
    # def _(e: Engine, x: Any):
    #     seqnames = np_eng.r2py(r["as.character"](r["seqnames"](x)))
    #     ranges = e.r2py(r["ranges"](x))
    #     strand = np_eng.r2py(r["as.character"](r["strand"](x)))
    #     r_mcols = s4_pkg.mcols(x)
    #     mcols = e.r2py(r_mcols) if not isinstance(r_mcols, type(r_NULL)) and r["ncol"](r_mcols)[0] > 0 else None
    #     return GenomicRanges(seqnames=seqnames, ranges=ranges, strand=strand, mcols=mcols)
    @eng.register_s4("GRanges")
    def _(e: Engine, x: Any):
        # Use 'e' for strings to get StringList/list, then cast to list for the constructor
        seqnames = list(e.r2py(r["as.character"](r["seqnames"](x))))
        ranges = e.r2py(r["ranges"](x))
        strand = list(e.r2py(r["as.character"](r["strand"](x))))
        
        r_mcols = s4_pkg.mcols(x)
        mcols = biocpy_plugin().r2py(r_mcols) if not isinstance(r_mcols, type(r_NULL)) and r["ncol"](r_mcols)[0] > 0 else None
        
        return GenomicRanges(
            seqnames=seqnames, 
            ranges=ranges, 
            strand=strand, 
            mcols=mcols
        )
        
    @eng.register_py(CompressedGenomicRangesList)
    def _(e: Engine, x: CompressedGenomicRangesList):
        # 1. Convert the flat 'unlist_data' (GenomicRanges)
        # Use THIS engine (e) recursively to hit the GenomicRanges rule
        r_unlist_data = e.py2r(x._unlist_data)
        
        # 2. Convert Partitioning ends
        r_ends = np_eng.py2r(x._partitioning.get_ends())
        
        # 3. Use R's 'relist' to build the CompressedGRangesList
        # PartitioningByEnd is the R equivalent of the Partitioning object
        r_partitioning = r["PartitioningByEnd"](r_ends)
        
        if x.names is not None:
             r_partitioning = r["names<-"](r_partitioning, value=biocpy_plugin().py2r(StringList(x.names)))

        return r["relist"](r_unlist_data, skeleton=r_partitioning)

    # -------------------------------------------------
    # R -> Python
    # -------------------------------------------------

    @eng.register_s4("CompressedGRangesList")
    @eng.register_s4("GRangesList")
    def _(e: Engine, x: Any):
        # 1. Get the flat 'unlist_data' from R
        # In R: unlist(x) returns the underlying GRanges
        r_unlist_data = r["unlist"](x)
        py_unlist_data = e.r2py(r_unlist_data)
        
        # 2. Get the Partitioning ends
        # In R: PartitioningByEnd(x) or cumsum(elementNROWS(x))
        r_partitioning = r["PartitioningByEnd"](x)
        r_ends = r["end"](r_partitioning)
        py_ends = np_eng.r2py(r_ends)
        
        # 3. Handle Names
        r_names = r["names"](x)
        py_names = list(e.r2py(r_names)) if not isinstance(r_names, type(r_NULL)) else None
        
        # 4. Construct the Python CompressedGenomicRangesList
        partitioning = Partitioning(ends=py_ends, names=py_names)
        return CompressedGenomicRangesList(py_unlist_data, partitioning)

    return eng



# bioc_ranges_plugin.py

# # --- bioc_ranges_plugin.py ---
# from __future__ import annotations
# from typing import Any
# from bioc2ri.lazy_r_env import get_r_environment

# __author__ = "MaximilianNuber"
# __license__ = "MIT"


# def bioc_ranges_plugin() -> "Engine":
#     from iranges import IRanges
#     from genomicranges import GenomicRanges
#     from bioc2ri.engine import Engine
#     from bioc2ri.numpy_plugin import numpy_plugin
#     from bioc2ri import biocpy_plugin
#     # We use these to ensure lists become R atomic vectors
#     from biocutils import StringList
    
#     eng = Engine()
#     renv = get_r_environment()
#     r = renv.ro.r
#     np_eng = numpy_plugin()
    
#     ir_pkg = renv.importr("IRanges")
#     gr_pkg = renv.importr("GenomicRanges")
#     s4_pkg = renv.importr("S4Vectors")

#     def _ensure_atomic_r(e: Engine, x: Any):
#         """Helper to ensure a sequence becomes an R atomic vector, not an R list."""
#         if x is None:
#             return r["NULL"]
#         if hasattr(x, "dtype"): # NumPy array
#             return np_eng.py2r(x)
#         if isinstance(x, list):
#             # Force conversion to StringList so biocpy_plugin makes it a character vector
#             return e.py2r(StringList(x))
#         return e.py2r(x)

#     # -------------------------------------------------
#     # IRanges: Python -> R
#     # -------------------------------------------------
#     @eng.register_py(IRanges)
#     def _(e: Engine, x: IRanges):
#         r_start = np_eng.py2r(x.get_start())
#         r_width = np_eng.py2r(x.get_width())
        
#         # Use helper to ensure names become a character vector
#         r_names = _ensure_atomic_r(biocpy_plugin(), x.get_names())

#         r_ir = ir_pkg.IRanges(start=r_start, width=r_width, names=r_names)

#         mcols_py = x.get_mcols()
#         if mcols_py is not None and len(mcols_py.get_column_names()) > 0:
#             r_ir = r["mcols<-"](r_ir, value=e.py2r(mcols_py))
            
#         return r_ir

#     # -------------------------------------------------
#     # IRanges: R -> Python
#     # -------------------------------------------------
#     @eng.register_s4("IRanges")
#     @eng.register_s4("NormalIRanges")
#     def _(e: Engine, x: Any):
#         start = np_eng.r2py(r["start"](x))
#         width = np_eng.r2py(r["width"](x))
        
#         r_names = r["names"](x)
#         names = list(r_names) if not isinstance(r_names, type(r("NULL"))) else None
        
#         r_mcols = s4_pkg.mcols(x)
#         # Only convert mcols if it's not a zero-column DataFrame
#         mcols = None
#         if not isinstance(r_mcols, type(r["NULL"])) and r["ncol"](r_mcols)[0] > 0:
#             mcols = e.r2py(r_mcols)
        
#         return IRanges(start=start, width=width, names=names, mcols=mcols)

#     # -------------------------------------------------
#     # GenomicRanges: Python -> R
#     # -------------------------------------------------
#     @eng.register_py(GenomicRanges)
#     def _(e: Engine, x: GenomicRanges):
#         # Seqnames and Strand MUST be atomic vectors in R
#         r_seqnames = _ensure_atomic_r(biocpy_plugin(), x.get_seqnames())
#         r_strand = _ensure_atomic_r(biocpy_plugin(), x.get_strand())
        
#         # Ranges is a recursive call
#         r_ranges = e.py2r(x.get_ranges()) 
        
#         r_gr = gr_pkg.GRanges(
#             seqnames=r_seqnames,
#             ranges=r_ranges,
#             strand=r_strand
#         )
        
#         mcols_py = x.get_mcols()
#         if mcols_py is not None and len(mcols_py.get_column_names()) > 0:
#             r_gr = r["mcols<-"](r_gr, value=e.py2r(mcols_py))
            
#         return r_gr

    # -------------------------------------------------
    # GenomicRanges: R -> Python
    # -------------------------------------------------
    # @eng.register_s4("GRanges")
    # def _(e: Engine, x: Any):
    #     seqnames = np_eng.r2py(r["as.character"](r["seqnames"](x)))
    #     ranges = e.r2py(r["ranges"](x))
    #     strand = np_eng.r2py(r["as.character"](r["strand"](x)))
        
    #     r_mcols = s4_pkg.mcols(x)
    #     mcols = None
    #     if not isinstance(r_mcols, type(r["NULL"])) and r["ncol"](r_mcols)[0] > 0:
    #         mcols = e.r2py(r_mcols)
        
    #     return GenomicRanges(
    #         seqnames=seqnames,
    #         ranges=ranges,
    #         strand=strand,
    #         mcols=mcols
    #     )

    # return eng

# def bioc_ranges_plugin() -> "Engine":
#     # Python imports (internal to stay lazy)
#     from iranges import IRanges
#     from genomicranges import GenomicRanges
#     from bioc2ri.engine import Engine
#     from bioc2ri.numpy_plugin import numpy_plugin
    
#     # Initialize Engine and R Environment
#     eng = Engine()
#     renv = get_r_environment()
#     r = renv.ro.r
#     np_eng = numpy_plugin()
    
#     # R Package Imports via lazy env
#     ir_pkg = renv.importr("IRanges")
#     gr_pkg = renv.importr("GenomicRanges")
#     s4_pkg = renv.importr("S4Vectors")

#     # -------------------------------------------------
#     # IRanges: Python -> R
#     # -------------------------------------------------
#     @eng.register_py(IRanges)
#     def _(e: Engine, x: IRanges):
#         # Use numpy_plugin for the core coordinate arrays
#         r_start = np_eng.py2r(x.get_start())
#         r_width = np_eng.py2r(x.get_width())
        
#         # Names: convert to list first so base_plugin can handle strings
#         names_py = x.get_names()
#         r_names = e.py2r(list(names_py)) if names_py is not None else r["NULL"]

#         # Construct the IRanges object
#         r_ir = ir_pkg.IRanges(start=r_start, width=r_width, names=r_names)

#         # Assign mcols if they exist
#         mcols_py = x.get_mcols()
#         if mcols_py is not None and len(mcols_py) > 0:
#             r_mcols = e.py2r(mcols_py)
#             # CRITICAL: 'value' must be named to satisfy the R generic (x, ..., value)
#             r_ir = r["mcols<-"](r_ir, value=r_mcols)
            
#         return r_ir

#     # -------------------------------------------------
#     # IRanges: R -> Python
#     # -------------------------------------------------
#     @eng.register_s4("IRanges")
#     @eng.register_s4("NormalIRanges")
#     def _(e: Engine, x: Any):
#         # Extract data using R getters and the numpy engine
#         start = np_eng.r2py(r["start"](x))
#         width = np_eng.r2py(r["width"](x))
        
#         r_names = r["names"](x)
#         names = list(r_names) if not isinstance(r_names, type(r["NULL"])) else None
        
#         r_mcols = s4_pkg.mcols(x)
#         mcols = e.r2py(r_mcols) if not isinstance(r_mcols, type(r["NULL"])) else None
        
#         return IRanges(start=start, width=width, names=names, mcols=mcols)

#     # -------------------------------------------------
#     # GenomicRanges: Python -> R
#     # -------------------------------------------------
#     @eng.register_py(GenomicRanges)
#     def _(e: Engine, x: GenomicRanges):
#         # Convert internal components
#         r_seqnames = np_eng.py2r(x.get_seqnames())
#         r_ranges = e.py2r(x.get_ranges()) # Recursively hits the IRanges rule above
#         r_strand = np_eng.py2r(x.get_strand())
        
#         r_gr = gr_pkg.GRanges(
#             seqnames=r_seqnames,
#             ranges=r_ranges,
#             strand=r_strand
#         )
        
#         # Handle metadata
#         mcols_py = x.get_mcols()
#         if mcols_py is not None and len(mcols_py) > 0:
#             r_mcols = e.py2r(mcols_py)
#             r_gr = r["mcols<-"](r_gr, value=r_mcols)
            
#         return r_gr

#     # -------------------------------------------------
#     # GenomicRanges: R -> Python
#     # -------------------------------------------------
#     @eng.register_s4("GRanges")
#     def _(e: Engine, x: Any):
#         # Extract using R generics
#         seqnames = np_eng.r2py(r["as.character"](r["seqnames"](x)))
#         ranges = e.r2py(r["ranges"](x))
#         strand = np_eng.r2py(r["as.character"](r["strand"](x)))
        
#         r_mcols = s4_pkg.mcols(x)
#         mcols = e.r2py(r_mcols) if not isinstance(r_mcols, type(r["NULL"])) else None
        
#         return GenomicRanges(
#             seqnames=seqnames,
#             ranges=ranges,
#             strand=strand,
#             mcols=mcols
#         )

#     return eng

# from __future__ import annotations
# from typing import Any
# from .engine import Engine
# # from rpy2.robjects import r, vectors as rv
# # from rpy2.robjects.packages import importr
# from bioc2ri.lazy_r_env import get_r_environment


# __author__ = "MaximilianNuber"
# __license__ = "MIT"

# def bioc_ranges_plugin() -> Engine:
    
    
#     # Python imports
#     from iranges import IRanges
#     from genomicranges import GenomicRanges

#     eng = Engine()
#     # Instantiate the cached numpy engine once for internal use
#     from bioc2ri import numpy_plugin
#     renv = get_r_environment()
#     r = renv.ro.r
#     np_eng = numpy_plugin()
    
#     ir_pkg = renv.importr("IRanges")
#     gr_pkg = renv.importr("GenomicRanges")
#     s4_pkg = renv.importr("S4Vectors")

#     # -------------------------------------------------
#     # IRanges: Python -> R
#     # -------------------------------------------------
#     @eng.register_py(IRanges)
#     def _(e: Engine, x: IRanges):
#         # Explicitly convert NumPy arrays via np_eng
#         # This bypasses rpy2's internal 'py2rpy' dispatch
#         r_start = np_eng.py2r(x.get_start())
#         r_width = np_eng.py2r(x.get_width())
        
#         # Names are usually a list of strings or None
#         # We can use the engine 'e' (likely base_plugin) for these
#         r_names = e.py2r(x.get_names()) if x.get_names() is not None else renv.ro.NULL
        
#         r_ir = ir_pkg.IRanges(start=r_start, width=r_width, names=r_names)
        
#         if x.get_mcols() is not None and len(x.get_mcols()) > 0:
#             # r_ir = s4_pkg.mcols_set(r_ir, e.py2r(x.get_mcols()))
#             r_ir = renv.ro.r["mcols<-"](r_ir, value=e.py2r(x.get_mcols()))
            
#         return r_ir

#     # -------------------------------------------------
#     # IRanges: R -> Python
#     # -------------------------------------------------
#     @eng.register_s4("IRanges")
#     def _(e: Engine, x: Any):
#         r = renv.ro.r
#         # Extract core vectors
#         start = list(r["as.integer"](r["start"](x)))
#         width = list(r["as.integer"](r["width"](x)))
#         names = list(r["names"](x)) if not isinstance(r["names"](x), type(renv.ro.NULL)) else None
        
#         # Handle mcols
#         mcols_r = s4_pkg.mcols(x)
#         mcols_py = e.r2py(mcols_r) if not isinstance(mcols_r, type(renv.ro.NULL)) else None
        
#         return IRanges(start=start, width=width, names=names, mcols=mcols_py)

#     # -------------------------------------------------
#     # GenomicRanges: Python -> R
#     # -------------------------------------------------
#     @eng.register_py(GenomicRanges)
#     def _(e: Engine, x: GenomicRanges):
#         # 1. Seqnames (often a Factor-like or StringList in BiocPy)
#         # We use np_eng if it's an ndarray, otherwise the current engine 'e'
#         sn = x.get_seqnames()
#         r_seqnames = np_eng.py2r(sn) if hasattr(sn, "dtype") else e.py2r(sn)
        
#         # 2. Ranges (Recursive call to the IRanges rule above)
#         r_ranges = e.py2r(x.get_ranges())
        
#         # 3. Strand
#         st = x.get_strand()
#         r_strand = np_eng.py2r(st) if hasattr(st, "dtype") else e.py2r(st)
        
#         r_gr = gr_pkg.GRanges(
#             seqnames=r_seqnames,
#             ranges=r_ranges,
#             strand=r_strand
#         )
        
#         if x.get_mcols() is not None and len(x.get_mcols()) > 0:
#             # r_gr = s4_pkg.mcols_set(r_gr, e.py2r(x.get_mcols()))
#             r_gr = renv.ro.r["mcols<-"](r_gr, value=e.py2r(x.get_mcols()))
            
#         return r_gr

#     # -------------------------------------------------
#     # GenomicRanges: R -> Python
#     # -------------------------------------------------
#     @eng.register_s4("GRanges")
#     def _(e: Engine, x: Any):
#         r = renv.ro.r
#         # Extract R components
#         seqnames_r = r["as.character"](r["seqnames"](x))
#         ranges_r = r["ranges"](x)
#         strand_r = r["as.character"](r["strand"](x))
#         mcols_r = s4_pkg.mcols(x)
        
#         # Convert to Python via engine
#         seqnames = e.r2py(seqnames_r)
#         ranges = e.r2py(ranges_r) # Returns IRanges object
#         strand = e.r2py(strand_r)
#         mcols = e.r2py(mcols_r) if not isinstance(mcols_r, type(renv.ro.NULL)) else None
        
#         return GenomicRanges(
#             seqnames=seqnames,
#             ranges=ranges,
#             strand=strand,
#             mcols=mcols
#         )

#     return eng