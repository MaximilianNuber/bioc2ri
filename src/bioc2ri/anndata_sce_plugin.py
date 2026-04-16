from functools import cache
import biocframe
import anndata as ad
from rpy2.robjects import r, vectors as rv
from rpy2.robjects.packages import importr

from .engine import Engine
from .biocpy_plugin import biocpy_plugin
from .scipy_sparse_plugin import scipy_sparse_plugin
from .numpy_plugin import numpy_plugin
from .pandas_plugin import pandas_plugin

@cache
def anndata2sce_plugin():
    # 1. Initialize isolated engines
    bioc_eng = biocpy_plugin()
    sp_eng = scipy_sparse_plugin()
    np_eng = numpy_plugin()
    pd_eng = pandas_plugin()

    # 2. Load R libraries
    SCE = importr("SingleCellExperiment")
    SummarizedExperiment = importr("SummarizedExperiment")
    
    r_get_item = r["[["]

    eng = Engine()

    # ---------- Python -> R: AnnData to SingleCellExperiment ----------

    @eng.register_py(ad.AnnData)
    def _(e, adata: ad.AnnData):
        # Helper to transpose and convert
        def convert_and_transpose(mtx):
            # AnnData (Cells x Genes) -> R (Genes x Cells)
            t_mtx = mtx.T 
            try:
                return sp_eng.py2r(t_mtx)
            except TypeError:
                return np_eng.py2r(t_mtx)
                
        # 1. Assays: Use "X" for .X and preserve layer names exactly
        assays_dict = {}
        if adata.X is not None:
            assays_dict["X"] = convert_and_transpose(adata.X)
        for name, data in adata.layers.items():
            assays_dict[str(name)] = convert_and_transpose(data)
        r_assays = rv.ListVector(assays_dict)

        # 2. Metadata: obs -> colData, var -> rowData
        # Our pd_eng handles the data.frame conversion correctly
        r_col_data = pd_eng.py2r(adata.obs)
        r_row_data = pd_eng.py2r(adata.var)

        # 3. Reduced Dims: AnnData.obsm is (Cells x Dims)
        # R's reducedDims also expects (Cells x Dims), so NO transpose here.
        rd_dict = {str(k): np_eng.py2r(v) for k, v in adata.obsm.items()}
        r_reduced_dims = rv.ListVector(rd_dict)

        # 4. Assembly
        return SCE.SingleCellExperiment(
            assays=r_assays,
            colData=r_col_data, 
            rowData=r_row_data,
            reducedDims=r_reduced_dims
        )

    # ---------- R -> Python: SingleCellExperiment to AnnData ----------

    # ---------- R -> Python ----------
    @eng.register_s4("SingleCellExperiment")
    def _(e, x):
        r_assays = SummarizedExperiment.assays(x)
        assay_names = list(r_assays.names)
        
        def r_to_py_mtx(r_obj):
            try:
                return sp_eng.r2py(r_obj).T
            except:
                return np_eng.r2py(r_obj).T

        # 1. Map all R assays into layers only
        layers = {}
        for name in assay_names:
            # Use the explicit R [[ operator for the SimpleList
            assay_r = r_get_item(r_assays, str(name))
            layers[str(name)] = r_to_py_mtx(assay_r)
            
        r_as_data_frame = r["as.data.frame"]
        
        r_cd = r_as_data_frame(SummarizedExperiment.colData(x))
        r_rd = r_as_data_frame(SummarizedExperiment.rowData(x))
        
        # 2. Extract metadata
        obs = pd_eng.r2py(r_cd)
        var = pd_eng.r2py(r_rd)
        
        obs_names = r["rownames"](r_cd)
        obs_names = list(obs_names) if obs_names is not None else None
        
        var_names = r["rownames"](r_rd)
        var_names = list(var_names)
        
        obs.index = obs_names
        var.index = var_names


        # 3. Extract reduced dims
        r_rd = SCE.reducedDims(x)
        obsm = {}
        if not isinstance(r_rd, type(None)):
            for name in list(r_rd.names):
                # Using [[ here as well for safety with List-like S4
                dim_r = r_get_item(r_rd, str(name))
                obsm[str(name)] = np_eng.r2py(dim_r)

        # Create AnnData with empty X, relying on layers
        print(obs.shape)
        print(var.shape)
        
        adata = ad.AnnData(
            obs = obs,
            var = var,
            obsm = obsm,
            layers = layers
        )

        return adata
    

    return eng