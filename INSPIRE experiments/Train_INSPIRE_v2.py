import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import umap
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import os
import logging
import warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys

path_to_inspire_parent = "/opt/INSPIRE/"

if path_to_inspire_parent not in sys.path:
    sys.path.append(path_to_inspire_parent)

import INSPIRE

def load_data(data_dir: str, file_name: str):
    """
    Load a single .h5ad file from a directory.
    
    Args:
        data_dir: Directory containing the .h5ad file
        file_name: Name of the file to load
    
    Returns:
        AnnData object
    """
    filepath = os.path.join(data_dir, file_name)
    logger.info(f"Loading {filepath}")
    adata = sc.read_h5ad(filepath)
    
    # Ensure spatial coordinates are in obsm['spatial']
    if 'spatial' not in adata.obsm:
        if 'x' in adata.obs.columns and 'y' in adata.obs.columns:
            adata.obsm['spatial'] = adata.obs[['x', 'y']].to_numpy()
        else:
            logger.warning(f"No spatial coordinates found for {file_name}")
            adata.obsm['spatial'] = np.zeros((adata.n_obs, 2))
    
    # Add file name to obs for tracking
    adata.obs['sample_id'] = file_name.replace('.h5ad', '')
    
    return adata

diff_age_slices = ["T1098_MT69_3M_Visual_Sub-CTX_V12_b100_filtered.h5ad", "T1090_MT65_P0_Visual_Sub-CTX_b100_filtered.h5ad", "T1091_MT64_P32_Visual_Sub-CTX_V12_b100_filtered.h5ad"]

data_dir = "/opt/INSPIRE/data/"

n_sp_arr = [20,40,60,80,100]
for n in n_sp_arr:
    ads_3ages=[]
    for f in diff_age_slices:
        adata = load_data(data_dir, f)
        adata.var_names_make_unique()
        ads_3ages.append(adata)
    adata_st_list_3ages, adata_full_3ages = INSPIRE.utils.preprocess(adata_st_list=ads_3ages,
                                                     num_hvgs=2000,
                                                     min_genes_qc=50,
                                                     min_cells_qc=50,
                                                     spot_size=100)
    for i in range(len(ads_3ages)):
        ads_3ages[i].obs["library_size"]=ads_3ages[i].obs["library_size"].astype("int64")
    for i in range(len(ads_3ages)):
        ads_3ages[i].obsm["count"]=ads_3ages[i].obsm["count"].astype("int32")
    adata_st_list_3ages = INSPIRE.utils.build_graph_LGCN(adata_st_list=ads_3ages,
                                               rad_cutoff_list=[1,1,1])
    print(f'Training for n={n}')
    model = INSPIRE.model.Model_LGCN(adata_st_list=ads_3ages,
                                n_spatial_factors=n, batch_size=2048,
                                n_training_steps=10000)
    model.train(ads_3ages)

    adata_full2, basis_df2 = model.eval(ads_3ages, adata_full_3ages)
    basis2 = np.array(basis_df2.values)

    name_adata = f"adata_full_n{n}.h5ad"
    name_basis = f'basis_{n}.npy'
    adata_full2.write(name_adata)
    np.save(name_basis, basis2)