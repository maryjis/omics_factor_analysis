"""Run INSPIRE LGCN on selected annotated Stereo-seq slices.

This script keeps the INSPIRE model unchanged and replaces the tutorial's dense
pairwise-distance preprocessing with a sparse KD-tree implementation suitable
for the local high-resolution slices.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.spatial import cKDTree
import umap
import sys

path_to_inspire_parent = "/opt/INSPIRE/"

if path_to_inspire_parent not in sys.path:
    sys.path.append(path_to_inspire_parent)
import INSPIRE

DEFAULT_INPUTS = [
     "/opt/INSPIRE/data/T1098_MT69_3M_Visual_Sub-CTX_V12_b100_filtered.h5ad",
    "/opt/INSPIRE/data/T1090_MT65_P0_Visual_Sub-CTX_b100_filtered.h5ad",
    "/opt/INSPIRE/data/T1091_MT64_P32_Visual_Sub-CTX_V12_b100_filtered.h5ad"]

METADATA = ["/opt/INSPIRE/meta/data/annotation/v2_251209/T1098_MT69_3M_Visual_Sub-CTX_V12_b100_anno.csv",
       "/opt/INSPIRE/meta/data/annotation/v2_251209/T1090_MT65_P0_Visual_Sub-CTX_b100_anno.csv",
       "/opt/INSPIRE/meta/data/annotation/v2_251209/T1091_MT64_P32_Visual_Sub-CTX_V12_b100_anno.csv"]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", default=None, help="Input h5ad. Repeatable.")
    parser.add_argument("--meta", action="append", default=None)
    parser.add_argument("--output-dir", default="output/inspire_lgcn_T1098_T1090_T1091_v3")
    parser.add_argument("--num-hvgs", type=int, default=3000)
    parser.add_argument("--min-genes-qc", type=int, default=50)
    parser.add_argument("--min-cells-qc", type=int, default=50)
    parser.add_argument("--rad-coef", type=float, default=1.15)
    parser.add_argument("--n-spatial-factors", type=int, default=16)
    parser.add_argument("--training-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--spot-size", type=float, default=2.0)
    parser.add_argument("--min-concat-dist", type=float, default=500.0)
    parser.add_argument("--max-point-size", type=float, default=3.5)
    return parser.parse_args()


def sample_name(path: str) -> str:
    name = Path(path).name
    suffix = "_b100_filtered.h5ad"
    return name[: -len(suffix)] if name.endswith(suffix) else Path(path).stem

def sample_short_name_from_ann(path: str) -> str:
    name = Path(path).name
    suffix = "_b100_anno.csv"
    return name[:5] if name.endswith(suffix) else Path(path).stem

def ensure_count_sample_is_count_like(adata: ad.AnnData, sample: str) -> None:
    x = adata.X[: min(200, adata.n_obs), : min(200, adata.n_vars)]
    arr = x.toarray() if sp.issparse(x) else np.asarray(x)
    vals = arr[arr != 0]
    if vals.size == 0:
        print(f"[WARN] {sample}: sampled X block contains no nonzero values.")
        return
    integer_like = bool(np.allclose(vals, np.round(vals), atol=1e-6))
    print(
        f"[INFO] {sample}: sampled nonzero X min={vals.min():.4g}, "
        f"max={vals.max():.4g}, integer_like={integer_like}"
    )


def load_and_qc(path: str, sample_meta: str, args: argparse.Namespace) -> ad.AnnData:
    sample = sample_name(path)
    print(f"[INFO] Loading {sample}: {path}")
    adata = sc.read_h5ad(path)
    adata.var_names_make_unique()
    ensure_count_sample_is_count_like(adata, sample)

    keep_gene = ~np.asarray(pd.Index(adata.var_names).isna())
    mt_mask = np.asarray(adata.var_names.str.upper().str.startswith("MT-"))
    keep_gene &= ~mt_mask
    adata = adata[:, keep_gene].copy()

    # add metadata
   # print(sample_meta)
   # print(adata.obs.index[:10])
    short_name = sample_short_name_from_ann(sample_meta)
   # print(short_name)
    meta = pd.read_csv(sample_meta)
   # spots = [short_name + '_' + ind for ind in adata.obs.index]
   # print(spots[:10])
   # reg = meta.loc[meta['chip_cell'].isin(np.array(spots)),["region"]]
   # print(reg[:10])
   # lay = meta.loc[meta['chip_cell'].isin(np.array(spots)),["manual_layer"]]
   # print(lay[:10])
   # adata.obs['region'] = np.array(reg)
   # adata.obs['manual_layer'] = np.array(lay)

    adata.obs['temp_spot_id'] = [short_name + '_' + ind for ind in adata.obs.index]
    original_index = adata.obs.index.copy()
    # Merge
    adata.obs = adata.obs.merge(
        meta[['chip_cell', 'region', 'manual_layer']],
        left_on='temp_spot_id',
        right_on='chip_cell',
        how='left'
    ).drop(columns=['temp_spot_id', 'chip_cell'])
    adata.obs.index = original_index

    print(f"[INFO] {sample}: before QC {adata.shape}")
    sc.pp.filter_cells(adata, min_genes=args.min_genes_qc)
    sc.pp.filter_genes(adata, min_cells=args.min_cells_qc)
    print(f"[INFO] {sample}: after QC {adata.shape}")

    if "spatial" not in adata.obsm:
        if {"x", "y"}.issubset(adata.obs.columns):
            adata.obsm["spatial"] = adata.obs[["x", "y"]].to_numpy(dtype=float)
        elif {"spatial_1", "spatial_2"}.issubset(adata.obs.columns):
            adata.obsm["spatial"] = adata.obs[["spatial_1", "spatial_2"]].to_numpy(dtype=float)
        else:
            raise ValueError(f"{sample} has no obsm['spatial'] or x/y columns.")
    adata.obsm["spatial"] = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    adata.obs["sample_id"] = sample
    adata.obs["spot_id_original"] = adata.obs_names.astype(str)
    return adata
#def add_meta(METADATA, adata_list):
#    for i, sample_meta in enumerate(METADATA):
#        short_name = sample_short_name_from_ann(sample_meta)
#        meta = pd.read_csv(sample_meta)
#        spots = [short_name + '_' + ind[:-2] for ind in adata_list[i].obs.index]
#        reg = meta.loc[meta['chip_cell'].isin(np.array(spots)),["region"]]
#        lay = meta.loc[meta['chip_cell'].isin(np.array(spots)),["manual_layer"]]
#        adata_list[i].obs['region'] = np.array(reg)
#        adata_list[i].obs['manual_layer'] = np.array(lay)
#    return adata_list

def find_shared_hvgs(adata_list: list[ad.AnnData], num_hvgs: int) -> list[str]:
    hvgs_shared: pd.Index | None = None
    for i, adata in enumerate(adata_list):
        sample = adata.obs["sample_id"].iloc[0]
        print(f"[INFO] Finding HVGs for {sample}")
        tmp = adata.copy()
        sc.pp.highly_variable_genes(tmp, flavor="seurat_v3", n_top_genes=num_hvgs)
        hvgs = tmp.var.loc[tmp.var["highly_variable"]].sort_values("highly_variable_rank").index
        hvgs_shared = hvgs if hvgs_shared is None else hvgs_shared.intersection(hvgs)
        print(f"[INFO] {sample}: {len(hvgs)} HVGs, shared_so_far={len(hvgs_shared)}")
    hvgs_out = sorted(map(str, hvgs_shared))
    if len(hvgs_out) < 500:
        raise RuntimeError(f"Too few shared HVGs across slices: {len(hvgs_out)}")
    print(f"[INFO] Shared HVGs: {len(hvgs_out)}")
    return hvgs_out


def row_sum(x) -> np.ndarray:
    sums = np.asarray(x.sum(axis=1)).reshape(-1)
    sums[sums <= 0] = 1.0
    return sums.astype(np.float32)


def sparse_log_normalize_hvgs(adata: ad.AnnData, hvgs: list[str]) -> np.ndarray:
    total = row_sum(adata.X)
    counts = adata[:, hvgs].X.copy()
    if sp.issparse(counts):
        counts = counts.tocsr().astype(np.float32)
        norm = counts.multiply((1e4 / total)[:, None]).tocsr()
        norm.data = np.log1p(norm.data)
        return norm.toarray().astype(np.float32, copy=False)
    counts = np.asarray(counts, dtype=np.float32)
    return np.log1p(counts * (1e4 / total)[:, None]).astype(np.float32, copy=False)


def counts_hvgs_dense(adata: ad.AnnData, hvgs: list[str]) -> np.ndarray:
    counts = adata[:, hvgs].X.copy()
    if sp.issparse(counts):
        return counts.toarray().astype(np.float32, copy=False)
    return np.asarray(counts, dtype=np.float32)


def normalized_radius_graph(coords: np.ndarray, rad_coef: float) -> tuple[sp.csr_matrix, float, float]:
    tree = cKDTree(coords)
    nn_dist, _ = tree.query(coords, k=min(2, coords.shape[0]))
    if nn_dist.ndim == 1:
        positive = nn_dist[nn_dist > 0]
    else:
        positive = nn_dist[:, 1]
        positive = positive[positive > 0]
    if positive.size == 0:
        raise RuntimeError("Could not estimate nearest-neighbor distance from coordinates.")
    base_dist = float(np.median(positive))
    radius = base_dist * rad_coef

    pairs = np.array(list(tree.query_pairs(radius)), dtype=np.int64)
    n = coords.shape[0]
    if pairs.size == 0:
        rows = np.arange(n, dtype=np.int64)
        cols = np.arange(n, dtype=np.int64)
    else:
        rows = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(n, dtype=np.int64)])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(n, dtype=np.int64)])
    data = np.ones(rows.shape[0], dtype=np.float32)
    graph = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

    deg = np.asarray(graph.sum(axis=1)).reshape(-1)
    deg[deg <= 0] = 1.0
    inv_sqrt = np.power(deg, -0.5).astype(np.float32)
    graph = sp.diags(inv_sqrt) @ graph @ sp.diags(inv_sqrt)
    avg_neighbors = float(np.asarray((graph > 0).sum(axis=1)).reshape(-1).mean() - 1.0)
    return graph.tocsr(), radius, avg_neighbors


def prepare_inspire_inputs(
    adata_list: list[ad.AnnData], hvgs: list[str], args: argparse.Namespace
) -> tuple[list[ad.AnnData], ad.AnnData, pd.DataFrame]:
    st_list: list[ad.AnnData] = []
    graph_rows = []

    for i, adata in enumerate(adata_list):
        sample = adata.obs["sample_id"].iloc[0]
        print(f"[INFO] Preparing LGCN inputs for {sample}")
        counts = counts_hvgs_dense(adata, hvgs)
        library_size = counts.sum(axis=1).astype(np.float32)
        library_size[library_size <= 0] = 1.0
        lognorm = sparse_log_normalize_hvgs(adata, hvgs)
        graph, radius, avg_neighbors = normalized_radius_graph(adata.obsm["spatial"], args.rad_coef)
        ax = (graph @ lognorm).astype(np.float32, copy=False)
        node_features = np.concatenate([lognorm, ax], axis=1).astype(np.float32, copy=False)

        st = ad.AnnData(np.zeros((adata.n_obs, len(hvgs)), dtype=np.float32))
        st.var_names = hvgs
        st.obs_names = [f"{idx}-{i}" for idx in adata.obs_names.astype(str)]
        st.obs["slice"] = i
        st.obs["slice"] = st.obs["slice"].astype(int)
        st.obs["slice_id"] = sample
        for col in ["manual_layer", "layer", "region", "area", "partition", "samplename", "spot_id_original"]:
            if col in adata.obs.columns:
                st.obs[col] = adata.obs[col].astype(str).to_numpy()
        st.obs["library_size"] = library_size
        st.obsm["count"] = counts
        st.obsm["node_features"] = node_features
        st.obsm["spatial_raw"] = adata.obsm["spatial"].copy()
        st.obsm["spatial"] = adata.obsm["spatial"].copy()
        st_list.append(st)

        graph_rows.append(
            {
                "slice_index": i,
                "slice_id": sample,
                "n_spots": int(adata.n_obs),
                "n_hvgs": int(len(hvgs)),
                "radius": radius,
                "avg_neighbors": avg_neighbors,
                "node_feature_dim": int(node_features.shape[1]),
            }
        )
        print(
            f"[INFO] {sample}: radius={radius:.3f}, avg_neighbors={avg_neighbors:.2f}, "
            f"node_features={node_features.shape}"
        )

    shifted_ads = []
    x_offset = 0.0
    for i, st in enumerate(st_list):
        coords = st.obsm["spatial_raw"].copy()
        coords[:, 0] = coords[:, 0] - coords[:, 0].min() + x_offset
        coords[:, 1] = coords[:, 1] - coords[:, 1].min()
        x_offset = float(coords[:, 0].max() + args.min_concat_dist)
        ad_tmp = ad.AnnData(np.zeros((st.n_obs, 1), dtype=np.float32), obs=st.obs.copy())
        ad_tmp.var_names = ["placeholder"]
        ad_tmp.obsm["spatial"] = coords
        ad_tmp.obsm["spatial_raw"] = st.obsm["spatial_raw"].copy()
        shifted_ads.append(ad_tmp)
    adata_full = ad.concat(shifted_ads, join="outer")
    return st_list, adata_full, pd.DataFrame(graph_rows)


def categorical_scatter(ax, x, y, values, title: str, size: float) -> None:
    vals = pd.Series(values).astype(str).fillna("NA")
    categories = sorted(vals.unique())
    cmap = plt.get_cmap("tab20", max(1, len(categories)))
    for i, cat in enumerate(categories):
        mask = vals.to_numpy() == cat
        ax.scatter(x[mask], y[mask], s=size, c=[cmap(i)], label=cat, linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    if len(categories) <= 14:
        ax.legend(markerscale=3, fontsize=5, frameon=False, loc="best")


def continuous_scatter(ax, x, y, values, title: str, size: float) -> None:
    sca = ax.scatter(x, y, s=size, c=values, cmap="viridis", linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(sca, ax=ax, fraction=0.046, pad=0.02)


def plot_outputs(adata_full: ad.AnnData, outdir: Path, args: argparse.Namespace) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    xy = adata_full.obsm["spatial"]
    raw = adata_full.obsm["spatial_raw"]
    size = min(args.max_point_size, max(0.3, 60000.0 / max(adata_full.n_obs, 1)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    categorical_scatter(axes[0, 0], xy[:, 0], xy[:, 1], adata_full.obs["slice_id"], "slice", size)
    categorical_scatter(axes[0, 1], xy[:, 0], xy[:, 1], adata_full.obs.get("manual_layer", "NA"), "manual_layer", size)
    categorical_scatter(axes[1, 0], xy[:, 0], xy[:, 1], adata_full.obs.get("region", "NA"), "region", size)
    categorical_scatter(
        axes[1, 1],
        xy[:, 0],
        xy[:, 1],
        adata_full.obs["dominant_spatial_factor"],
        "dominant INSPIRE factor",
        size,
    )
    fig.savefig(outdir / "combined_spatial_overview.png", dpi=250)
    plt.close(fig)

    um = adata_full.obsm["X_umap"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    categorical_scatter(axes[0, 0], um[:, 0], um[:, 1], adata_full.obs["slice_id"], "UMAP: slice", size)
    categorical_scatter(axes[0, 1], um[:, 0], um[:, 1], adata_full.obs.get("manual_layer", "NA"), "UMAP: manual_layer", size)
    categorical_scatter(axes[1, 0], um[:, 0], um[:, 1], adata_full.obs.get("region", "NA"), "UMAP: region", size)
    categorical_scatter(axes[1, 1], um[:, 0], um[:, 1], adata_full.obs["INSPIRE_louvain"], "UMAP: INSPIRE_louvain", size)
    fig.savefig(outdir / "umap_metadata.png", dpi=250)
    plt.close(fig)

    factor_cols = [c for c in adata_full.obs.columns if c.startswith("Proportion of spatial factor ")]
    for sample in adata_full.obs["slice_id"].astype(str).unique():
        mask = adata_full.obs["slice_id"].astype(str).to_numpy() == sample
        x = raw[mask, 0]
        y = raw[mask, 1]
        obs = adata_full.obs.loc[mask]

        fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
        categorical_scatter(axes[0, 0], x, y, obs.get("manual_layer", "NA"), f"{sample}: manual_layer", size)
        categorical_scatter(axes[0, 1], x, y, obs.get("region", "NA"), f"{sample}: region", size)
        categorical_scatter(axes[1, 0], x, y, obs["dominant_spatial_factor"], f"{sample}: dominant factor", size)
        categorical_scatter(axes[1, 1], x, y, obs["INSPIRE_louvain"], f"{sample}: INSPIRE_louvain", size)
        fig.savefig(outdir / f"{sample}_metadata_spatial.png", dpi=250)
        plt.close(fig)

        n_plot = min(6, len(factor_cols))
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
        axes = axes.ravel()
        for i in range(n_plot):
            continuous_scatter(axes[i], x, y, obs[factor_cols[i]].to_numpy(), f"{sample}: factor {i + 1}", size)
        for i in range(n_plot, len(axes)):
            axes[i].axis("off")
        fig.savefig(outdir / f"{sample}_factor_spatial_top6.png", dpi=250)
        plt.close(fig)


def write_summaries(adata_full: ad.AnnData, basis_df: pd.DataFrame, graph_df: pd.DataFrame, outdir: Path) -> None:
    obs = adata_full.obs.copy()
    obs.to_csv(outdir / "obs_with_inspire_results.csv")
    basis_df.to_csv(outdir / "basis_gene_loadings.csv")
    graph_df.to_csv(outdir / "graph_preprocessing_summary.csv", index=False)

    factor_cols = [c for c in obs.columns if c.startswith("Proportion of spatial factor ")]
    top_gene_rows = []
    for i, row in basis_df.iterrows():
        top = row.sort_values(ascending=False).head(30)
        for rank, (gene, weight) in enumerate(top.items(), start=1):
            top_gene_rows.append({"factor": int(i) + 1, "rank": rank, "gene": gene, "weight": float(weight)})
    pd.DataFrame(top_gene_rows).to_csv(outdir / "top_genes_per_factor.csv", index=False)

    for col in ["slice_id", "manual_layer", "layer", "region", "dominant_spatial_factor", "INSPIRE_louvain"]:
        if col in obs.columns:
            obs[col].astype(str).value_counts().rename_axis(col).reset_index(name="n_spots").to_csv(
                outdir / f"counts_by_{col}.csv", index=False
            )
    if {"slice_id", "manual_layer", "dominant_spatial_factor"}.issubset(obs.columns):
        pd.crosstab([obs["slice_id"], obs["manual_layer"]], obs["dominant_spatial_factor"]).to_csv(
            outdir / "dominant_factor_by_slice_layer.csv"
        )
    if {"slice_id", "region", "dominant_spatial_factor"}.issubset(obs.columns):
        pd.crosstab([obs["slice_id"], obs["region"]], obs["dominant_spatial_factor"]).to_csv(
            outdir / "dominant_factor_by_slice_region.csv"
        )
    if factor_cols:
        obs.groupby("slice_id")[factor_cols].mean().to_csv(outdir / "mean_factor_by_slice.csv")
        if "manual_layer" in obs.columns:
            obs.groupby(["slice_id", "manual_layer"])[factor_cols].mean().to_csv(outdir / "mean_factor_by_slice_layer.csv")
        if "region" in obs.columns:
            obs.groupby(["slice_id", "region"])[factor_cols].mean().to_csv(outdir / "mean_factor_by_slice_region.csv")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    sc.settings.verbosity = 2
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    input_paths = args.input or DEFAULT_INPUTS
    meta_paths = args.meta or METADATA
    config = vars(args).copy()
    config["input"] = input_paths
    config["meta"] = meta_paths
    config["inspire_source"] = str(Path("INSPIRE").resolve())
    (outdir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    adata_list = [load_and_qc(path, sample_meta, args) for path, sample_meta in zip(input_paths, meta_paths)]
    
    hvgs = find_shared_hvgs(adata_list, args.num_hvgs)
    (outdir / "shared_hvgs.txt").write_text("\n".join(hvgs) + "\n", encoding="utf-8")

    st_list, adata_full, graph_df = prepare_inspire_inputs(adata_list, hvgs, args)

    print("[INFO] Training INSPIRE Model_LGCN")
    model = INSPIRE.model.Model_LGCN(
        adata_st_list=st_list,
        n_spatial_factors=args.n_spatial_factors,
        n_training_steps=args.training_steps,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    model.step_interval = max(1, args.training_steps // 5)
    model.train(st_list)

    print("[INFO] Evaluating INSPIRE Model_LGCN with minibatches")
    adata_full, basis_df = model.eval_minibatch(st_list, adata_full, batch_size=args.eval_batch_size)

    factor_cols = [c for c in adata_full.obs.columns if c.startswith("Proportion of spatial factor ")]
    factor_values = adata_full.obs[factor_cols].to_numpy()
    adata_full.obs["dominant_spatial_factor"] = (np.argmax(factor_values, axis=1) + 1).astype(str)

    print("[INFO] Calculating UMAP and Louvain clusters from INSPIRE latent space")
    reducer = umap.UMAP(
        n_neighbors=30,
        n_components=2,
        metric="correlation",
        min_dist=0.3,
        random_state=args.seed,
    )
    adata_full.obsm["X_umap"] = reducer.fit_transform(adata_full.obsm["latent"]).astype(np.float32)
    sc.pp.neighbors(adata_full, use_rep="latent", n_neighbors=30)
    sc.tl.louvain(adata_full, key_added="INSPIRE_louvain", resolution=1.0, random_state=args.seed)

    adata_full.write_h5ad(outdir / "inspire_lgcn_results.h5ad", compression="gzip")
    write_summaries(adata_full, basis_df, graph_df, outdir)
    plot_outputs(adata_full, outdir, args)
    print(f"[INFO] Done. Outputs: {outdir.resolve()}")


if __name__ == "__main__":
    main()
