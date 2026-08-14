from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("spagcn-modern")
except PackageNotFoundError:
    __version__ = "1.2.7.post1"

from .SpaGCN import SpaGCN, multiSpaGCN
from .calculate_adj import (
    calculate_adj_matrix,
    euclid_dist,
    extract_color,
    pairwise_distance,
)
from .calculate_moran_I import Geary_C, Moran_I
from .ez_mode import (
    detect_spatial_domains_ez_mode,
    detect_SVGs_ez_mode,
    detect_meta_genes_ez_mode,
    plot_meta_genes_ez_mode,
    plot_spatial_domains_ez_mode,
    plot_SVGs_ez_mode,
    spatial_domains_refinement_ez_mode,
)
from .util import (
    calculate_p,
    count_nbr,
    detect_subclusters,
    find_l,
    find_meta_gene,
    find_neighbor_clusters,
    plot_log_exp,
    plot_relative_exp,
    prefilter_cells,
    prefilter_genes,
    prefilter_specialgenes,
    rank_genes_groups,
    refine,
    relative_func,
    search_l,
    search_radius,
    search_res,
    test_l,
)

__all__ = [
    "Geary_C",
    "Moran_I",
    "SpaGCN",
    "calculate_adj_matrix",
    "calculate_p",
    "count_nbr",
    "detect_SVGs_ez_mode",
    "detect_meta_genes_ez_mode",
    "detect_spatial_domains_ez_mode",
    "detect_subclusters",
    "euclid_dist",
    "extract_color",
    "find_l",
    "find_meta_gene",
    "find_neighbor_clusters",
    "multiSpaGCN",
    "pairwise_distance",
    "plot_SVGs_ez_mode",
    "plot_log_exp",
    "plot_meta_genes_ez_mode",
    "plot_relative_exp",
    "plot_spatial_domains_ez_mode",
    "prefilter_cells",
    "prefilter_genes",
    "prefilter_specialgenes",
    "rank_genes_groups",
    "refine",
    "relative_func",
    "search_l",
    "search_radius",
    "search_res",
    "spatial_domains_refinement_ez_mode",
    "test_l",
]
