"""
src/parcel/clustering.py

Spatial adjacency graph construction and building clustering analysis.
Evaluates spatial separation thresholds to identify coherent urban clusters.
Vectorized and indexed for high-performance spatial processing.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.strtree import STRtree
from shapely.geometry import box
from scipy.spatial import cKDTree
import networkx as nx


def calculate_nearest_neighbor_distances(buildings_gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Calculates the minimum distance from each building polygon to its nearest neighbor.
    Uses cKDTree candidate filtering followed by exact geometric polygon distance.
    """
    n = len(buildings_gdf)
    if n <= 1:
        return np.zeros(n)

    geoms = buildings_gdf.geometry.values
    centroids = np.array([[g.centroid.x, g.centroid.y] for g in geoms])
    k_cand = min(8, n)
    tree = cKDTree(centroids)
    _, idxs = tree.query(centroids, k=k_cand)

    nn_distances = np.full(n, np.inf)
    for i in range(n):
        for j in idxs[i]:
            if i != j:
                d = geoms[i].distance(geoms[j])
                if d < nn_distances[i]:
                    nn_distances[i] = d

    # Replace any inf with 0
    nn_distances[np.isinf(nn_distances)] = 0.0
    return nn_distances


def build_spatial_adjacency_graph(
    buildings_gdf: gpd.GeoDataFrame,
    distance_threshold: float,
) -> nx.Graph:
    """
    Constructs an undirected graph connecting buildings whose edge-to-edge
    separation distance is less than or equal to `distance_threshold` meters.
    
    Uses envelope box queries on Shapely STRtree for fast O(N log N) scaling.
    """
    n = len(buildings_gdf)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))

    if n <= 1:
        return graph

    geoms = buildings_gdf.geometry.values
    tree = STRtree(geoms)

    for i, g in enumerate(geoms):
        minx, miny, maxx, maxy = g.bounds
        search_box = box(
            minx - distance_threshold,
            miny - distance_threshold,
            maxx + distance_threshold,
            maxy + distance_threshold,
        )
        candidates = tree.query(search_box)
        for j in candidates:
            if j > i:
                dist = geoms[i].distance(geoms[j])
                if dist <= distance_threshold:
                    graph.add_edge(i, j, distance=dist)

    return graph


def cluster_buildings(
    buildings_gdf: gpd.GeoDataFrame,
    distance_threshold: float,
) -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
    """
    Assigns cluster IDs to buildings based on connected components in the
    spatial adjacency graph with the given distance threshold.
    """
    graph = build_spatial_adjacency_graph(buildings_gdf, distance_threshold)
    components = list(nx.connected_components(graph))

    cluster_ids = np.zeros(len(buildings_gdf), dtype=int)
    cluster_sizes = []

    for c_id, comp in enumerate(components):
        cluster_sizes.append(len(comp))
        for idx in comp:
            cluster_ids[idx] = c_id

    gdf = buildings_gdf.copy()
    gdf["cluster_id"] = cluster_ids

    cluster_sizes = np.array(cluster_sizes)
    isolated_count = int(np.sum(cluster_sizes == 1))
    small_clusters = int(np.sum((cluster_sizes >= 2) & (cluster_sizes <= 5)))
    medium_clusters = int(np.sum((cluster_sizes >= 6) & (cluster_sizes <= 20)))
    large_clusters = int(np.sum(cluster_sizes > 20))

    summary = {
        "distance_threshold_m": distance_threshold,
        "total_buildings": len(buildings_gdf),
        "total_clusters": len(components),
        "isolated_buildings": isolated_count,
        "small_clusters_2_to_5": small_clusters,
        "medium_clusters_6_to_20": medium_clusters,
        "large_clusters_gt_20": large_clusters,
        "mean_cluster_size": float(np.mean(cluster_sizes)),
        "median_cluster_size": float(np.median(cluster_sizes)),
        "max_cluster_size": int(np.max(cluster_sizes)),
    }

    return gdf, summary


def evaluate_clustering_thresholds(
    buildings_gdf: gpd.GeoDataFrame,
    thresholds: List[float] = [5.0, 10.0, 15.0, 20.0, 30.0],
) -> pd.DataFrame:
    """
    Evaluates a spectrum of spatial clustering thresholds and returns
    a comparative analysis DataFrame.
    """
    records = []
    for t in thresholds:
        _, summary = cluster_buildings(buildings_gdf, distance_threshold=t)
        records.append(summary)

    return pd.DataFrame(records)
