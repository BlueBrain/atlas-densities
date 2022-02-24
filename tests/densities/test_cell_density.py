"""
Unit tests for overall cell density computation
"""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
from voxcell import RegionMap  # type: ignore

import atlas_densities.densities.cell_density as tested
from atlas_densities.densities.cell_counts import cell_counts
from atlas_densities.densities.utils import get_group_ids, get_region_masks

TESTS_PATH = Path(__file__).parent.parent


def test_compute_cell_density():
    region_map = RegionMap.load_json(Path(TESTS_PATH, "1.json"))
    annotation = np.arange(8000).reshape(20, 20, 20)
    voxel_volume = 25**3 / 1e9
    nissl = np.random.random_sample(annotation.shape)
    cell_density = tested.compute_cell_density(region_map, annotation, voxel_volume, nissl)
    # Each group has a prescribed cell count
    group_ids = get_group_ids(region_map)
    region_masks = get_region_masks(group_ids, annotation)
    for group, mask in region_masks.items():
        npt.assert_array_almost_equal(
            np.sum(cell_density[mask]) * voxel_volume, cell_counts()[group]
        )

    # The voxels in the Cerebellum group which belong to the Purkinje layer
    # should all have the same cell density.
    purkinje_layer_mask = np.isin(annotation, list(group_ids["Purkinje layer"]))
    purkinje_layer_mask = np.logical_and(region_masks["Cerebellum group"], purkinje_layer_mask)
    densities = np.unique(cell_density[purkinje_layer_mask])
    assert len(densities) == 1


def test_cell_density_with_soma_correction():
    region_map = RegionMap.load_json(Path(TESTS_PATH, "1.json"))
    annotation = np.arange(8000).reshape(20, 20, 20)
    voxel_volume = 25**3 / 1e9
    nissl = np.random.random_sample(annotation.shape)
    cell_density = tested.compute_cell_density(
        region_map,
        annotation,
        voxel_volume,
        nissl,
    )
    # Each group has a prescribed cell count
    group_ids = get_group_ids(region_map)
    region_masks = get_region_masks(group_ids, annotation)
    for group, mask in region_masks.items():
        npt.assert_array_almost_equal(
            np.sum(cell_density[mask]) * voxel_volume, cell_counts()[group]
        )

    # The voxels in the Cerebellum group which belong to the Purkinje layer
    # should all have the same cell density.
    purkinje_layer_mask = np.isin(annotation, list(group_ids["Purkinje layer"]))
    purkinje_layer_mask = np.logical_and(region_masks["Cerebellum group"], purkinje_layer_mask)
    densities = np.unique(cell_density[purkinje_layer_mask])
    assert len(densities) == 1


def test_cell_density_options():
    region_map = RegionMap.load_json(Path(TESTS_PATH, "1.json"))
    annotation = np.arange(8000).reshape(20, 20, 20)
    voxel_volume = 25**3 / 1e9
    nissl = np.random.random_sample(annotation.shape)
    group_ids = get_group_ids(region_map)
    region_masks = get_region_masks(group_ids, annotation)
    with patch(
        "atlas_densities.densities.cell_density.compensate_cell_overlap",
        return_value=nissl,
    ):
        actual = tested.compute_cell_density(region_map, annotation, voxel_volume, nissl)
        expected_intensity = nissl.copy()
        tested.fix_purkinje_layer_intensity(
            region_map, annotation, cell_counts(), expected_intensity
        )
        for group, mask in region_masks.items():
            expected_intensity[mask] = nissl[mask] * (cell_counts()[group] / np.sum(nissl[mask]))
        npt.assert_array_almost_equal(expected_intensity, actual * voxel_volume)
