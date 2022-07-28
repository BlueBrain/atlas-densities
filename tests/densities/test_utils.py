"""
Unit tests for densities utils
"""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest
from voxcell import RegionMap

import atlas_densities.densities.utils as tested
from atlas_densities.exceptions import AtlasDensitiesError

TESTS_PATH = Path(__file__).parent.parent


def test_normalize():
    annotation_raw = np.array(
        [
            [
                [0, 0, 0, 0],
                [0, 255, 1211, 0],
                [0, 347, 100, 0],
                [0, 0, 0, 0],
            ]
        ],
        dtype=np.uint32,
    )
    marker = np.array(
        [
            [
                [0.0, 0.5, 0.5, 0.2],
                [0.1, 0.0, 1.0, 0.1],
                [0.1, 3.0, 5.0, 0.1],
                [0.2, 0.1, 0.1, 0.0],
            ]
        ],
        dtype=float,
    )
    original_marker = marker.copy()
    expected = np.array(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.4 / 4.4, 0.0],
                [0.0, 2.4 / 4.4, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=float,
    )
    # Return a modified copy of the input
    actual = tested.normalize_intensity(marker, annotation_raw)
    npt.assert_array_almost_equal(actual, expected)
    npt.assert_array_equal(marker, original_marker)
    # In place modification
    actual = tested.normalize_intensity(marker, annotation_raw, copy=False)
    npt.assert_array_almost_equal(marker, expected)


def test_compensate_cell_overlap():
    annotation_raw = np.array(
        [
            [
                [0, 0, 0, 0],
                [0, 15, 1, 0],
                [0, 999, 1001, 0],
                [0, 0, 0, 0],
            ]
        ],
        dtype=np.uint32,
    )
    marker = np.array(
        [
            [
                [0.0, -0.5, 0.5, 0.2],
                [0.1, 10.0, 1.0, 0.1],
                [0.1, 3.0, 5.0, 0.1],
                [0.2, 0.1, -0.1, 0.0],
            ]
        ],
        dtype=float,
    )
    original_marker = marker.copy()

    # Return a modified copy of the input
    actual = tested.compensate_cell_overlap(marker, annotation_raw)
    npt.assert_array_equal(marker, original_marker)
    assert np.all(actual >= 0.0)
    assert np.all(actual <= 1.0)
    assert np.all(actual[annotation_raw == 0] == 0.0)
    # In place modification
    tested.normalize_intensity(marker, annotation_raw, copy=False)
    assert np.all(marker >= 0.0)
    assert np.all(marker <= 1.0)
    assert np.all(marker[annotation_raw == 0] == 0.0)


def test_get_group_ids():
    region_map = RegionMap.load_json(Path(TESTS_PATH, "1.json"))
    group_ids = tested.get_group_ids(region_map)
    for ids in group_ids.values():
        assert len(ids) > 0
    assert len(group_ids["Molecular layer"] & group_ids["Purkinje layer"]) == 0
    assert len(group_ids["Cerebellum group"] & group_ids["Isocortex group"]) == 0
    assert len(group_ids["Cerebellum group"] & group_ids["Molecular layer"]) > 0
    assert len(group_ids["Cerebellum group"] & group_ids["Purkinje layer"]) > 0
    assert len(group_ids["Isocortex group"] & group_ids["Molecular layer"]) == 0
    assert len(group_ids["Isocortex group"] & group_ids["Purkinje layer"]) == 0
    assert len(group_ids["Isocortex group"] & group_ids["Rest"]) == 0
    assert len(group_ids["Cerebellum group"] & group_ids["Rest"]) == 0
    assert group_ids["Cerebellar cortex"].issubset(group_ids["Cerebellum group"])


def test_get_region_masks():
    region_map = RegionMap.load_json(Path(TESTS_PATH, "1.json"))
    group_ids = tested.get_group_ids(region_map)
    annotation_raw = np.arange(27000).reshape(30, 30, 30)
    region_masks = tested.get_region_masks(group_ids, annotation_raw)
    brain_mask = np.logical_or(
        np.logical_or(region_masks["Cerebellum group"], region_masks["Isocortex group"]),
        region_masks["Rest"],
    )
    all_nonzero_ids = region_map.find("root", attr="name", with_descendants=True)
    npt.assert_array_equal(np.isin(annotation_raw, list(all_nonzero_ids)), brain_mask)
    np.all(~np.logical_and(region_masks["Cerebellum group"], region_masks["Isocortex group"]))


def test_optimize_distance_to_line_2D():
    line_direction_vector = np.array([1, 2], dtype=float)
    upper_bounds = np.array([3, 1], dtype=float)
    optimum = tested.optimize_distance_to_line(
        line_direction_vector, upper_bounds, 3.0, threshold=1e-7, copy=True
    )
    npt.assert_array_almost_equal(optimum, np.array([2.0, 1.0]))

    line_direction_vector = np.array([2, 3], dtype=float)
    upper_bounds = np.array([1, 3], dtype=float)
    optimum = tested.optimize_distance_to_line(
        line_direction_vector, upper_bounds, 2.0, threshold=1e-7, copy=False
    )
    npt.assert_array_almost_equal(optimum, np.array([0.8, 1.2]))

    line_direction_vector = np.array([1, 2], dtype=float)
    upper_bounds = np.array([3, 1], dtype=float)
    optimum = tested.optimize_distance_to_line(
        line_direction_vector, upper_bounds, 5.0, threshold=1e-7
    )
    npt.assert_array_almost_equal(optimum, np.array([3.0, 1.0]))


def test_optimize_distance_to_line_3D():
    line_direction_vector = np.array([0.5, 2.0, 1.0], dtype=float)
    upper_bounds = np.array([1.0, 1.0, 1.0], dtype=float)
    optimum = tested.optimize_distance_to_line(
        line_direction_vector, upper_bounds, 2.0, threshold=1e-7, copy=True
    )
    npt.assert_array_almost_equal(optimum, np.array([1.0 / 3.0, 1.0, 2.0 / 3.0]))


def test_constrain_cell_counts_per_voxel():
    upper_bound = np.array([[[1.0, 1.0, 2.0, 0.5, 0.5]]])
    cell_counts = np.array([[[0.1, 0.1, 0.6, 0.25, 0.15]]])
    zero_cell_counts_mask = np.array([[[True, True, False, False, False]]])
    max_cell_counts_mask = np.array([[[False, False, True, False, False]]])
    cell_counts = tested.constrain_cell_counts_per_voxel(
        3.0,
        cell_counts,
        upper_bound,
        max_cell_counts_mask,
        zero_cell_counts_mask,
        epsilon=1e-7,
        copy=True,
    )
    expected = np.array([[[0.0, 0.0, 2.0, 0.5, 0.5]]])
    npt.assert_almost_equal(cell_counts, expected, decimal=6)

    upper_bound = np.array([[[2.0, 0.9, 0.75, 0.5, 1.5, 0.1]]])
    cell_counts = np.array([[[0.1, 0.8, 0.25, 0.25, 0.85, 0.1]]])
    zero_cell_counts_mask = np.array([[[True, False, False, False, False, True]]])
    max_cell_counts_mask = np.array([[[False, True, False, False, True, False]]])
    cell_counts = tested.constrain_cell_counts_per_voxel(
        3.4,
        cell_counts,
        upper_bound,
        max_cell_counts_mask,
        zero_cell_counts_mask,
        epsilon=1e-7,
        copy=False,
    )
    expected = np.array([[[0.0, 0.9, 0.5, 0.5, 1.5, 0.0]]])
    npt.assert_almost_equal(cell_counts, expected, decimal=6)

    # Same constraints, but with a different line
    cell_counts = np.array([[[0.1, 0.8, 0.6, 0.2, 0.85, 0.1]]])
    expected = np.array([[[0.0, 0.9, 0.75, 0.25, 1.5, 0.0]]])
    cell_counts = tested.constrain_cell_counts_per_voxel(
        3.4,
        cell_counts,
        upper_bound,
        max_cell_counts_mask,
        zero_cell_counts_mask,
        epsilon=1e-7,
        copy=True,
    )
    npt.assert_almost_equal(cell_counts, expected, decimal=6)


def test_constrain_cell_counts_per_voxel_exceptions():
    # Should raise because the contribution of voxels
    # with maximum cell counts exceeds the target sum
    with pytest.raises(AtlasDensitiesError):
        upper_bound = np.array([[[1.0, 1.0, 4.0, 0.5, 0.5]]])
        cell_counts = np.array([[[0.1, 0.1, 0.6, 0.25, 0.15]]])
        zero_cell_counts_mask = np.array([[[True, True, False, False, False]]])
        max_cell_counts_mask = np.array([[[False, False, True, False, False]]])
        tested.constrain_cell_counts_per_voxel(
            3.0,
            cell_counts,
            upper_bound,
            max_cell_counts_mask,
            zero_cell_counts_mask,
            epsilon=1e-7,
            copy=True,
        )

    # Should raise because the maximum contribution of voxels
    # with non-zero cell counts is less than the target sum.
    with pytest.raises(AtlasDensitiesError):
        upper_bound = np.array([[[1.0, 1.0, 1.0, 0.5, 0.5]]])
        cell_counts = np.array([[[0.1, 0.1, 0.6, 0.25, 0.15]]])
        zero_cell_counts_mask = np.array([[[True, True, False, False, False]]])
        max_cell_counts_mask = np.array([[[False, False, True, False, False]]])
        tested.constrain_cell_counts_per_voxel(
            3.0,
            cell_counts,
            upper_bound,
            max_cell_counts_mask,
            zero_cell_counts_mask,
            epsilon=1e-7,
            copy=True,
        )

    # Should raise because the target sum is not reached
    with pytest.raises(AtlasDensitiesError):
        upper_bound = np.array([[[1.0, 1.0, 1.0, 0.5, 0.5]]])
        cell_counts = np.array([[[0.1, 0.1, 0.6, 0.25, 0.15]]])
        zero_cell_counts_mask = np.array([[[True, True, False, False, False]]])
        max_cell_counts_mask = np.array([[[False, False, True, False, False]]])
        with patch(
            "atlas_densities.densities.utils.optimize_distance_to_line",
            return_value=cell_counts,
        ):
            tested.constrain_cell_counts_per_voxel(
                3.0,
                cell_counts,
                upper_bound,
                max_cell_counts_mask,
                zero_cell_counts_mask,
                copy=True,
            )


def get_hierarchy():
    return {
        "id": 8,
        "name": "Basic cell groups and regions",
        "acronym": "grey",
        "parent_structure_id": None,  # would be null in json
        "children": [
            {
                "id": 920,
                "acronym": "CENT",
                "name": "Central lobule",
                "parent_structure_id": 645,
                "children": [
                    {
                        "id": 976,
                        "acronym": "CENT2",
                        "name": "Lobule II",
                        "parent_structure_id": 920,
                        "children": [
                            {
                                "id": 10710,
                                "acronym": "CENT2mo",
                                "name": "Lobule II, molecular layer",
                                "parent_structure_id": 976,
                                "children": [],
                            },
                            {
                                "id": 10709,
                                "acronym": "CENT2pu",
                                "name": "Lobule II, Purkinje layer",
                                "parent_structure_id": 976,
                                "children": [],
                            },
                            {
                                "id": 10708,
                                "acronym": "CENT2gr",
                                "name": "Lobule II, granular layer",
                                "parent_structure_id": 976,
                                "children": [],
                            },
                        ],
                    }
                ],
            }
        ],
    }


@pytest.fixture
def region_map():
    return RegionMap.from_dict(get_hierarchy())


def get_hierarchy_info():
    return pd.DataFrame(
        {
            "brain_region": [
                "Central lobule",
                "Lobule II",
                "Lobule II, granular layer",
                "Lobule II, Purkinje layer",
                "Lobule II, molecular layer",
            ],
            "descendant_id_set": [
                {920, 976, 10708, 10709, 10710},
                {976, 10708, 10709, 10710},
                {10708},
                {10709},
                {10710},
            ],
        },
        index=[920, 976, 10708, 10709, 10710],
    )


def test_get_hierarchy(region_map):
    pdt.assert_frame_equal(
        pd.DataFrame(tested.get_hierarchy_info(region_map, root="Central lobule")),
        get_hierarchy_info(),
    )


@pytest.fixture
def annotation():
    return np.array([[[920, 10710, 10710], [10709, 10708, 976], [10708, 10710, 10709]]])


@pytest.fixture
def volumes(voxel_volume=2):
    hierarchy_info = get_hierarchy_info()
    return pd.DataFrame(
        {
            "brain_region": hierarchy_info["brain_region"],
            "id_volume": voxel_volume * np.array([1.0, 1.0, 2.0, 2.0, 3.0], dtype=float),
            "volume": voxel_volume * np.array([9.0, 8.0, 2.0, 2.0, 3.0], dtype=float),
        },
        index=hierarchy_info.index,
    )


def test_compute_region_volumes(volumes, annotation):
    pdt.assert_frame_equal(
        volumes,  # expected
        tested.compute_region_volumes(
            annotation, voxel_volume=2.0, hierarchy_info=get_hierarchy_info()
        ),
    )


@pytest.fixture
def cell_counts(voxel_volume=2):
    counts = voxel_volume * np.array([5.0, 4.0, 1.0, 1.0, 1.0])
    hierarchy_info = get_hierarchy_info()
    return pd.DataFrame(
        {"brain_region": hierarchy_info["brain_region"], "cell_count": counts},
        index=hierarchy_info.index,
    )


@pytest.fixture
def cell_density():
    return np.array([[[1.0, 1.0 / 3.0, 1.0 / 3.0], [0.5, 0.5, 1.0], [0.5, 1.0 / 3.0, 0.5]]])


def test_compute_cell_counts(annotation, cell_density, cell_counts):
    pdt.assert_frame_equal(
        cell_counts,  # expected
        tested.compute_region_cell_counts(
            annotation, cell_density, voxel_volume=2.0, hierarchy_info=get_hierarchy_info()
        ),
    )


def test_zero_negative_values():
    array = np.array([0, 1, -0.02], dtype=float)
    with pytest.raises(
        AtlasDensitiesError,
        match="absolute value of the sum of all negative values exceeds 1 percent of the sum of all positive values",
    ):
        tested.zero_negative_values(array)

    array = np.array([0, 1, -0.01], dtype=float)
    with pytest.raises(
        AtlasDensitiesError,
        match="smallest negative value is not negligible wrt to the mean of all non-negative values",
    ):
        tested.zero_negative_values(array)

    array = np.array([0, 1, -1e-8 / 2.0], dtype=float)
    tested.zero_negative_values(array)
    npt.assert_array_almost_equal(array, np.array([0, 1, 0], dtype=float))

    array = np.array([0, 1, 1], dtype=float)
    tested.zero_negative_values(array)
    npt.assert_array_almost_equal(array, np.array([0, 1, 1], dtype=float))
