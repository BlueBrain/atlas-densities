"""
Unit tests for inhibitory cell density computation
"""
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest

import atlas_densities.densities.refined_inhibitory_neuron_densities as tested

TESTS_PATH = Path(__file__).parent.parent


@pytest.fixture
def density_helper_1():
    annotation = np.array([[[1, 2, 3]]], dtype=int)
    neuron_density = np.array([[[0.1, 0.2, 0.3]]], dtype=float)
    return tested.VolumetricDensityHelper(
        annotation,
        (25**3) * 1e-9,  # voxel volume in mm^3
        neuron_density,
        pd.DataFrame(
            {
                "pv+": [1.0, 2.0],
                "pv+_standard_deviation": [0.1, 0.2],
                "sst+": [3.0, 4.0],
                "sst+_standard_deviation": [0.3, 0.4],
                "vip+": [5.0, 6.0],
                "vip+_standard_deviation": [0.5, 0.6],
                "gad67+": [9.0, 12.0],
                "gad67+_standard_deviation": [0.9, 1.2],
            },
            index=["A", "B"],
        ),
        "gad67+",
        ["pv+", "sst+", "vip+"],
    )


def test_VolumetricDensityHelper_constructor(density_helper_1):
    density_helper_1.initialize_volumetric_densities()

    assert set(density_helper_1.volumetric_densities.keys()) == {"gad67+", "pv+", "sst+", "vip+"}
    npt.assert_array_equal(
        np.array(list(density_helper_1.volumetric_densities.values())),
        np.zeros((4, 1, 1, 3), dtype=float),
    )


def test_get_zero_counts(density_helper_1):
    assert density_helper_1.get_zero_counts() == {
        "gad67+": 0.0,
        "pv+": 0.0,
        "sst+": 0.0,
        "vip+": 0.0,
    }


def test_get_neuron_count(density_helper_1):
    assert np.isclose(
        density_helper_1.get_neuron_count(np.array([[[True, False, False]]])), (25**3) * 1e-10
    )
    assert np.isclose(
        density_helper_1.get_neuron_count(np.array([[[True, True, False]]])), (25**3) * 3e-10
    )


def test_fill_volumetric_densities(density_helper_1):
    density_helper_1.initialize_volumetric_densities()
    density_helper_1.fill_volumetric_densities(
        1, {"gad67+": 1.0, "pv+": 2.0, "sst+": 3.0, "vip+": 4.0}
    )
    mask = density_helper_1.annotation == 1
    actual_counts = {
        cell_type: np.sum(density_helper_1.volumetric_densities[cell_type][mask]) * (25**3) * 1e-9
        for cell_type in ["gad67+", "pv+", "sst+", "vip+"]
    }
    assert actual_counts == {"gad67+": 1.0, "pv+": 2.0, "sst+": 3.0, "vip+": 4.0}


def test_compute_consistent_region_counts(density_helper_1):
    # If the cell_count (gad67+) is less than tolerance = 1e-2, return zero counts
    assert density_helper_1.compute_consistent_region_counts(
        1, 1e-3, 0.1, [1.0, 2.0, 3.0], 0.5
    ) == {"gad67+": 0.0, "pv+": 0.0, "sst+": 0.0, "vip+": 0.0}

    # No consistency issue: returns the initial counts
    density_helper_1.voxel_volume = 100.0
    assert density_helper_1.compute_consistent_region_counts(1, 10, 0.1, [1.0, 2.0, 3.0], 0.5) == {
        "gad67+": 10.0,
        "pv+": 1.0,
        "sst+": 2.0,
        "vip+": 3.0,
    }

    # Inconsistency: cell_count > np.sum(subcounts)
    # Keep each count within the range [count - count_deviation, count + count_deviation]
    density_helper_1.voxel_volume = 100.0
    counts = density_helper_1.compute_consistent_region_counts(1, 5, 0.1, [1.0, 2.0, 3.0], 1.0)
    assert counts["gad67+"] <= 5.1 and counts["gad67+"] >= 4.9
    assert counts["gad67+"] <= density_helper_1.get_neuron_count(density_helper_1.annotation == 1)
    subsum = counts["pv+"] + counts["sst+"] + counts["vip+"]
    assert subsum >= 5.0 and subsum <= 7.0

    density_helper_1.voxel_volume = 50.0
    counts = density_helper_1.compute_consistent_region_counts(2, 3, 0.1, [1.0, 1.0, 1.5], 1.0)
    assert counts["gad67+"] <= 3.1 and counts["gad67+"] >= 2.9
    assert counts["gad67+"] <= density_helper_1.get_neuron_count(density_helper_1.annotation == 2)
    subsum = counts["pv+"] + counts["sst+"] + counts["vip+"]
    assert subsum >= 2.5 and subsum <= 4.5

    # Inconsistency cannot be resolved within the prescribed ranges.
    # We still preserve subcounts proportions
    density_helper_1.voxel_volume = 55.0
    counts = density_helper_1.compute_consistent_region_counts(1, 5, 0.1, [1.0, 2.0, 3.0], 0.1)
    assert counts["gad67+"] <= density_helper_1.get_neuron_count(density_helper_1.annotation == 1)
    subsum = counts["pv+"] + counts["sst+"] + counts["vip+"]
    expected_proportions = np.array([1.0, 2.0, 3.0]) / 6.0
    actual_proportions = np.array([counts["pv+"], counts["sst+"], counts["vip+"]])
    actual_proportions = actual_proportions / np.sum(actual_proportions)
    npt.assert_array_almost_equal(expected_proportions, actual_proportions)

    # Special case with zero deviations
    # Inconsistency cannot be resolved within the prescribed ranges.
    # We still preserve subcounts proportions
    density_helper_1.voxel_volume = 55.0
    counts = density_helper_1.compute_consistent_region_counts(1, 5, 0.0, [1.0, 2.0, 3.0], 0.0)
    assert np.isclose(counts["gad67+"], (5.0 + 6.0) / 2.0)
    assert counts["gad67+"] <= density_helper_1.get_neuron_count(density_helper_1.annotation == 1)
    subsum = counts["pv+"] + counts["sst+"] + counts["vip+"]
    expected_proportions = np.array([1.0, 2.0, 3.0]) / 6.0
    actual_proportions = np.array([counts["pv+"], counts["sst+"], counts["vip+"]])
    actual_proportions = actual_proportions / np.sum(actual_proportions)
    npt.assert_array_almost_equal(expected_proportions, actual_proportions)


@pytest.fixture
def density_helper_2():
    annotation = np.array([[[1, 2, 3]]], dtype=int)
    neuron_density = np.array([[[0.1, 0.2, 0.3]]], dtype=float)
    return tested.VolumetricDensityHelper(
        annotation,
        100.0,
        neuron_density,
        pd.DataFrame(
            {
                "pv+": [1.0, 2.0],
                "pv+_standard_deviation": [0.1, 0.2],
                "sst+": [3.0, 4.0],
                "sst+_standard_deviation": [0.3, 0.4],
                "vip+": [5.0, 6.0],
                "vip+_standard_deviation": [0.5, 0.6],
                "gad67+": [8.0, 11.0],
                "gad67+_standard_deviation": [0.9, 1.2],
            },
            index=["A", "B"],
        ),
        "gad67+",
        ["pv+", "sst+", "vip+"],
    )


def test_compute_consistent_leaf_counts(density_helper_1, density_helper_2):
    density_helper_1.voxel_volume = 100.0
    hierarchy_leaf_info = pd.DataFrame({"brain_region": ["A", "B"]}, index=[2, 3])

    expected = {
        "A": {
            "gad67+": 9.0,
            "pv+": 1.0,
            "sst+": 3.0,
            "vip+": 5.0,
        },
        "B": {
            "gad67+": 12.0,
            "pv+": 2.0,
            "sst+": 4.0,
            "vip+": 6.0,
        },
    }
    assert expected == density_helper_1.compute_consistent_leaf_counts(hierarchy_leaf_info)

    leaf_counts = density_helper_2.compute_consistent_leaf_counts(hierarchy_leaf_info)
    expected_proportions = {
        "A": [1.0 / 9.0, 3.0 / 9.0, 5.0 / 9.0],
        "B": [2.0 / 12.0, 4.0 / 12.0, 6.0 / 12.0],
    }
    for region_name, id_ in zip(["A", "B"], [2, 3]):
        counts = leaf_counts[region_name]
        assert counts["gad67+"] <= density_helper_2.get_neuron_count(
            density_helper_2.annotation == id_
        )
        subsum = counts["pv+"] + counts["sst+"] + counts["vip+"]
        assert np.isclose(counts["gad67+"], subsum)
        actual_proportions = np.array([counts["pv+"], counts["sst+"], counts["vip+"]]) / subsum
        npt.assert_array_almost_equal(expected_proportions[region_name], actual_proportions)


@pytest.fixture
def density_helper_3():
    annotation = np.array([[[1, 2, 3]]], dtype=int)
    neuron_density = np.array([[[0.1, 0.2, 0.3]]], dtype=float)
    return tested.VolumetricDensityHelper(
        annotation,
        150.0,
        neuron_density,
        pd.DataFrame(
            {
                "pv+": [1.0, 2.0, 3.0],
                "pv+_standard_deviation": [0.1, 0.2, 0.3],
                "sst+": [4.0, 5.0, 9.0],
                "sst+_standard_deviation": [0.3, 0.4, 0.5],
                "vip+": [5.0, 6.0, 13.0],
                "vip+_standard_deviation": [0.5, 0.6, 0.7],
                "gad67+": [11.0, 13.0, 24.0],
                "gad67+_standard_deviation": [0.9, 1.2, 1.5],
            },
            index=["A", "B", "C"],
        ),
        "gad67+",
        ["pv+", "sst+", "vip+"],
    )


@pytest.fixture
def density_helper_4():
    annotation = np.array([[[1, 2, 3]]], dtype=int)
    neuron_density = np.array([[[0.1, 0.2, 0.3]]], dtype=float)
    return tested.VolumetricDensityHelper(
        annotation,
        150.0,
        neuron_density,
        pd.DataFrame(
            {
                "pv+": [1.0, 2.0, 2.0],
                "pv+_standard_deviation": [0.1, 0.2, 0.3],
                "sst+": [4.0, 5.0, 5.0],
                "sst+_standard_deviation": [0.3, 0.4, 0.5],
                "vip+": [5.0, 6.0, 12.0],
                "vip+_standard_deviation": [0.5, 0.6, 0.7],
                "gad67+": [11.0, 13.0, 25.0],
                "gad67+_standard_deviation": [0.9, 1.2, 1.5],
            },
            index=["A", "B", "C"],
        ),
        "gad67+",
        ["pv+", "sst+", "vip+"],
    )


def test_compute_consistent_counts(density_helper_3, density_helper_4):
    leaf_counts = {
        region_name: {
            cell_type: density_helper_3.region_counts.at[region_name, cell_type]
            for cell_type in ["gad67+", "pv+", "sst+", "vip+"]
        }
        for region_name in ["A", "B"]
    }
    hierarchy = {
        "name": "C",
        "id": 3,
        "children": [
            {
                "name": "A",
                "id": 1,
                "children": [],
            },
            {
                "name": "B",
                "id": 2,
            },
        ],
    }
    nomansland_counts = density_helper_3.compute_consistent_counts(leaf_counts, hierarchy)
    assert nomansland_counts == {
        "C": {
            "gad67+": 0.0,
            "pv+": 0.0,
            "sst+": 0.0,
            "vip+": 0.0,
        }
    }

    leaf_counts = {
        region_name: {
            cell_type: density_helper_4.region_counts.at[region_name, cell_type]
            for cell_type in ["gad67+", "pv+", "sst+", "vip+"]
        }
        for region_name in ["A", "B"]
    }
    nomansland_counts = density_helper_4.compute_consistent_counts(leaf_counts, hierarchy)
    assert nomansland_counts == {
        "C": {
            "gad67+": 1.0,
            "pv+": 0.0,
            "sst+": 0.0,
            "vip+": 1.0,
        }
    }


def test_average_densities_to_cell_counts():
    average_densities = pd.DataFrame(
        {
            "pv+": [1.0, 2.0],
            "pv+_standard_deviation": [0.1, 0.2],
            "sst+": [3.0, 4.0],
            "sst+_standard_deviation": [0.3, 0.4],
        },
        index=["A", "B"],
    )
    region_volumes = pd.DataFrame(
        {"volume": [10.0, 20.0], "brain_region": ["A", "B"]}, index=[1, 2]
    )
    actual = tested.average_densities_to_cell_counts(average_densities, region_volumes)
    pdt.assert_frame_equal(
        actual,
        pd.DataFrame(
            {
                "pv+": [10.0, 40.0],
                "pv+_standard_deviation": [1.0, 4.0],
                "sst+": [30.0, 80.0],
                "sst+_standard_deviation": [3.0, 8.0],
            },
            index=["A", "B"],
        ),
    )


def get_inhibitory_neuron_densities_data():
    return {
        "hierarchy": {
            "name": "root",
            "id": 3,
            "children": [
                {
                    "name": "A",
                    "id": 1,
                    "children": [],
                },
                {
                    "name": "B",
                    "id": 2,
                },
            ],
        },
        "annotation": np.array([[[1, 2, 3]]], dtype=int),
        "neuron_density": np.array([[[0.11, 0.2, 0.3]]], dtype=float),
        "average_densities": pd.DataFrame(
            {
                "pv+": [0.01, 0.02, 0.02],
                "pv+_standard_deviation": [0.001, 0.002, 0.003],
                "sst+": [0.04, 0.05, 0.05],
                "sst+_standard_deviation": [0.003, 0.004, 0.005],
                "vip+": [0.05, 0.06, 0.12],
                "vip+_standard_deviation": [0.005, 0.006, 0.007],
                "gad67+": [0.11, 0.13, 0.25 / 3.0],
                "gad67+_standard_deviation": [0.009, 0.012, 0.015 / 3.0],
            },
            index=["A", "B", "root"],
        ),
        "expected_proportions": {
            "A": [1.0 / 10.0, 4.0 / 10.0, 5.0 / 10.0],
            "B": [2.0 / 13.0, 5.0 / 13.0, 6.0 / 13.0],
        },
    }


def test_create_inhibitory_neuron_densities():
    data = get_inhibitory_neuron_densities_data()
    expected_proportions = data["expected_proportions"]
    voxel_volume = 100.0

    volumetric_densities = tested.create_inhibitory_neuron_densities(
        data["hierarchy"],
        data["annotation"],
        voxel_volume,
        data["neuron_density"],
        data["average_densities"],
    )

    for index, region_name in enumerate(["A", "B", "root"], 1):
        subsum = 0.0
        region_mask = data["annotation"] == index
        subcounts = []
        for cell_subtype in ["pv+", "sst+", "vip+"]:
            subcounts.append(np.sum(volumetric_densities[cell_subtype][region_mask]) * voxel_volume)
        subsum = np.sum(subcounts)
        assert np.sum(volumetric_densities["gad67+"][region_mask]) * voxel_volume >= subsum
        assert np.all(
            volumetric_densities["gad67+"][region_mask] <= data["neuron_density"][region_mask]
        )
        if region_name in ["A", "B"]:  # leaf regions
            actual_proportions = np.array(subcounts) / subsum
            npt.assert_array_almost_equal(expected_proportions[region_name], actual_proportions)
