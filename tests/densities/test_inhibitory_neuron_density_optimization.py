"""
Unit tests for inhibitory cell density computation
(linear program minimizing distances to initial estimates)
"""

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest

import atlas_densities.densities.inhibitory_neuron_densities_optimization as tested
from atlas_densities.exceptions import AtlasDensitiesError


def test_create_volumetric_densities():
    annotation = np.array([[[1, 1, 2]]], dtype=int)
    neuron_density = np.array([[[0.5, 0.5, 1.0]]])
    neuron_counts = pd.DataFrame({"cell_count": [4.0, 4.0]}, index=[1, 2])
    cell_types = ["pv+", "sst+"]
    x_result = pd.DataFrame({"pv+": [3.0, 1.0], "sst+": [1.0, 3.0]}, index=[1, 2])

    densities = tested.create_volumetric_densities(
        x_result, annotation, neuron_density, neuron_counts, cell_types
    )
    npt.assert_array_almost_equal(densities["pv+"], np.array([[[3.0 / 8.0, 3.0 / 8.0, 1.0 / 4.0]]]))
    npt.assert_array_almost_equal(
        densities["sst+"], np.array([[[1.0 / 8.0, 1.0 / 8.0, 3.0 / 4.0]]])
    )


def get_initialization_data_1():
    hierarchy = {
        "id": 1,
        "name": "root",
        "children": [],
    }

    average_densities = pd.DataFrame(
        {
            "gad67+": [3.5],
            "gad67+_standard_deviation": [1.0],
            "pv+": [1.0],
            "pv+_standard_deviation": [1.0],
            "sst+": [1.0],
            "sst+_standard_deviation": [1.0],
            "vip+": [1.0],
            "vip+_standard_deviation": [1.0],
        },
        index=["root"],
    )

    return {
        "hierarchy": hierarchy,
        "annotation": np.array([[[1]]]),
        "voxel_volume": 1.0,
        "neuron_density": np.array([[[4.0]]]),
        "average_densities": average_densities,
    }


def test_create_inhibitory_neuron_densities_1():
    data = get_initialization_data_1()
    densities = tested.create_inhibitory_neuron_densities(
        data["hierarchy"],
        data["annotation"],
        data["voxel_volume"],
        data["neuron_density"],
        data["average_densities"],
    )

    expected = {
        "gad67+": np.array([[[3.5]]]),
        "pv+": np.array([[[1.0]]]),
        "sst+": np.array([[[1.0]]]),
        "vip+": np.array([[[1.0]]]),
    }

    for cell_type, density in densities.items():
        npt.assert_array_almost_equal(density, expected[cell_type])


def get_initialization_data_2():
    hierarchy = {
        "id": 1,
        "name": "root",
        "children": [],
    }

    average_densities = pd.DataFrame(
        {
            "gad67+": [3.5],
            "gad67+_standard_deviation": [0.0],
            "pv+": [1.0],
            "pv+_standard_deviation": [0.0],
            "sst+": [1.0],
            "sst+_standard_deviation": [0.0],
            "vip+": [1.0],
            "vip+_standard_deviation": [0.0],
        },
        index=["root"],
    )

    return {
        "hierarchy": hierarchy,
        "annotation": np.array([[[1]]]),
        "voxel_volume": 1.0,
        "neuron_density": np.array([[[4.0]]]),
        "average_densities": average_densities,
    }


def test_create_inhibitory_neuron_densities_2():
    data = get_initialization_data_2()
    densities = tested.create_inhibitory_neuron_densities(
        data["hierarchy"],
        data["annotation"],
        data["voxel_volume"],
        data["neuron_density"],
        data["average_densities"],
    )

    expected = {
        "gad67+": np.array([[[3.5]]]),
        "pv+": np.array([[[1.0]]]),
        "sst+": np.array([[[1.0]]]),
        "vip+": np.array([[[1.0]]]),
    }

    for cell_type, density in densities.items():
        npt.assert_array_almost_equal(density, expected[cell_type])


def test_unfeasible_constraint():
    data = get_initialization_data_2()
    data["average_densities"].loc["root", "vip+"] = 5.0
    data["average_densities"].loc["root", "vip+_standard_deviation"] = 0.0
    with pytest.raises(AtlasDensitiesError, match="neuron count"):
        tested.create_inhibitory_neuron_densities(
            data["hierarchy"],
            data["annotation"],
            data["voxel_volume"],
            data["neuron_density"],
            data["average_densities"],
        )


def get_initialization_data_3():
    hierarchy = {
        "id": 8,
        "name": "root",
        "children": [
            {"id": 1, "name": "A", "children": []},
            {"id": 2, "name": "B", "children": []},
        ],
    }

    average_densities = pd.DataFrame(
        {
            "gad67+": [3.5, 3.0, 6.0],
            "gad67+_standard_deviation": [1.0, 1.0, 1.0],
            "pv+": [1.5, 1.0, 2.0],
            "pv+_standard_deviation": [1.0, 1.0, 1.0],
            "sst+": [1.0, 1.0, 2.0],
            "sst+_standard_deviation": [1.0, 1.0, 1.0],
            "vip+": [1.0, 1.0, 2.0],
            "vip+_standard_deviation": [1.0, 1.0, 1.0],
        },
        index=["A", "B", "root"],
    )

    return {
        "hierarchy": hierarchy,
        "annotation": np.array([[[1, 2]]]),
        "voxel_volume": 1.0,
        "neuron_density": np.array([[[4.0, 3.0]]]),
        "average_densities": average_densities,
    }


def test_create_inhibitory_neuron_densities_3():
    data = get_initialization_data_3()
    densities = tested.create_inhibitory_neuron_densities(
        data["hierarchy"],
        data["annotation"],
        data["voxel_volume"],
        data["neuron_density"],
        data["average_densities"],
    )

    expected = {
        "gad67+": np.array([[[3.5, 3.0]]]),
        "pv+": np.array([[[1.5, 1.0]]]),
        "sst+": np.array([[[1.0, 1.0]]]),
        "vip+": np.array([[[1.0, 1.0]]]),
    }

    for cell_type, density in densities.items():
        npt.assert_array_almost_equal(density, expected[cell_type])


def test_error_on_sigmas_inconsistencies():
    # Test that an error is raised when our simplifying assumption between the x's and the deltas
    # is violated
    data = get_initialization_data_3()
    data["average_densities"].at["root", "pv+_standard_deviation"] = 0.0
    with pytest.raises(AtlasDensitiesError, match="not certain"):
        tested.create_inhibitory_neuron_densities(
            data["hierarchy"],
            data["annotation"],
            data["voxel_volume"],
            data["neuron_density"],
            data["average_densities"],
        )


def get_initialization_data_4():
    hierarchy = {
        "id": 8,
        "name": "root",
        "children": [
            {
                "id": 1,
                "name": "A",
                "children": [
                    {"id": 2, "name": "B", "children": []},
                    {"id": 3, "name": "C", "children": []},
                ],
            }
        ],
    }

    average_densities = pd.DataFrame(
        {
            "gad67+": [1.0, 1.0, 1.5, 1.0],
            "gad67+_standard_deviation": [1.0, 1.0, 1.0, 1.0],
            "pv+": [1.0 / 2.0, 1.0 / 2.0, 1.0, 1.0 / 2.0],
            "pv+_standard_deviation": [1.0, 0.0, 1.0, 1.0],
            "sst+": [1.0 / 3.0, 1.0 / 2.0, 1.0 / 2.0, 1.0 / 3.0],
            "sst+_standard_deviation": [1.0, 1.0, 0.0, 1.0],
        },
        index=["A", "B", "C", "root"],
    )

    return {
        "hierarchy": hierarchy,
        "annotation": np.array([[[1, 1, 2, 2, 3, 3]]]),
        "voxel_volume": 1.0,
        "neuron_density": np.array([[[3.0, 3.0, 3.0, 3.0, 3.0, 3.0]]]),
        "average_densities": average_densities,
    }


def test_create_inhibitory_neuron_densities_4():
    data = get_initialization_data_4()
    densities = tested.create_inhibitory_neuron_densities(
        data["hierarchy"],
        data["annotation"],
        data["voxel_volume"],
        data["neuron_density"],
        data["average_densities"],
    )

    expected = {
        "gad67+": np.array([[[0.5, 0.5, 1.0, 1.0, 1.5, 1.5]]]),
        "pv+": np.array([[[0.0, 0.0, 0.5, 0.5, 1.0, 1.0]]]),
        "sst+": np.array([[[0.0, 0.0, 0.5, 0.5, 0.5, 0.5]]]),
    }

    for cell_type, density in densities.items():
        npt.assert_array_almost_equal(density, expected[cell_type])


def get_initialization_data():
    hierarchy = {
        "id": 8,
        "name": "root",
        "children": [
            {
                "id": 1,
                "name": "A",
                "children": [
                    {"id": 2, "name": "B", "children": []},
                    {"id": 3, "name": "C", "children": []},
                ],
            }
        ],
    }

    average_densities = pd.DataFrame(
        {
            "gad67+": [3.0, 3.0, 3.0, 3.0],
            "gad67+_standard_deviation": [1.0, 1.0, 1.0, 1.0],
            "pv+": [1.25, 1.5, 1.0, 1.25],
            "pv+_standard_deviation": [1.0, 1.0, 1.0, 1.0],
            "sst+": [1.25, 1.5, 1.0, 1.25],
            "sst+_standard_deviation": [1.0, 1.0, 0.0, 1.0],
            "vip+": [1.5, 1.5, 1.5, 1.5],
            "vip+_standard_deviation": [0.0, 0.0, 0.0, 1.0],
        },
        index=["A", "B", "C", "root"],
    )

    return {
        "hierarchy": hierarchy,
        "annotation": np.array([[[1, 2, 2, 3, 3]]]),
        "voxel_volume": 2.0,
        "neuron_density": np.array([[[3.0, 3.0, 3.0, 3.0, 3.0]]]),
        "average_densities": average_densities,
    }


def test_create_inhibitory_neuron_densities():
    data = get_initialization_data()
    densities = tested.create_inhibitory_neuron_densities(
        data["hierarchy"],
        data["annotation"],
        data["voxel_volume"],
        data["neuron_density"],
        data["average_densities"],
    )

    sum_ = np.zeros_like(data["neuron_density"])
    for cell_type, density in densities.items():
        assert np.all(density >= 0.0)
        if cell_type != "gad67+":
            sum_ += density

    assert np.all(densities["gad67+"] <= data["neuron_density"])
    assert np.all(sum_ <= data["neuron_density"])
    assert np.all(sum_ <= densities["gad67+"] + 1e-6)
