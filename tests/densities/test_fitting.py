"""
Unit tests for overall cell density computation
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from voxcell import RegionMap

import atlas_densities.densities.fitting as tested
from atlas_densities.densities import utils
from atlas_densities.exceptions import AtlasDensitiesError, AtlasDensitiesWarning

TESTS_PATH = Path(__file__).parent.parent


@pytest.fixture
def group_ids_config():
    return utils.load_json(utils.GROUP_IDS_PATH)


def test_create_dataframe_from_known_densities():
    average_densities = pd.DataFrame(
        {
            "brain_region": ["Isocortex", "Isocortex", "Isocortex", "Cerebellum", "Cerebrum"],
            "measurement": [1.0, 2.0, 3.0, 4.0, 5.0],
            "standard_deviation": [0.1, 0.2, 0.3, 0.4, 0.5],
            "cell_type": ["PV+", "SST+", "PV+", "inhibitory neuron", "VIP+"],
        }
    )
    region_names = ["Cerebellum", "Cerebrum", "Isocortex", "Thalamus"]
    expected = pd.DataFrame(
        {
            "inhibitory_neuron": [4.0, np.nan, np.nan, np.nan],
            "inhibitory_neuron_standard_deviation": [0.4, np.nan, np.nan, np.nan],
            "pv+": [np.nan, np.nan, 2.0, np.nan],
            "pv+_standard_deviation": [np.nan, np.nan, 0.2, np.nan],
            "sst+": [np.nan, np.nan, 2.0, np.nan],
            "sst+_standard_deviation": [np.nan, np.nan, 0.2, np.nan],
            "vip+": [np.nan, 5.0, np.nan, np.nan],
            "vip+_standard_deviation": [np.nan, 0.5, np.nan, np.nan],
        },
        index=region_names,
    )
    actual = tested.create_dataframe_from_known_densities(region_names, average_densities)
    pdt.assert_frame_equal(actual, expected)


@pytest.fixture
def region_map():
    hierarchy = {
        "id": 8,
        "name": "Basic cell groups and regions",
        "acronym": "grey",
        "children": [
            {
                "id": 920,
                "acronym": "CENT",
                "name": "Central lobule",
                "children": [
                    {
                        "id": 976,
                        "acronym": "CENT2",
                        "name": "Lobule II",
                        "children": [
                            {
                                "id": 10709,
                                "acronym": "CENT2pu",
                                "name": "Lobule II, Purkinje layer",
                                "children": [],
                            },
                            {
                                "id": 10708,
                                "acronym": "CENT2gr",
                                "name": "Lobule II, granular layer",
                                "children": [],
                            },
                        ],
                    }
                ],
            },
            {
                "id": 936,
                "acronym": "DEC",
                "name": "Declive (VI)",
                "children": [
                    {
                        "id": 10725,
                        "acronym": "DECmo",
                        "name": "Declive (VI), molecular layer",
                        "children": [],
                    },
                    {
                        "id": 10724,
                        "acronym": "DECpu",
                        "name": "Declive (VI), Purkinje layer",
                        "children": [],
                    },
                ],
            },
        ],
    }

    return RegionMap.from_dict(hierarchy)


@pytest.fixture
def hierarchy_info(region_map):
    return utils.get_hierarchy_info(region_map)


def test_fill_in_homogenous_regions(hierarchy_info):
    homogenous_regions = pd.DataFrame(
        {
            "brain_region": [
                "Lobule II",
                "Declive (VI)",
                "Declive (VI), molecular layer",
                "Lobule II, Purkinje layer",
            ],
            "cell_type": ["inhibitory", "excitatory", "excitatory", "inhibitory"],
        }
    )
    annotation = np.array([[[936, 936, 10724, 976, 10709, 10708, 10725, 920, 920]]])
    neuron_density = np.array([[[0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7]]])
    assert np.all(
        hierarchy_info["brain_region"]
        == [
            "Basic cell groups and regions",
            "Central lobule",
            "Declive (VI)",
            "Lobule II",
            "Lobule II, granular layer",
            "Lobule II, Purkinje layer",
            "Declive (VI), Purkinje layer",
            "Declive (VI), molecular layer",
        ]
    )
    input_densities_dataframe = pd.DataFrame(
        {
            "inhibitory_neuron": np.full((8,), np.nan),
            "inhibitory_neuron_standard_deviation": np.full((8,), np.nan),
            "pv+": np.full((8,), np.nan),
            "pv+_standard_deviation": np.full((8,), np.nan),
            "sst+": [0.1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            "sst+_standard_deviation": [
                0.1,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            "vip+": np.full((8,), np.nan),
            "vip+_standard_deviation": np.full((8,), np.nan),
        },
        index=hierarchy_info["brain_region"],
    )
    expected = pd.DataFrame(
        {
            "inhibitory_neuron": [np.nan, np.nan, 0.0, 0.4, 0.5, 0.4, 0.0, 0.0],
            "inhibitory_neuron_standard_deviation": [np.nan, np.nan, 0.0, 0.4, 0.5, 0.4, 0.0, 0.0],
            "pv+": np.full((8,), np.nan),
            "pv+_standard_deviation": np.full((8,), np.nan),
            "sst+": [0.1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            "sst+_standard_deviation": [
                0.1,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            "vip+": np.full((8,), np.nan),
            "vip+_standard_deviation": np.full((8,), np.nan),
        },
        index=hierarchy_info["brain_region"],
    )

    tested.fill_in_homogenous_regions(
        homogenous_regions, annotation, neuron_density, input_densities_dataframe, hierarchy_info
    )

    pdt.assert_frame_equal(input_densities_dataframe, expected)

    cell_density_stddevs = {
        "Lobule II": 0.4,
        "Lobule II, Purkinje layer": 0.4,
        "Lobule II, granular layer": 0.5,
    }
    tested.fill_in_homogenous_regions(
        homogenous_regions,
        annotation,
        neuron_density,
        input_densities_dataframe,
        hierarchy_info,
        cell_density_stddevs,
    )
    pdt.assert_frame_equal(input_densities_dataframe, expected)


def test_compute_average_intensity():
    volume_mask = np.zeros((2, 2, 2), dtype=bool)
    intensity = np.array([[[0.1, 0.1], [0.1, 0.2]], [[0.0, 0.1], [0.1, 0.1]]])
    actual = tested.compute_average_intensity(intensity, volume_mask)
    assert actual == 0.0

    volume_mask = np.ones((2, 2, 2), dtype=bool)
    actual = tested.compute_average_intensity(intensity, volume_mask)
    assert actual == 0.1

    volume_mask = np.array([[[True, True], [False, False]], [[False, False], [True, True]]])
    actual = tested.compute_average_intensity(intensity, volume_mask)
    assert actual == 0.1

    volume_mask = np.array([[[False, False], [True, True]], [[True, True], [False, False]]])
    actual = tested.compute_average_intensity(intensity, volume_mask, slices=[0])
    assert np.allclose(actual, 0.3 / 2.0)
    actual = tested.compute_average_intensity(intensity, volume_mask, slices=[1])
    assert np.allclose(actual, 0.1 / 2.0)
    actual = tested.compute_average_intensity(intensity, volume_mask, slices=[-1, 1, 3])
    assert np.allclose(actual, 0.1 / 2.0)
    actual = tested.compute_average_intensity(intensity, volume_mask, slices=[-1, 3])
    assert actual == 0


def test_compute_average_intensities(region_map, hierarchy_info):
    annotation = np.array(
        [[[0, 976], [976, 936]], [[976, 936], [936, 936]]]  # 976 = Lobule II, 936 = "Declive (VI)""
    )
    marker_volumes = {
        "gad67": {
            "intensity": np.array([[[1.0, 1.0], [1.0, 2.0]], [[1.0, 2.0], [2.0, 2.0]]]),
            "slices": [1],
        },
        "pv": {
            "intensity": np.array([[[0.5, 2.0], [2.0, 1.0]], [[2.0, 1.0], [1.0, 1.0]]]),
            "slices": None,
        },
    }
    assert np.all(
        hierarchy_info["brain_region"]
        == [
            "Basic cell groups and regions",
            "Central lobule",
            "Declive (VI)",  # 936
            "Lobule II",  # 976
            "Lobule II, granular layer",
            "Lobule II, Purkinje layer",
            "Declive (VI), Purkinje layer",
            "Declive (VI), molecular layer",
        ]
    )
    expected = pd.DataFrame(
        {
            "gad67": [7.0 / 4.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "pv": [10.0 / 7.0, 2.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=hierarchy_info["brain_region"],
    )

    actual = tested.compute_average_intensities(
        annotation, marker_volumes, hierarchy_info, region_map
    )
    pdt.assert_frame_equal(actual, expected)


def test_linear_fitting_xy():
    min_data_points = 1
    actual = tested.linear_fitting_xy(
        [0.0, 1.0, 2.0], [0.0, 2.0, 4.0], [1.0, 1.0, 1.0], min_data_points=min_data_points
    )
    assert np.allclose(actual["coefficient"], 2.0)
    assert np.allclose(actual["r_square"], 1.0)
    assert not np.isinf(actual["standard_deviation"])

    actual = tested.linear_fitting_xy(
        [0.0, 1.0, 2.0], [0.0, 1.0, 4.0], [1.0, 1.0, 1e-5], min_data_points=min_data_points
    )
    assert np.allclose(actual["coefficient"], 2.0)
    assert not np.isinf(actual["standard_deviation"])
    assert np.allclose(actual["r_square"], 0.89286)

    actual = tested.linear_fitting_xy(
        [0.0, 1.0, 2.0], [0.0, 2.0, 4.0], [1.0, 0.0, 1.0], min_data_points=min_data_points
    )
    assert np.allclose(actual["coefficient"], 2.0)
    assert not np.isinf(actual["standard_deviation"])
    assert np.allclose(actual["r_square"], 1.0)


def get_fitting_input_data_(hierarchy_info):
    intensities = pd.DataFrame(
        {
            "gad67": [7.0 / 4.0, 1.0, 2.0, 1.0, np.nan, np.nan, np.nan, np.nan],
            "pv": [10.0 / 7.0, 2.0, 1.0, 2.0, np.nan, np.nan, np.nan, np.nan],
        },
        index=hierarchy_info["brain_region"],
    )
    densities = pd.DataFrame(
        {
            "gad67+": [7.0 / 2.0, 2.0, 4.0, 2.0, np.nan, np.nan, np.nan, np.nan],
            "gad67+_standard_deviation": [1.0, 0.0, 3.0, 0.0, np.nan, np.nan, np.nan, np.nan],
            "pv+": [30.0 / 7.0, 6.0, 3.0, 6.0, np.nan, np.nan, np.nan, np.nan],
            "pv+_standard_deviation": [2.0, 1.0, 3.0, 4.0, np.nan, np.nan, np.nan, np.nan],
        },
        index=hierarchy_info["brain_region"],
    )
    data = {
        "groups": {"Whole": {"Lobule II", "Declive (VI)"}, "Central lobule": {"Lobule II"}},
        "intensities": intensities,
        "densities": densities,
    }

    return data


def test_compute_fitting_coefficients(hierarchy_info):
    data = get_fitting_input_data_(hierarchy_info)

    actual = tested.compute_fitting_coefficients(
        data["groups"],
        data["intensities"],
        data["densities"],
        min_data_points=1,
    )

    for group_name in ["Whole", "Central lobule"]:
        assert actual[group_name]["gad67+"]["coefficient"] == 2.0
        assert actual[group_name]["pv+"]["coefficient"] == 3.0
        assert not np.isinf(actual[group_name]["gad67+"]["standard_deviation"])
        assert not np.isnan(actual[group_name]["gad67+"]["standard_deviation"])
        assert not np.isinf(actual[group_name]["pv+"]["standard_deviation"])
        assert not np.isnan(actual[group_name]["pv+"]["standard_deviation"])
    assert actual["Whole"]["pv+"]["r_square"] == 1.0
    assert actual["Whole"]["gad67+"]["r_square"] == 1.0
    assert np.isnan(actual["Central lobule"]["pv+"]["r_square"])  # only one value so no variation.
    assert np.isnan(actual["Central lobule"]["gad67+"]["r_square"])


def test_compute_fitting_coefficients_exceptions(hierarchy_info):
    data = get_fitting_input_data_(hierarchy_info)
    data["densities"].drop(index=["Central lobule"], inplace=True)

    with pytest.raises(AtlasDensitiesError):
        tested.compute_fitting_coefficients(
            data["groups"],
            data["intensities"],
            data["densities"],
            min_data_points=1,
        )

    data = get_fitting_input_data_(hierarchy_info)
    data["densities"].drop(columns=["pv+"], inplace=True)

    with pytest.raises(AtlasDensitiesError):
        tested.compute_fitting_coefficients(
            data["groups"],
            data["intensities"],
            data["densities"],
            min_data_points=1,
        )

    data = get_fitting_input_data_(hierarchy_info)
    data["densities"].at["Lobule II", "pv+_standard_deviation"] = np.nan
    with pytest.raises(AssertionError):
        tested.compute_fitting_coefficients(
            data["groups"],
            data["intensities"],
            data["densities"],
            min_data_points=1,
        )


@pytest.fixture
def fitting_coefficients():
    return {
        "Lobule II": {
            "gad67+": {"coefficient": 2.0, "standard_deviation": 0.5},
            "pv+": {"coefficient": 3.0, "standard_deviation": 1.0},
        },
        "Declive (VI)": {
            "gad67+": {"coefficient": 2.0, "standard_deviation": 1.0},
            "pv+": {"coefficient": 3.0, "standard_deviation": 0.5},
        },
    }


def test_fit_unknown_densities(hierarchy_info, fitting_coefficients):
    groups = {
        "Lobule II": {"Lobule II", "Lobule II, granular layer", "Lobule II, Purkinje layer"},
        "Declive (VI)": {
            "Declive (VI)",
            "Declive (VI), Purkinje layer",
            "Declive (VI), molecular layer",
        },
    }
    intensities = pd.DataFrame(
        {
            "gad67": [7.0 / 4.0, 1.0, 2.0, 1.0, 1.5, np.nan, 1.0, np.nan],
            "pv": [10.0 / 7.0, 2.0, 1.0, 2.0, 0.75, np.nan, 0.5, np.nan],
        },
        index=hierarchy_info["brain_region"],
    )
    densities = pd.DataFrame(
        {
            "gad67+": [7.0 / 2.0, 2.0, 4.0, 2.0, np.nan, np.nan, np.nan, np.nan],
            "gad67+_standard_deviation": [1.0, 0.0, 3.0, 0.0, np.nan, np.nan, np.nan, np.nan],
            "pv+": [30.0 / 7.0, 6.0, 3.0, 6.0, np.nan, np.nan, np.nan, np.nan],
            "pv+_standard_deviation": [2.0, 1.0, 3.0, 4.0, np.nan, np.nan, np.nan, np.nan],
        },
        index=hierarchy_info["brain_region"],
    )

    expected = pd.DataFrame(
        {
            "gad67+": [7.0 / 2.0, 2.0, 4.0, 2.0, 3.0, np.nan, 2.0, np.nan],
            "gad67+_standard_deviation": [1.0, 0.0, 3.0, 0.0, 1.5, np.nan, 2.0, np.nan],
            "pv+": [30.0 / 7.0, 6.0, 3.0, 6.0, 2.25, np.nan, 1.5, np.nan],
            "pv+_standard_deviation": [2.0, 1.0, 3.0, 4.0, 2.25, np.nan, 0.75, np.nan],
        },
        index=hierarchy_info["brain_region"],
    )
    tested.fit_unknown_densities(groups, intensities, densities, fitting_coefficients)
    pdt.assert_frame_equal(densities, expected)


def get_hierarchy():
    return {
        "id": 8,
        "name": "root",
        "acronym": "root",
        "children": [
            {
                "id": 315,
                "acronym": "Isocortex",
                "name": "Isocortex",
                "children": [],
            },
            {
                "id": 512,
                "acronym": "CB",
                "name": "Cerebellum",
                "children": [],
            },
            {
                "id": 549,
                "acronym": "TH",
                "name": "Thalamus",
                "children": [],
            },
            {
                "id": 1089,
                "acronym": "HPF",
                "name": "Hippocampal formation",
                "children": [],
            },
            {
                "id": 42,
                "acronym": "PKL",
                "name": "Purkinje layer",
                "children": [],
            },
            {
                "id": 38,
                "acronym": "CCTX",
                "name": "Cerebellar cortex",
                "children": [
                    {
                        "id": 37,
                        "acronym": "MLL",
                        "name": "molecular layer",
                        "children": [],
                    },
                ],
            },
        ],
    }


def get_fitting_input_data():
    # The first three dicts are only used to test app/cell_densities
    realigned_slices = {"1": [1], "2": None}
    slice_map = {"gad67": "1", "pv": "2"}
    cell_density_stddevs = {
        "|root|Basic cell groups and regions|Brain stem|Interbrain|Thalamus": 1.0,
    }

    annotation = np.array([[[0, 315], [315, 512]], [[512, 549], [1089, 315]]])
    neuron_density = np.array([[[1.0, 1.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]]])
    gene_marker_volumes = {
        "gad67": {
            "intensity": np.array([[[0.0, 2.0], [2.0, 2.0]], [[2.0, 2.0], [0.0, 2.0]]]),
            "slices": [1],
        },
        "pv": {
            "intensity": np.array([[[3.0, 3.0], [0.0, 3.0]], [[3.0, 3.0], [3.0, 3.0]]]),
            "slices": None,
        },
    }
    region_names = ["root", "Isocortex", "Cerebellum", "Thalamus"]
    densities = pd.DataFrame(
        {
            "brain_region": region_names,
            "cell_type": ["inhibitory neuron", "pv+", "inhibitory neuron", "pv+"],
            "measurement": [10.0, 5.0, 3.0, 2.0],
            "measurement_type": ["cell density", "cell density", "cell density", "cell density"],
            "standard_deviation": [1.0, 2.0, 0.0, 1.0],
        },
        index=region_names,
    )
    homogenous_regions = pd.DataFrame(
        {
            "brain_region": [
                "Thalamus",
            ],
            "cell_type": ["inhibitory"],
        }
    )

    return {
        "realigned_slices": realigned_slices,  # only useful for testing app/cell_densities
        "slice_map": slice_map,  # only useful for testing app/cell_densities
        "cell_density_stddevs": cell_density_stddevs,  # only useful for testing app/cell_densities
        "hierarchy": get_hierarchy(),
        "annotation": annotation,
        "neuron_density": neuron_density,
        "gene_marker_volumes": gene_marker_volumes,
        "average_densities": densities,
        "homogenous_regions": homogenous_regions,
    }


def test_linear_fitting(group_ids_config):
    data = get_fitting_input_data()

    with warnings.catch_warnings(record=True) as warnings_:
        densities, fitting_maps = tested.linear_fitting(
            RegionMap.from_dict(data["hierarchy"]),
            data["annotation"],
            data["neuron_density"],
            data["gene_marker_volumes"],
            data["average_densities"],
            data["homogenous_regions"],
            group_ids_config=group_ids_config,
            min_data_points=1,
        )
        warnings_ = [w for w in warnings_ if isinstance(w.message, AtlasDensitiesWarning)]
        # Three warnings for recording NaN coefficients, three warnings for using them
        assert len(warnings_) == 9

    assert np.allclose(fitting_maps["Rest"]["pv+"]["coefficient"], 2.0 / 3.0)

    assert np.allclose(densities.at["Thalamus", "gad67+"], 0.0)
    assert np.isnan(densities.at["Hippocampal formation", "gad67+"])
    assert np.allclose(densities.at["Thalamus", "pv+"], 2.0)
    assert np.allclose(densities.at["Hippocampal formation", "pv+"], 2.0)
    assert np.isnan(densities.at["Hippocampal formation", "gad67+_standard_deviation"])
    assert densities.at["Hippocampal formation", "pv+_standard_deviation"] >= 0.0
    assert not np.isinf(densities.at["Hippocampal formation", "pv+_standard_deviation"])


def test_linear_fitting_exception_average_densities(group_ids_config):
    data = get_fitting_input_data()
    data["average_densities"].at["Thalamus", "measurement_type"] = "volume"

    with pytest.raises(AtlasDensitiesError, match="unexpected measurement type.*volume.*"):
        tested.linear_fitting(
            RegionMap.from_dict(data["hierarchy"]),
            data["annotation"],
            data["neuron_density"],
            data["gene_marker_volumes"],
            data["average_densities"],
            data["homogenous_regions"],
            group_ids_config=group_ids_config,
        )

    data["average_densities"].at["Thalamus", "measurement_type"] = "cell density"
    data["average_densities"].at["Thalamus", "measurement"] = -1.0

    with pytest.raises(AtlasDensitiesError, match="negative measurement"):
        tested.linear_fitting(
            RegionMap.from_dict(data["hierarchy"]),
            data["annotation"],
            data["neuron_density"],
            data["gene_marker_volumes"],
            data["average_densities"],
            data["homogenous_regions"],
            group_ids_config=group_ids_config,
        )


def test_linear_fitting_exception_homogenous_regions(group_ids_config):
    data = get_fitting_input_data()
    data["homogenous_regions"].at["Thalamus", "cell_type"] = "Inhibitory"

    with pytest.raises(AtlasDensitiesError, match="unexpected cell type"):
        tested.linear_fitting(
            RegionMap.from_dict(data["hierarchy"]),
            data["annotation"],
            data["neuron_density"],
            data["gene_marker_volumes"],
            data["average_densities"],
            data["homogenous_regions"],
            group_ids_config=group_ids_config,
        )


def test__get_group_names(group_ids_config):
    region_map = RegionMap.load_json(Path(TESTS_PATH, "1.json"))
    ret = tested._get_group_names(region_map, group_ids_config=group_ids_config)
    for k, length in (("Isocortex group", 409), ("Cerebellum group", 88), ("Rest", 825)):
        assert length == len(ret[k])
