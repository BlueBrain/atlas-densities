"""
Unit tests for inhibitory cell density computation
"""
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import numpy.testing as npt
import pytest
from voxcell import RegionMap

import atlas_densities.densities.inhibitory_neuron_density as tested
from atlas_densities.exceptions import AtlasDensitiesError

TESTS_PATH = Path(__file__).parent.parent


def get_intensity_data():
    region_masks = {
        "Cerebellum group": np.array([[[False, False, True, False, False]]]),
        "Isocortex group": np.array([[[False, False, False, True, False]]]),
        "Rest": np.array([[[False, False, False, False, True]]]),
    }
    inhibitory_data = {
        "proportions": {
            "Cerebellum group": 0.2,
            "Isocortex group": 0.2,
            "Rest": 0.6,
        },
        "neuron_count": 6,
        "region_masks": region_masks,
    }
    return {
        "region_masks": region_masks,
        "inhibitory_data": inhibitory_data,
        "gad1": np.array([[[0.0, 0.0, 0.5, 0.25, 0.25]]]),
        "nrn1": np.array([[[0.0, 0.0, 0.4, 0.35, 0.25]]]),
    }


def test_compute_inhibitory_neuron_intensity():
    # Only the global inhibitory neuron proportion is provided
    gad = np.array([[[0.0, 0.0, 0.5, 0.5]]])
    nrn1 = np.array([[[0.0, 0.0, 0.4, 0.6]]])
    expected = np.array([[[0.0, 0.0, 1.0, 23.0 / 27.0]]])
    inhibitory_data = {
        "proportions": {"whole brain": 0.6},
        "neuron_count": 10,
        "region_masks": {"whole brain": np.ones((1, 1, 4), dtype=bool)},
    }
    actual = tested.compute_inhibitory_neuron_intensity(gad, nrn1, inhibitory_data)
    npt.assert_almost_equal(actual, expected)

    # Inhibitory neuron proportions are provided for the 3 different groups
    data = get_intensity_data()
    actual = tested.compute_inhibitory_neuron_intensity(
        data["gad1"],
        data["nrn1"],
        inhibitory_data=data["inhibitory_data"],
    )
    expected = np.array([[[0.0, 0.0, 25.0 / 63.0, 25.0 / 99.0, 1.0]]])
    npt.assert_almost_equal(actual, expected)


def test_compute_inhibitory_neuron_density():
    annotation = np.array([[[1, 2, 6, 3, 4]]], dtype=np.uint32)
    voxel_volume = (25**3) / 1e9  # mm^3
    neuron_density = np.array([[[0.0, 0.0, 4.0, 4.0, 2.0]]]) / voxel_volume
    group_ids = {
        "Purkinje layer": set({1, 2}),
        "Cerebellar cortex": set({1, 6}),
        "Molecular layer": set({2, 6}),
    }
    data = get_intensity_data()

    with patch(
        "atlas_densities.densities.inhibitory_neuron_density.get_group_ids",
        return_value=group_ids,
    ):
        with patch(
            "atlas_densities.densities.inhibitory_neuron_density.get_region_masks",
            return_value=data["region_masks"],
        ):
            with patch(
                "atlas_densities.densities.inhibitory_neuron_density.compensate_cell_overlap",
                side_effect=[data["gad1"], data["nrn1"]],
            ):
                # have any empty region map
                region_map = Mock()
                region_map.find = lambda name, attr, with_descendants: set()

                inhibitory_neuron_density = tested.compute_inhibitory_neuron_density(
                    region_map,
                    annotation,
                    voxel_volume,
                    data["gad1"],
                    data["nrn1"],
                    neuron_density,
                    inhibitory_data=data["inhibitory_data"],
                )
                expected = np.array([[[0.0, 0.0, 4.0, 25.0 / 62.0, 99.0 / 62.0]]]) / voxel_volume
                npt.assert_almost_equal(inhibitory_neuron_density, expected)


def test_compute_inhibitory_neuron_density_exception():
    # At least one of `inhibitory_proportion` or `inhibitory_data` must be provided
    with pytest.raises(AtlasDensitiesError) as error_:
        tested.compute_inhibitory_neuron_density(
            {},
            np.array([[[1]]], dtype=np.uint32),
            1.0,
            np.array([[[1]]], dtype=float),
            np.array([[[1]]], dtype=float),
            np.array([[[1]]], dtype=float),
        )
    assert "inhibitory_proportion" in str(error_)
    assert "inhibitory_data" in str(error_)


def get_inhibitory_neuron_input_data(neuron_count):
    shape = (20, 20, 20)
    annotation = np.arange(8000).reshape(shape)
    voxel_volume = (25**3) / 1e9
    rng = np.random.default_rng(seed=42)
    neuron_density = rng.random(annotation.shape)
    neuron_density[0][0][0] = 0.0  # Neuron density outside the brain should be null.
    neuron_density = (2.5 * neuron_count * neuron_density / np.sum(neuron_density)) / voxel_volume
    gad1_expr = rng.random(annotation.shape)
    gad1_expr[0][0][0] = 1e-5
    nrn1_expr = rng.random(annotation.shape)
    nrn1_expr[0][0][0] = 1e-5
    inhibitory_data = {
        "proportions": {
            "Cerebellum group": 0.4,
            "Isocortex group": 0.35,
            "Rest": 0.25,
        },
        "neuron_count": neuron_count,
    }
    return {
        "annotation": annotation,
        "neuron_density": neuron_density,
        "voxel_volume": voxel_volume,
        "gad1": gad1_expr,
        "nrn1": nrn1_expr,
        "inhibitory_data": inhibitory_data,
    }


def test_compute_inhibitory_density_large_input():
    region_map = RegionMap.load_json(Path(TESTS_PATH, "1.json"))
    neuron_count = 30000
    data = get_inhibitory_neuron_input_data(neuron_count)

    inhibitory_neuron_density = tested.compute_inhibitory_neuron_density(
        region_map,
        data["annotation"],
        data["voxel_volume"],
        data["gad1"],
        data["nrn1"],
        data["neuron_density"],
        inhibitory_data=data["inhibitory_data"],
    )

    assert np.all(inhibitory_neuron_density <= data["neuron_density"])
    npt.assert_allclose(
        np.sum(inhibitory_neuron_density) * data["voxel_volume"], neuron_count, rtol=1e-3
    )
    assert inhibitory_neuron_density.dtype == np.float64
