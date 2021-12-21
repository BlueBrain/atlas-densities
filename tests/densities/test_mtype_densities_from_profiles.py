import tempfile
import warnings
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest
from voxcell import RegionMap, VoxelData  # type: ignore

import atlas_densities.densities.mtype_densities_from_profiles as tested
from atlas_densities.exceptions import AtlasDensitiesError

TESTS_PATH = Path(__file__).parent.parent
DATA_PATH = TESTS_PATH / "densities" / "data"


def test_slice_layer():
    raw = np.array([[[0, 1, 2, 3, 0]]], dtype=int)
    annotation = VoxelData(raw, (1.0, 1.0, 1.0))
    vector_field = np.array([[[[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]]]).astype(np.float32)
    mask = np.array([[[0, 1, 1, 1, 0]]]).astype(bool)
    actual = tested.DensityProfileCollection.slice_layer(mask, annotation, vector_field, 3)
    npt.assert_array_equal(actual, np.array([[[0, 1, 2, 3, 0]]]))


def create_density_profile_collection(mapping_path="mapping.tsv"):
    return tested.DensityProfileCollection.load(
        DATA_PATH / "meta" / mapping_path,
        DATA_PATH / "meta" / "layers.tsv",
        DATA_PATH / "mtypes",
    )


def create_slicer_data():
    annotation = np.array([[[3, 3, 3, 2, 2]]])
    hierarchy = {
        "acronym": "Isocortex",
        "name": "Isocortex",
        "id": 0,
        "children": [
            {
                "acronym": "L2",
                "name": "layer_2",
                "id": 2,
                "children": [],
            },
            {
                "acronym": "L3",
                "name": "layer_3",
                "id": 3,
                "children": [],
            },
        ],
    }
    metadata = {
        "region": {
            "name": "Isocortex",
            "query": "Isocortex",
            "attribute": "acronym",
            "with_descendants": True,
        },
        "layers": {
            "names": ["layer_2", "layer_3"],
            "queries": ["L2", "L3"],
            "attribute": "acronym",
            "with_descendants": True,
        },
    }
    direction_vectors = np.array(
        [[[[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]]], dtype=float
    )

    return {
        "annotation": VoxelData(annotation, offset=[1.0, 2.0, 3.0], voxel_dimensions=[1.0] * 3),
        "hierarchy": hierarchy,
        "region_map": RegionMap.from_dict(hierarchy),
        "metadata": metadata,
        "direction_vectors": direction_vectors,
    }


def create_slice_voxel_indices():
    return [([0], [0], [slice_index]) for slice_index in range(5)]


def create_excitatory_neuron_density():
    return VoxelData(
        np.array([[[3.0, 1.0, 4.0, 2.0, 2.0]]]), offset=[1.0, 2.0, 3.0], voxel_dimensions=[1.0] * 3
    )


def create_inhibitory_neuron_density():
    return VoxelData(
        np.array([[[4.0, 3.0, 1.0, 2.0, 1.0]]]), offset=[1.0, 2.0, 3.0], voxel_dimensions=[1.0] * 3
    )


def create_expected_cell_densities():
    return {
        "L2_TPC:A": np.array([[[0.0, 0.0, 0.0, 2.0 * 10100.0 / 15800.0, 2.0 * 11000.0 / 17200.0]]]),
        "L23_BP": np.array([[[0.0, 3.0, 1.0, 2.0, 1.0]]]),
        "L3_TPC:B": np.array(
            [
                [
                    [
                        3.0 * 20500.0 / 82000.0,
                        1.0 * 50200.0 / 107600.0,
                        4.0 * 36500.0 / 93000.0,
                        0.0,
                        0.0,
                    ]
                ]
            ]
        ),
    }


@pytest.fixture
def expected_profile_data():
    return {
        "layer_2": {
            "inhibitory": pd.DataFrame({"L23_BP": [1.0, 1.0]}).rename(index={0: 3, 1: 4}),
            "excitatory": pd.DataFrame(
                {
                    "L2_IPC": [5700.0 / 15800.0, 6200.0 / 17200.0],
                    "L2_TPC:A": [10100.0 / 15800.0, 11000.0 / 17200.0],
                }
            ).rename(index={0: 3, 1: 4}),
        },
        "layer_3": {
            "inhibitory": pd.DataFrame({"L23_BP": [0.0, 1.0, 1.0]}),
            "excitatory": pd.DataFrame(
                {
                    "L3_TPC:A": [
                        61500.0 / 82000.0,
                        57400.0 / 107600.0,
                        56500.0 / 93000.0,
                    ],
                    "L3_TPC:B": [
                        20500.0 / 82000.0,
                        50200.0 / 107600.0,
                        36500.0 / 93000.0,
                    ],
                }
            ),
        },
    }


def test_density_profile_collection_loading(expected_profile_data):
    density_profile_collection = None
    with warnings.catch_warnings(record=True) as w:
        density_profile_collection = create_density_profile_collection()
        msg = str(w[0].message)
        assert "No inhibitory cells assigned to slice 0 of layer_3" in msg

    for layer in ["layer_2", "layer_3"]:
        for synapse_class in ["excitatory", "inhibitory"]:
            pdt.assert_frame_equal(
                density_profile_collection.profile_data[layer][synapse_class],
                expected_profile_data[layer][synapse_class],
            )


def test_density_profile_collection_loading_exc_only(expected_profile_data):
    density_profile_collection = None
    with warnings.catch_warnings(record=True) as w:
        density_profile_collection = create_density_profile_collection(
            mapping_path="mapping_exc_only.tsv"
        )
        assert not w

    for layer in ["layer_2", "layer_3"]:
        expected_profile_data[layer]["inhibitory"] = pd.DataFrame()

    for layer in ["layer_2", "layer_3"]:
        for synapse_class in ["excitatory", "inhibitory"]:
            pdt.assert_frame_equal(
                density_profile_collection.profile_data[layer][synapse_class],
                expected_profile_data[layer][synapse_class],
            )


def test_compute_layer_slice_voxel_indices_assert():
    data = create_slicer_data()
    density_profile_collection = create_density_profile_collection(
        mapping_path="mapping_exc_only.tsv"
    )
    # Create empty slices due to oversized direction vectors
    data["direction_vectors"] = 100.0 * data["direction_vectors"]
    with pytest.raises(AtlasDensitiesError):
        density_profile_collection.compute_layer_slice_voxel_indices(
            data["annotation"], data["region_map"], data["metadata"], data["direction_vectors"]
        )


def test_compute_layer_slice_voxel_indices():
    data = create_slicer_data()
    density_profile_collection = create_density_profile_collection()
    actual_voxel_indices = density_profile_collection.compute_layer_slice_voxel_indices(
        data["annotation"], data["region_map"], data["metadata"], data["direction_vectors"]
    )

    expected = create_slice_voxel_indices()
    for slice_index in range(0, 5):
        npt.assert_array_equal(actual_voxel_indices[slice_index], expected[slice_index])


def test_create_mtype_density():
    density_profile_collection = create_density_profile_collection()
    with tempfile.TemporaryDirectory() as tempdir:
        density_profile_collection.create_density(
            "L2_TPC:A",
            "excitatory",
            create_excitatory_neuron_density(),
            create_slice_voxel_indices(),
            tempdir,
        )
        expected_cell_densities = create_expected_cell_densities()
        created_cell_density = VoxelData.load_nrrd(str(Path(tempdir, "L2_TPC:A_density.nrrd"))).raw
        npt.assert_array_equal(created_cell_density, expected_cell_densities["L2_TPC:A"])
        density_profile_collection.create_density(
            "L23_BP",
            "inhibitory",
            create_inhibitory_neuron_density(),
            create_slice_voxel_indices(),
            tempdir,
        )
        created_cell_density = VoxelData.load_nrrd(str(Path(tempdir, "L23_BP_density.nrrd"))).raw
        npt.assert_array_equal(created_cell_density, expected_cell_densities["L23_BP"])


def test_create_mtype_densities():
    density_profile_collection = create_density_profile_collection()
    data = create_slicer_data()
    with tempfile.TemporaryDirectory() as tempdir:
        density_profile_collection.create_mtype_densities(
            data["annotation"],
            data["region_map"],
            data["metadata"],
            data["direction_vectors"],
            tempdir,
            create_excitatory_neuron_density(),
            create_inhibitory_neuron_density(),
        )
        expected_cell_densities = create_expected_cell_densities()
        for mtype, expected_cell_density in expected_cell_densities.items():
            created_cell_density = VoxelData.load_nrrd(
                str(Path(tempdir, f"{mtype}_density.nrrd"))
            ).raw
            npt.assert_array_equal(created_cell_density, expected_cell_density)
