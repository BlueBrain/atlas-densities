"""test utils"""
from pathlib import Path

TEST_PATH = Path(Path(__file__).parent)

import numpy as np
import numpy.testing as npt
import pytest
from voxcell import RegionMap

import atlas_densities.utils as tested
from atlas_densities.exceptions import AtlasDensitiesError


class Test_load_region_map:
    def test_wrong_input(self):
        with pytest.raises(TypeError):
            tested.load_region_map()

        with pytest.raises(TypeError):
            tested.load_region_map([])

    def test_region_map_input(self):
        region_map = RegionMap.load_json(str(Path(TEST_PATH, "1.json")))
        assert isinstance(tested.load_region_map(region_map), RegionMap)

    def test_str_input(self):
        assert isinstance(tested.load_region_map(str(Path(TEST_PATH, "1.json"))), RegionMap)

    def test_dict_input(self):
        hierachy = {
            "id": 1,
            "acronym": "root",
            "children": [{"id": 2, "children": [{"id": 3}, {"id": 4}]}],
        }
        assert isinstance(tested.load_region_map(hierachy), RegionMap)


def test_query_region_mask():
    annotation = np.arange(27).reshape((3, 3, 3))
    expected = np.zeros((3, 3, 3), dtype=bool)
    expected[0, 0, 2] = True  # label 2, "SSp-m6b"
    expected[1, 0, 0] = True  # label 9, "SSp-tr6a"
    expected[2, 1, 1] = True  # label 22, "SSp-tr6a"

    # RegionMap instantiated from string
    region = {
        "query": "Isocortex",
        "attribute": "acronym",
        "with_descendants": True,
    }
    mask = tested.query_region_mask(region, annotation, str(Path(TEST_PATH, "1.json")))
    npt.assert_array_equal(mask, expected)

    # True RegionMap object
    region_map = RegionMap.load_json(str(Path(TEST_PATH, "1.json")))
    mask = tested.query_region_mask(region, annotation, region_map)
    npt.assert_array_equal(mask, expected)


def test_get_region_mask():
    annotation = np.arange(27).reshape((3, 3, 3))
    expected = np.zeros((3, 3, 3), dtype=bool)
    expected[0, 0, 2] = True  # label 2, "SSp-m6b"
    expected[1, 0, 0] = True  # label 9, "SSp-tr6a"
    expected[2, 1, 1] = True  # label 22, "SSp-tr6a"

    # RegionMap instantiated from string
    mask = tested.get_region_mask("Isocortex", annotation, str(Path(TEST_PATH, "1.json")))
    npt.assert_array_equal(mask, expected)

    # True RegionMap object
    region_map = RegionMap.load_json(str(Path(TEST_PATH, "1.json")))
    mask = tested.get_region_mask("Isocortex", annotation, region_map)
    npt.assert_array_equal(mask, expected)


def test_split_into_halves():
    volume = np.array(
        [
            [[0, 1, 2], [2, 3, 4]],
            [[4, 5, 6], [7, 8, 9]],
        ],
        dtype=np.int64,
    )
    halves = tested.split_into_halves(volume)
    npt.assert_array_equal(
        halves[0],
        np.array(
            [
                [[0, 0, 0], [2, 0, 0]],
                [[4, 0, 0], [7, 0, 0]],
            ],
            dtype=np.int64,
        ),
    )
    npt.assert_array_equal(
        halves[1],
        np.array(
            [
                [[0, 1, 2], [0, 3, 4]],
                [[0, 5, 6], [0, 8, 9]],
            ]
        ),
    )


def test_is_obtuse_angle():
    vector_field_1 = np.array(
        [
            [
                [[1.0, 0.0, -1.0], [10.3, 5.6, 9.0]],
                [[0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]],
            ],
            [
                [[1.0, 2.0, -1.0], [-5.1, 1.0, 1.0]],
                [[1.0, 2.0, 3.0], [0.0, -1.0, -1.0]],
            ],
        ]
    )
    vector_field_2 = np.array(
        [
            [
                [[-1.0, 1.0, -1.0], [2.0, -2.0, 1.0]],
                [[-0.3, 0.1, -1.9], [1.0, -1.0, -1.0]],
            ],
            [
                [[-1.0, 2.0, -1.0], [5.1, 0.0, 26.0]],
                [[3.0, 2.0, -3.0], [-6.0, -1.0, -1.0]],
            ],
        ]
    )

    expected = [
        [[False, False], [False, False]],
        [[False, True], [True, False]],
    ]
    npt.assert_array_equal(tested.is_obtuse_angle(vector_field_1, vector_field_2), expected)


def test_copy_array():
    array = np.array([1, 2])
    copied_array = tested.copy_array(array)
    array[0] = 0
    npt.assert_array_equal(copied_array, [1, 2])

    copied_array = tested.copy_array(array, copy=True)
    array[0] = 1
    npt.assert_array_equal(copied_array, [0, 2])

    copied_array = tested.copy_array(array, copy=False)
    array[0] = 0
    npt.assert_array_equal(copied_array, [0, 2])


def test_compute_boundary():
    v_1 = np.zeros((5, 5, 5), dtype=bool)
    v_1[1:4, 1:4, 1:4] = True
    v_2 = ~v_1
    boundary = tested.compute_boundary(v_1, v_2)
    expected = np.copy(v_1)
    expected[2, 2, 2] = False
    npt.assert_array_equal(boundary, expected)

    v_1 = np.zeros((5, 5, 5), dtype=bool)
    v_1[0:2, :, 1:4] = True
    v_2 = np.zeros_like(v_1)
    v_2[2:, :, 1:4] = True
    boundary = tested.compute_boundary(v_1, v_2)
    expected = np.zeros_like(v_1)
    expected[1, :, 1:4] = True
    npt.assert_array_equal(boundary, expected)


def get_hierarchy_excerpt():
    return {
        "id": 315,
        "acronym": "Isocortex",
        "name": "Isocortex",
        "children": [
            {
                "id": 500,
                "acronym": "MO",
                "name": "Somatomotor areas",
                "children": [
                    {
                        "id": 107,
                        "acronym": "MO1",
                        "name": "Somatomotor areas, Layer 1",
                        "children": [],
                    },
                    {
                        "id": 219,
                        "acronym": "MO2/3",
                        "name": "Somatomotor areas, Layer 2/3",
                        "children": [],
                    },
                    {
                        "id": 299,
                        "acronym": "MO5",
                        "name": "Somatomotor areas, layer 5",
                        "children": [],
                    },
                ],
            },
            {
                "id": 453,
                "acronym": "SS",
                "name": "Somatosensory areas",
                "children": [
                    {"id": 12993, "acronym": "SS1", "name": "Somatosensory areas, layer 1"}
                ],
            },
        ],
    }


def get_metadata(region_fullname="Isocortex"):
    return {
        "region": {
            "name": region_fullname,
            "query": region_fullname,
            "attribute": "name",
            "with_descendants": True,
        },
        "layers": {
            "names": ["layer_1", "layer_23", "layer_5"],
            "queries": ["@.*1$", "@.*2/3$", "@.*5$"],
            "attribute": "acronym",
            "with_descendants": True,
        },
    }


@pytest.fixture
def region_map():
    return RegionMap.from_dict(get_hierarchy_excerpt())


@pytest.fixture
def annotated_volume():
    return np.array([[[107, 107, 107, 12993, 219, 219, 219, 299, 299, 299]]], dtype=np.uint32)


def test_create_layered_volume(region_map, annotated_volume):
    metadata = get_metadata("Isocortex")
    expected_layers_volume = np.array([[[1, 1, 1, 1, 2, 2, 2, 3, 3, 3]]], dtype=np.uint8)
    actual = tested.create_layered_volume(annotated_volume, region_map, metadata)
    npt.assert_array_equal(expected_layers_volume, actual)

    metadata = get_metadata("Somatomotor areas")
    expected_layers_volume = np.array([[[1, 1, 1, 0, 2, 2, 2, 3, 3, 3]]], dtype=np.uint8)
    actual = tested.create_layered_volume(annotated_volume, region_map, metadata)
    npt.assert_array_equal(expected_layers_volume, actual)


def test_get_layer_masks(region_map, annotated_volume):
    metadata = get_metadata("Isocortex")
    expected_layer_masks = {
        "layer_1": np.array([[[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]], dtype=bool),
        "layer_23": np.array([[[0, 0, 0, 0, 1, 1, 1, 0, 0, 0]]], dtype=bool),
        "layer_5": np.array([[[0, 0, 0, 0, 0, 0, 0, 1, 1, 1]]], dtype=bool),
    }
    actual = tested.get_layer_masks(annotated_volume, region_map, metadata)
    for layer_name in expected_layer_masks:
        npt.assert_array_equal(expected_layer_masks[layer_name], actual[layer_name])


def test_assert_metadata_content(region_map, annotated_volume):
    with pytest.raises(AtlasDensitiesError):
        metadata = get_metadata("Isocortex")
        del metadata["layers"]
        tested.assert_metadata_content(metadata)

    with pytest.raises(AtlasDensitiesError):
        metadata = get_metadata("Isocortex")
        del metadata["region"]
        tested.assert_metadata_content(metadata)

    with pytest.raises(AtlasDensitiesError):
        metadata = get_metadata("Isocortex")
        del metadata["layers"]["attribute"]
        tested.assert_metadata_content(metadata)

    with pytest.raises(AtlasDensitiesError):
        metadata = get_metadata("Isocortex")
        del metadata["layers"]["names"]
        tested.assert_metadata_content(metadata)

    with pytest.raises(AtlasDensitiesError):
        metadata = get_metadata("Isocortex")
        del metadata["layers"]["queries"]
        tested.assert_metadata_content(metadata)
