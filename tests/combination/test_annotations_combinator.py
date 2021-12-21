"""test annotations_combinator"""
import numpy as np
import numpy.testing as npt
import voxcell

import atlas_densities.combination.annotations_combinator as tested


def test_is_ancestor():
    hierachy = {
        "id": 1,
        "acronym": "root",
        "children": [
            {
                "id": 2,
                "acronym": "below",
                "children": [
                    {"id": 3, "acronym": "below_3"},
                    {"id": 4, "acronym": "below_4"},
                ],
            }
        ],
    }
    ccfv2_raw = np.array([[[1, 2], [2, 1]], [[2, 3], [2, 4]], [[2, 0], [0, 4]]])
    ccfv3_raw = np.array([[[1, 1], [4, 3]], [[2, 4], [2, 2]], [[0, 3], [0, 0]]])
    expected = np.array(
        [
            [[True, True], [False, False]],
            [[True, False], [True, True]],
            [[False, False], [False, False]],
        ]
    )
    region_map = voxcell.RegionMap.from_dict(hierachy)
    npt.assert_array_equal(tested.is_ancestor(region_map, ccfv3_raw, ccfv2_raw), expected)


def test_combine_annotations_fiber_tracts_are_merged():
    hierachy = {
        "id": 1,
        "acronym": "root",
        "children": [
            {
                "id": 2,
                "acronym": "below",
                "children": [
                    {"id": 3, "acronym": "below_3"},
                    {"id": 4, "acronym": "below_4"},
                ],
            }
        ],
    }
    region_map = voxcell.RegionMap.from_dict(hierachy)
    shape = (2, 2, 2)
    voxel_dimensions = (10.0, 10.0, 10.0)

    brain_annotation_ccfv2 = np.zeros(shape)
    brain_annotation_ccfv2[0, :, :] = 3
    brain_annotation_ccfv2 = voxcell.VoxelData(brain_annotation_ccfv2, voxel_dimensions)

    fiber_annotation_ccfv2 = np.zeros(shape)
    fiber_annotation_ccfv2[1, :, :] = 4
    fiber_annotation_ccfv2 = voxcell.VoxelData(fiber_annotation_ccfv2, voxel_dimensions)

    brain_annotation_ccfv3 = np.zeros(shape)
    brain_annotation_ccfv3[:, :, :] = 1
    brain_annotation_ccfv3 = voxcell.VoxelData(brain_annotation_ccfv3, voxel_dimensions)

    expected_raw = np.zeros(shape)
    expected_raw[0, :, :] = 3
    expected_raw[1, :, :] = 4

    ret = tested.combine_annotations(
        region_map,
        brain_annotation_ccfv2,
        fiber_annotation_ccfv2,
        brain_annotation_ccfv3,
    )

    npt.assert_array_equal(ret.raw, expected_raw)


def test_combine_annotations_non_leaf_ids_are_replaced():
    hierachy = {
        "id": 1,
        "acronym": "root",
        "children": [
            {
                "id": 2,
                "acronym": "grey",
                "children": [
                    {"id": 3, "acronym": "grey_3"},
                    {"id": 4, "acronym": "grey_4"},
                ],
            }
        ],
    }
    region_map = voxcell.RegionMap.from_dict(hierachy)
    shape = (2, 2, 2)
    voxel_dimensions = (10.0, 10.0, 10.0)

    brain_annotation_ccfv2 = np.zeros(shape)
    brain_annotation_ccfv2[0, :, :] = 3
    brain_annotation_ccfv2[1, :, :] = 4
    brain_annotation_ccfv2 = voxcell.VoxelData(brain_annotation_ccfv2, voxel_dimensions)

    fiber_annotation_ccfv2 = np.copy(brain_annotation_ccfv2.raw)
    fiber_annotation_ccfv2 = voxcell.VoxelData(fiber_annotation_ccfv2, voxel_dimensions)

    brain_annotation_ccfv3 = np.zeros(shape)
    brain_annotation_ccfv3[0, :, :] = 3
    brain_annotation_ccfv3[1, :, :] = 1
    brain_annotation_ccfv3 = voxcell.VoxelData(brain_annotation_ccfv3, voxel_dimensions)

    expected_raw = np.zeros(shape)
    expected_raw[0, :, :] = 3
    expected_raw[1, :, :] = 4

    ret = tested.combine_annotations(
        region_map,
        brain_annotation_ccfv2,
        fiber_annotation_ccfv2,
        brain_annotation_ccfv3,
    )
    npt.assert_array_equal(ret.raw, expected_raw)


def test_combine_annotations_zeros_are_ignored():
    hierachy = {"id": 1, "acronym": "root", "children": []}
    region_map = voxcell.RegionMap.from_dict(hierachy)
    shape = (2, 2, 2)
    voxel_dimensions = (10.0, 10.0, 10.0)

    brain_annotation_ccfv2 = np.zeros(shape)
    brain_annotation_ccfv2[:, :, :] = 1
    brain_annotation_ccfv2 = voxcell.VoxelData(brain_annotation_ccfv2, voxel_dimensions)

    fiber_annotation_ccfv2 = np.copy(brain_annotation_ccfv2.raw)
    fiber_annotation_ccfv2 = voxcell.VoxelData(fiber_annotation_ccfv2, voxel_dimensions)

    brain_annotation_ccfv3 = np.zeros(shape)
    brain_annotation_ccfv3[:, :, :] = 1
    brain_annotation_ccfv3[0, 0, 0] = 0
    brain_annotation_ccfv3 = voxcell.VoxelData(brain_annotation_ccfv3, voxel_dimensions)

    expected_raw = np.copy(brain_annotation_ccfv3.raw)

    ret = tested.combine_annotations(
        region_map,
        brain_annotation_ccfv2,
        fiber_annotation_ccfv2,
        brain_annotation_ccfv3,
    )

    npt.assert_array_equal(ret.raw, expected_raw)


def test_ccfv2_overrides_ancestors():
    hierachy = {
        "id": 1,
        "acronym": "root",
        "children": [
            {
                "id": 2,
                "acronym": "left",
                "children": [
                    {"id": 3, "acronym": "below_3"},
                    {"id": 4, "acronym": "below_4"},
                    {"id": 8, "acronym": "below_8"},
                ],
            },
            {
                "id": 5,
                "acronym": "right",
                "children": [
                    {"id": 6, "acronym": "below_6"},
                    {"id": 7, "acronym": "below_7"},
                ],
            },
        ],
    }
    region_map = voxcell.RegionMap.from_dict(hierachy)
    ccfv2_raw = np.array([[[1, 2], [2, 7]], [[6, 3], [4, 1]], [[2, 0], [8, 4]]])
    ccfv3_raw = np.array([[[1, 7], [4, 5]], [[5, 4], [2, 2]], [[0, 3], [5, 0]]])
    expected_raw = np.array([[[1, 7], [4, 5]], [[6, 4], [2, 2]], [[0, 3], [5, 0]]])

    voxel_dimensions = (10.0, 10.0, 10.0)
    brain_annotation_ccfv2 = voxcell.VoxelData(ccfv2_raw, voxel_dimensions)
    fiber_annotation_ccfv2 = voxcell.VoxelData(np.copy(ccfv2_raw), voxel_dimensions)
    brain_annotation_ccfv3 = voxcell.VoxelData(ccfv3_raw, voxel_dimensions)

    ret = tested.combine_annotations(
        region_map,
        brain_annotation_ccfv2,
        fiber_annotation_ccfv2,
        brain_annotation_ccfv3,
    )
    npt.assert_array_equal(ret.raw, expected_raw)


def test_combine_annotations():
    hierachy = {
        "id": 1,
        "acronym": "root",
        "children": [
            {
                "id": 2,
                "acronym": "left",
                "children": [
                    {"id": 3, "acronym": "left_3"},
                    {"id": 4, "acronym": "left_4"},
                ],
            },
            {
                "id": 5,
                "acronym": "fibers",
                "children": [
                    {
                        "id": 6,
                        "acronym": "fiber_6",
                        "children": [
                            {"id": 8, "acronym": "fiber_8"},
                            {"id": 9, "acronym": "fiber_9"},
                        ],
                    },
                    {
                        "id": 7,
                        "acronym": "fiber_7",
                        "children": [
                            {"id": 10, "acronym": "fiber_10"},
                            {"id": 11, "acronym": "fiber_11"},
                        ],
                    },
                ],
            },
            {"id": 12, "acronym": "right"},
        ],
    }
    region_map = voxcell.RegionMap.from_dict(hierachy)

    shape = (3, 3, 3)
    voxel_dimensions = (10.0, 10.0, 10.0)

    brain_annotation_ccfv2 = np.zeros(shape)
    brain_annotation_ccfv2[0:2, 0, :] = 3
    brain_annotation_ccfv2[2, 0, :] = 4
    brain_annotation_ccfv2[:, 2, :] = 12
    brain_annotation_ccfv2 = voxcell.VoxelData(brain_annotation_ccfv2, voxel_dimensions)

    fiber_annotation_ccfv2 = np.zeros(shape)
    fiber_annotation_ccfv2[:, 1, 0] = 8
    fiber_annotation_ccfv2[:, 1, 1] = 9
    fiber_annotation_ccfv2[0:2, 1, 2] = 10
    fiber_annotation_ccfv2[2, 1, 2] = 11
    fiber_annotation_ccfv2 = voxcell.VoxelData(fiber_annotation_ccfv2, voxel_dimensions)

    brain_annotation_ccfv3 = np.zeros(shape)
    brain_annotation_ccfv3[:, 0, :] = 2
    brain_annotation_ccfv3[:, 2, :] = 12
    brain_annotation_ccfv3[:, 1, 0] = 6
    brain_annotation_ccfv3[:, 1, 1] = 5
    brain_annotation_ccfv3 = voxcell.VoxelData(brain_annotation_ccfv3, voxel_dimensions)

    expected_raw = np.copy(brain_annotation_ccfv2.raw)
    expected_raw[:, 1, 0] = 8
    expected_raw[:, 1, 1] = 9
    expected_raw[:, 1, 2] = 0

    ret = tested.combine_annotations(
        region_map,
        brain_annotation_ccfv2,
        fiber_annotation_ccfv2,
        brain_annotation_ccfv3,
    )

    npt.assert_array_equal(ret.raw, expected_raw)
