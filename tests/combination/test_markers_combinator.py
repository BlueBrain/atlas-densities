"""test markers_combinator"""
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
from voxcell import RegionMap  # type: ignore

import atlas_densities.combination.markers_combinator as tested

TEST_PATH = Path(Path(__file__).parent.parent)


def test_combine():
    region_map = RegionMap.load_json(str(Path(TEST_PATH, "1.json")))
    # Cerebellum ids: 512, 1143
    # Striatum ids: 56, 477, 485
    annotation_raw = np.array(
        [
            [[512, 512, 1143]],
            [[512, 512, 1143]],
            [[477, 56, 485]],
        ]
    )
    expected_proportions = [4.0 / 23.0, 8.0 / 23.0, 11.0 / 23.0]

    glia_celltype_densities = pd.DataFrame(
        {
            "cerebellum": {
                "oligodendrocyte": 12000.0,
                "astrocyte": 1500.0,
                "microglia": 6000.0,
            },
            "striatum": {
                "oligodendrocyte": 9000.0,
                "astrocyte": 9000.0,
                "microglia": 12000.0,
            },
        }
    )

    cnp = np.array(
        [
            [[0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0]],
        ]
    )
    mbp = np.array(
        [
            [[0.0, 1.0, 0.0]],
            [[9.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0]],
        ]
    )
    gfap = np.array(
        [
            [[0.0, 1.0, 5.0]],
            [[1.0, 4.0, 5.0]],
            [[0.0, 1.0, 0.0]],
        ]
    )
    s100b = np.array(
        [
            [[0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0]],
        ]
    )
    aldh1l1 = np.array(
        [
            [[1.0, 1.0, 0.0]],
            [[2.0, 1.0, 4.0]],
            [[0.0, 0.0, 0.0]],
        ]
    )
    tmem119 = np.array(
        [
            [[0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0]],
        ]
    )

    combination_data = pd.DataFrame(
        [
            ["oligodendrocyte", "cnp", 1.0, cnp],
            ["oligodendrocyte", "mbp", 1.0 / 2.0, mbp],
            ["astrocyte", "gfap", 1.0 / 3.0, gfap],
            ["astrocyte", "s100b", 1.0, s100b],
            ["astrocyte", "aldh1l1", 1.0 / 2.0, aldh1l1],
            ["microglia", "tmem119", 1.0 / 3.0, tmem119],
        ],
        columns=["cellType", "gene", "averageExpressionIntensity", "volume"],
    )

    expected_glia_intensities = [
        np.array(
            [
                [[2.0, 6.0, 15.0]],
                [[7.0, 15.0, 23.0]],
                [[0.0, 4.0, 0.0]],
            ]
        )
        / 6.0,  # astrocyte
        np.array(
            [
                [[0.0, 3.0, 0.0]],
                [[0.0, 0.0, 3.0]],
                [[3.0, 0.0, 0.0]],
            ]
        )
        / 3.0,  # microglia
        np.array(
            [
                [[0.0, 2.0, 0.0]],
                [[18.0, 3.0, 0.0]],
                [[0.0, 3.0, 0.0]],
            ]
        )
        / 3.0,  # oligodendrocyte
    ]
    expected_glia_intensities.insert(
        1,
        np.average(expected_glia_intensities, axis=0, weights=expected_proportions),
    )

    intensities = tested.combine(
        region_map, annotation_raw, glia_celltype_densities, combination_data
    ).sort_values(by="cellType")

    npt.assert_array_almost_equal(list(intensities.intensity), expected_glia_intensities, decimal=5)
