import json
import tempfile
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from voxcell import RegionMap, VoxelData  # type: ignore

import atlas_densities.densities.mtype_densities_from_map as tested
from atlas_densities.exceptions import AtlasDensitiesError


def create_from_probability_map_data():
    raw_probability_map = pd.DataFrame(
        {
            "region": [
                "AUDd2/3",
                "AUDd2/3",
                "AUDd2/3",
                "AUDd2/3",
                "AUDd4",
                "AUDd4",
                "AUDd4",
                "AUDd4",
                "AUDd5",
                "AUDd5",
                "AUDd5",
                "AUDd5",
                "AUDd6a",
                "AUDd6b",
            ],
            "molecular_type": [
                "approx_lamp5",
                "pv",
                "sst",
                "vip",
                "approx_lamp5",
                "pv",
                "sst",
                "vip",
                "approx_lamp5",
                "pv",
                "sst",
                "vip",
                "approx_lamp5",
                "approx_lamp5",
            ],
            "ChC": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.4,
            ],
            "LAC": [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.6,
            ],
        },
    )
    probability_map = raw_probability_map.copy()
    probability_map.set_index(["region", "molecular_type"], inplace=True)

    return {
        "annotation": VoxelData(
            np.array([[[678, 527, 243, 252, 600]]], dtype=int),
            voxel_dimensions=[25] * 3,
        ),
        "hierarchy": json.load(open("tests/1.json", "r")),
        "region_map": RegionMap.load_json("tests/1.json"),
        "molecular_type_densities": {
            "pv": np.array([[[1.0, 0.0, 0.0, 1.0, 1.0]]], dtype=float),
            "sst": np.array([[[0.0, 1.0, 1.0, 0.0, 0.0]]], dtype=float),
            "vip": np.array([[[0.0, 1.0, 0.0, 1.0, 0.0]]], dtype=float),
            "gad67": np.array([[[1.5, 2.0, 1.0, 2.0, 1.0]]], dtype=float),
            "approx_lamp5": np.array([[[0.5, 0.0, 0.0, 0.0, 0.0]]]),
        },
        "raw_probability_map": raw_probability_map,
        "probability_maps": [probability_map],
    }


class Test_create_from_probability_map:
    def setup_method(self, method):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data = create_from_probability_map_data()
        tested.create.create_from_probability_map(
            self.data["annotation"],
            self.data["region_map"],
            self.data["molecular_type_densities"],
            self.data["probability_maps"],
            self.tmpdir.name,
            1,
        )

    def teardown_method(self, method):
        self.tmpdir.cleanup()

    def test_filenames(self):
        tmpdir = self.tmpdir.name
        filepaths = {Path.resolve(f).name for f in Path(tmpdir).glob("*.nrrd")}
        assert filepaths == {"LAC_densities.nrrd", "ChC_densities.nrrd"}

    def test_output_values(self):
        tmpdir = self.tmpdir.name
        expected_densities = {
            "ChC": np.array([[[0.0, 0.0, 0.0, 2.0, 0.0]]], dtype=float),
            "LAC": np.array([[[1.5, 0.0, 0.0, 0.0, 1.0]]], dtype=float),
        }
        for mtype in ["ChC", "LAC"]:
            filepath = str(Path(tmpdir) / f"{mtype}_densities.nrrd")
            npt.assert_array_almost_equal(
                VoxelData.load_nrrd(filepath).raw, expected_densities[mtype]
            )


class Test_create_from_probability_map_exceptions:
    def setup_method(self, method):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data = create_from_probability_map_data()

    def teardown_method(self, method):
        self.tmpdir.cleanup()

    def create_densities(self):
        tested.create.create_from_probability_map(
            self.data["annotation"],
            self.data["region_map"],
            self.data["molecular_type_densities"],
            self.data["probability_maps"],
            self.tmpdir.name,
            1,
        )

    def test_probability_map_sanity_negative_probability(self):
        for probability_map in self.data["probability_maps"]:
            probability_map.at[("AUDd4", "sst"), "ChC"] = -0.0025
        with pytest.raises(AtlasDensitiesError):
            self.create_densities()

    def test_probability_map_sanity_row_sum_is_1(self):
        for probability_map in self.data["probability_maps"]:
            probability_map.at[("AUDd4", "sst"), "ChC"] = 2.0
        with pytest.raises(AtlasDensitiesError):
            self.create_densities()


class Test__merge_probability_maps:
    def create_probability_map(self, data):
        probability_map = pd.DataFrame(data)
        probability_map.set_index(["region", "molecular_type"], inplace=True)
        return probability_map

    def test_region_intersection(self):
        probability_maps = [
            self.create_probability_map(
                {
                    "region": [
                        "regionA",
                        "regionA",
                        "regionB",
                        "regionB",
                        "regionC",
                        "regionC",
                    ],
                    "molecular_type": [
                        "pv",
                        "sst",
                        "vip",
                        "pv",
                        "sst",
                        "vip",
                    ],
                    "mtype01": [
                        0.0,
                        0.0,
                        0.0,
                        0.5,
                        0.5,
                        0.5,
                    ],
                },
            ),
            self.create_probability_map(
                {
                    "region": [
                        "regionB",
                        "regionD",
                        "regionD",
                        "regionD",
                        "regionE",
                        "regionE",
                    ],
                    "molecular_type": [
                        "pv",
                        "sst",
                        "vip",
                        "pv",
                        "sst",
                        "vip",
                    ],
                    "mtype01": [
                        0.0,
                        0.0,
                        0.0,
                        0.5,
                        0.5,
                        0.5,
                    ],
                },
            ),
        ]
        with pytest.raises(ValueError, match="regionB"):
            tested.utils._merge_probability_maps(probability_maps)

    def test_merge(self):
        probability_maps = [
            self.create_probability_map(
                {
                    "region": [
                        "regionA",
                        "regionA",
                        "regionB",
                        "regionC",
                    ],
                    "molecular_type": [
                        "pv",
                        "sst",
                        "vip",
                        "pv",
                    ],
                    "mtype01": [
                        0.0,
                        0.0,
                        0.5,
                        0.5,
                    ],
                    "mtype02": [
                        0.5,
                        0.5,
                        0.0,
                        0.0,
                    ],
                    "mtype03": [
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                    ],
                }
            ),
            self.create_probability_map(
                {
                    "region": [
                        "regionD",
                        "regionD",
                    ],
                    "molecular_type": [
                        "pv",
                        "vip",
                    ],
                    "mtype01": [
                        0.2,
                        0.2,
                    ],
                    "mtype02": [
                        0.2,
                        0.2,
                    ],
                    "mtype04": [
                        0.6,
                        0.6,
                    ],
                }
            ),
            self.create_probability_map(
                {
                    "region": [
                        "regionE",
                        "regionE",
                    ],
                    "molecular_type": [
                        "pv",
                        "vip",
                    ],
                    "mtype03": [
                        0.1,
                        0.1,
                    ],
                    "mtype04": [
                        0.9,
                        0.9,
                    ],
                }
            ),
        ]

        result = tested.utils._merge_probability_maps(probability_maps)

        expected = self.create_probability_map(
            {
                "region": [
                    "regionA",
                    "regionA",
                    "regionB",
                    "regionC",
                    "regionD",
                    "regionD",
                    "regionE",
                    "regionE",
                ],
                "molecular_type": [
                    "pv",
                    "sst",
                    "vip",
                    "pv",
                    "pv",
                    "vip",
                    "pv",
                    "vip",
                ],
                "mtype01": [
                    0.0,
                    0.0,
                    0.5,
                    0.5,
                    0.2,
                    0.2,
                    0.0,
                    0.0,
                ],
                "mtype02": [
                    0.5,
                    0.5,
                    0.0,
                    0.0,
                    0.2,
                    0.2,
                    0.0,
                    0.0,
                ],
                "mtype03": [
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.0,
                    0.0,
                    0.1,
                    0.1,
                ],
                "mtype04": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.6,
                    0.6,
                    0.9,
                    0.9,
                ],
            }
        )
        assert expected.equals(result)
