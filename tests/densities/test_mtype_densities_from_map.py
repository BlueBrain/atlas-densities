import json
import tempfile
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from voxcell import RegionMap, VoxelData  # type: ignore

import atlas_densities.densities.mtype_densities_from_map.create as tested_create
import atlas_densities.densities.mtype_densities_from_map.utils as tested_utils
from atlas_densities.exceptions import AtlasDensitiesError


def create_from_probability_map_data():
    raw_probability_map01 = pd.DataFrame(
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
            ],
            "synapse_class": [
                "EXC",
                "EXC",
                "EXC",
                "EXC",
                "EXC",
                "EXC",
                "EXC",
                "EXC",
            ],
            "BP|bAC": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            "BP|bIR": [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
        },
    )
    raw_probability_map02 = pd.DataFrame(
        {
            "region": [
                "AUDd4",
                "AUDd5",
                "AUDd5",
                "AUDd5",
                "AUDd6a",
                "AUDd6b",
            ],
            "molecular_type": [
                "vip",
                "pv",
                "sst",
                "vip",
                "approx_lamp5",
                "approx_lamp5",
            ],
            "synapse_class": [
                "INH",
                "EXC",
                "INH",
                "EXC",
                "INH",
                "EXC",
            ],
            "BP|bAC": [
                0.2,
                1.0,
                1.0,
                1.0,
                1.0,
                0.4,
            ],
            "BP|bIR": [
                0.8,
                0.0,
                0.0,
                0.0,
                0.0,
                0.6,
            ],
        },
    )
    probability_map01 = raw_probability_map01.copy()
    probability_map02 = raw_probability_map02.copy()
    probability_map01.set_index(["region", "molecular_type", "synapse_class"], inplace=True)
    probability_map02.set_index(["region", "molecular_type", "synapse_class"], inplace=True)

    return {
        "annotation": VoxelData(
            np.array([[[678, 527, 243, 252, 600]]], dtype=int), voxel_dimensions=[25] * 3
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
        "raw_probability_map01": raw_probability_map01,
        "raw_probability_map02": raw_probability_map02,
        "probability_map01": probability_map01,
        "probability_map02": probability_map02,
    }


@pytest.mark.parametrize(
    "probability_maps_keys, synapse_class, metypes",
    [
        (
            ["probability_map01", "probability_map02"],
            "EXC",
            ["BP|bAC", "BP|bIR"],
        ),
        (
            ["probability_map01", "probability_map02"],
            "INH",
            [],
        ),
        (
            ["probability_map01"],
            "EXC",
            ["BP|bIR"],
        ),
    ],
)
class Test_create_from_probability_map:
    def setup_method(self, method):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data = create_from_probability_map_data()

    def call_create(self, probability_maps_keys, synapse_class):
        tested_create.create_from_probability_map(
            self.data["annotation"],
            self.data["region_map"],
            self.data["molecular_type_densities"],
            [self.data[key] for key in probability_maps_keys],
            synapse_class,
            self.tmpdir.name,
            1,
        )

    def teardown_method(self, method):
        self.tmpdir.cleanup()

    def test_filenames(self, probability_maps_keys, synapse_class, metypes):
        self.call_create(probability_maps_keys, synapse_class)
        tmpdir = self.tmpdir.name
        filepaths = {Path.resolve(f).name for f in Path(tmpdir).glob("*.nrrd")}
        assert filepaths == {f"{metype}_{synapse_class}_densities.nrrd" for metype in metypes}

    def test_output_values(self, probability_maps_keys, synapse_class, metypes):
        self.call_create(probability_maps_keys, synapse_class)
        tmpdir = self.tmpdir.name
        expected_densities = {
            "BP|bAC": np.array([[[0.0, 0.0, 0.0, 2.0, 0.0]]], dtype=float),
            "BP|bIR": np.array([[[1.5, 0.0, 0.0, 0.0, 1.0]]], dtype=float),
        }
        for metype in metypes:
            filepath = str(Path(tmpdir) / f"{metype}_{synapse_class}_densities.nrrd")
            npt.assert_array_almost_equal(
                VoxelData.load_nrrd(filepath).raw, expected_densities[metype]
            )
        # metadata
        with open(str(Path(tmpdir) / "metadata.json"), "r") as file:
            metadata = json.load(file)
        for metype in metypes:
            mtype, etype = metype.split(tested_create.SEPARATOR)
            assert mtype in metadata["density_files"]
            assert etype in metadata["density_files"][mtype]
        assert synapse_class == metadata["synapse_class"]


class Test_create_from_probability_map_empty:
    def test_empty_exception(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data = create_from_probability_map_data()
        with pytest.raises(AtlasDensitiesError):
            tested_create.create_from_probability_map(
                self.data["annotation"],
                self.data["region_map"],
                self.data["molecular_type_densities"],
                [self.data["probability_map01"]],
                "INH",
                self.tmpdir.name,
                1,
            )


class Test_create_from_probability_map_exceptions:
    def setup_method(self, method):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data = create_from_probability_map_data()

    def teardown_method(self, method):
        self.tmpdir.cleanup()

    def create_densities(self):
        tested_create.create_from_probability_map(
            self.data["annotation"],
            self.data["region_map"],
            self.data["molecular_type_densities"],
            [self.data["probability_map01"], self.data["probability_map02"]],
            "all",
            self.tmpdir.name,
            1,
        )

    def test_probability_map_sanity_negative_probability(self):
        self.data["probability_map01"].at[("AUDd4", "sst"), "BP|bAC"] = -0.0025
        self.data["probability_map02"].at[("AUDd5", "sst"), "BP|bAC"] = -0.0025
        with pytest.raises(AtlasDensitiesError):
            self.create_densities()

    def test_probability_map_sanity_row_sum_is_1(self):
        self.data["probability_map01"].at[("AUDd4", "sst"), "BP|bAC"] = 2.0
        self.data["probability_map02"].at[("AUDd5", "sst"), "BP|bAC"] = 2.0
        with pytest.raises(AtlasDensitiesError):
            self.create_densities()


class Test__merge_probability_maps:
    def create_probability_map(self, data):
        probability_map = pd.DataFrame(data)
        probability_map.set_index(["region", "molecular_type"], inplace=True)
        return probability_map

    def test_index_intersection_success(self):

        probability_maps = [
            self.create_probability_map(
                {
                    "region": [
                        "regionA",
                        "regionA",
                        "regionB",
                    ],
                    "molecular_type": ["pv", "sst", "vip"],
                    "mtype01": [0.0, 0.0, 0.0],
                },
            ),
            self.create_probability_map(
                {
                    "region": [
                        "regionB",
                        "regionD",
                        "regionD",
                    ],
                    "molecular_type": ["pv", "sst", "vip"],
                    "mtype01": [0.0, 0.5, 0.5],
                },
            ),
        ]
        tested_utils._merge_probability_maps(probability_maps)

    def test_index_intersection_fail(self):
        probability_maps = [
            self.create_probability_map(
                {
                    "region": [
                        "regionA",
                        "regionA",
                        "regionB",  # regionB, pv is in both maps
                    ],
                    "molecular_type": ["pv", "sst", "pv"],
                    "mtype01": [0.0, 0.0, 0.5],
                },
            ),
            self.create_probability_map(
                {
                    "region": [
                        "regionB",  # regionB, pv is in both maps
                        "regionD",
                        "regionD",
                    ],
                    "molecular_type": ["pv", "sst", "vip"],
                    "mtype01": [0.0, 0.5, 0.5],
                },
            ),
        ]
        with pytest.raises(ValueError):
            tested_utils._merge_probability_maps(probability_maps)

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

        result = tested_utils._merge_probability_maps(probability_maps)

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
