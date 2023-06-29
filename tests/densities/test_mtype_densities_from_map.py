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
    metadata = {
        "regions": [
            {
                "name": "Dorsal auditory area",
                "query": "AUDd",
                "attribute": "acronym",
            },
        ],
    }

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
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
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
                0.8,
                0.8,
                0.8,
                0.8,
                0.8,
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
            np.array([[[678, 527, 243, 252, 600]]], dtype=int), voxel_dimensions=[25] * 3
        ),
        "hierarchy": json.load(open("tests/1.json", "r")),
        "region_map": RegionMap.load_json("tests/1.json"),
        "metadata": metadata,
        "molecular_type_densities": {
            "pv": np.array([[[1.0, 0.0, 0.0, 1.0, 1.0]]], dtype=float),
            "sst": np.array([[[0.0, 1.0, 1.0, 0.0, 0.0]]], dtype=float),
            "vip": np.array([[[0.0, 1.0, 0.0, 1.0, 0.0]]], dtype=float),
            "gad67": np.array([[[1.5, 2.0, 1.0, 2.0, 1.0]]], dtype=float),
            "approx_lamp5": np.array([[[0.5, 0.0, 0.0, 0.0, 0.0]]]),
        },
        "raw_probability_map": raw_probability_map,
        "probability_map": probability_map,
    }


class Test_create_from_probability_map:
    def setup_method(self, method):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data = create_from_probability_map_data()
        tested.create.create_from_probability_map(
            self.data["annotation"],
            self.data["region_map"],
            self.data["metadata"],
            self.data["molecular_type_densities"],
            self.data["probability_map"],
            self.tmpdir.name,
        )

    def teardown_method(self, method):
        self.tmpdir.cleanup()

    def test_filenames(self):
        tmpdir = self.tmpdir.name
        no_regions_filepaths = {
            Path.resolve(f).name for f in Path(tmpdir, "no_regions").glob("*.nrrd")
        }
        assert no_regions_filepaths == {"LAC_densities.nrrd", "ChC_densities.nrrd"}
        with_regions_filepaths = {
            Path.resolve(f).name for f in Path(tmpdir, "with_regions").glob("*.nrrd")
        }
        assert with_regions_filepaths == {
            "AUDd4_ChC_densities.nrrd",
            "AUDd5_ChC_densities.nrrd",
            "AUDd23_LAC_densities.nrrd",
            "AUDd23_ChC_densities.nrrd",
            "AUDd4_LAC_densities.nrrd",
        }

    def test_output_consistency(self):
        sum_ = {
            "ChC": np.zeros(self.data["annotation"].shape, dtype=float),
            "LAC": np.zeros(self.data["annotation"].shape, dtype=float),
        }
        tmpdir = self.tmpdir.name
        for filename in [
            "AUDd4_ChC_densities.nrrd",
            "AUDd5_ChC_densities.nrrd",
            "AUDd23_ChC_densities.nrrd",
        ]:
            filepath = str(Path(tmpdir) / "with_regions" / filename)
            sum_["ChC"] += VoxelData.load_nrrd(filepath).raw

        for filename in [
            "AUDd23_LAC_densities.nrrd",
            "AUDd4_LAC_densities.nrrd",
        ]:
            filepath = str(Path(tmpdir) / "with_regions" / filename)
            sum_["LAC"] += VoxelData.load_nrrd(filepath).raw

        for mtype in ["ChC", "LAC"]:
            filepath = str(Path(tmpdir) / "no_regions" / f"{mtype}_densities.nrrd")
            density = VoxelData.load_nrrd(filepath)
            npt.assert_array_equal(
                density.voxel_dimensions, self.data["annotation"].voxel_dimensions
            )
            assert density.raw.dtype == float
            npt.assert_array_almost_equal(sum_[mtype], density.raw)

    def test_output_values(self):
        tmpdir = self.tmpdir.name
        expected_densities = {
            "ChC": np.array([[[0.1, 0.0, 0.0, 2.0, 0.2]]], dtype=float),
            "LAC": np.array([[[1.4, 0.0, 0.0, 0.0, 0.8]]], dtype=float),
        }
        for mtype in ["ChC", "LAC"]:
            filepath = str(Path(tmpdir) / "no_regions" / f"{mtype}_densities.nrrd")
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
            self.data["metadata"],
            self.data["molecular_type_densities"],
            self.data["probability_map"],
            self.tmpdir.name,
        )

    def test_probability_map_sanity_negative_probability(self):
        self.data["probability_map"].at[("AUDd4", "sst"), "ChC"] = -0.0025
        with pytest.raises(AtlasDensitiesError):
            self.create_densities()

    def test_probability_map_sanity_row_sum_is_1(self):
        self.data["probability_map"].at[("AUDd4", "sst"), "ChC"] = 2.0
        with pytest.raises(AtlasDensitiesError):
            self.create_densities()
