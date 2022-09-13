import tempfile
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
from pandas.core.frame import DataFrame
from voxcell import RegionMap, VoxelData  # type: ignore

import atlas_densities.densities.mtype_densities_from_map as tested
from atlas_densities.exceptions import AtlasDensitiesError


def create_from_probability_map_data():
    metadata = {
        "region": {
            "name": "Isocortex",
            "query": "Isocortex",
            "attribute": "acronym",
            "with_descendants": True,
        },
        "layers": {
            "names": ["layer_1", "layer_23", "layer_4", "layer_5", "layer_6"],
            "queries": ["@.*1[ab]?$", "@.*2/3$", "@.*4[ab]?$", "@.*5[ab]?$", "@.*6[ab]?$"],
            "attribute": "acronym",
            "with_descendants": True,
        },
    }

    hierarchy = {
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
                        "id": 101,
                        "acronym": "MO1",
                        "name": "Somatomotor areas, Layer 1",
                        "children": [],
                    },
                    {
                        "id": 203,
                        "acronym": "MO2/3",
                        "name": "Somatomotor areas, Layer 2/3",
                        "children": [],
                    },
                    {
                        "id": 104,
                        "acronym": "MO4",
                        "name": "Somatomotor areas, Layer 4",
                        "children": [],
                    },
                    {
                        "id": 105,
                        "acronym": "MO5",
                        "name": "Somatomotor areas, layer 5",
                        "children": [],
                    },
                    {
                        "id": 106,
                        "acronym": "MO6",
                        "name": "Somatomotor areas, layer 6",
                        "children": [],
                    },
                ],
            },
        ],
    }

    raw_probability_map = DataFrame(
        {
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
                1.0,
                1.0,
                1.0,
            ],
            "DLAC": [
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
                0.0,
                0.0,
                0.0,
            ],
        },
        index=[
            "L1_Gad2",
            "L23_Pvalb",
            "L23_Sst",
            "L23_Vip",
            "L23_Lamp5",
            "L4_Pvalb",
            "L4_Sst",
            "L4_Vip",
            "L4_Lamp5",
            "L5_Pvalb",
            "L5_Sst",
            "L5_Vip",
            "L5_Lamp5",
            "L6_Pvalb",
            "L6_Sst",
            "L6_Vip",
            "L6_Lamp5",
        ],
    )

    probability_map = DataFrame(
        {
            "chc": [
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
                1.0,
                1.0,
                1.0,
            ],
            "lac": [
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
                0.0,
                0.0,
                0.0,
            ],
        },
        index=[
            "layer_1_gad67",
            "layer_23_pv",
            "layer_23_sst",
            "layer_23_vip",
            "layer_23_lamp5",
            "layer_4_pv",
            "layer_4_sst",
            "layer_4_vip",
            "layer_4_lamp5",
            "layer_5_pv",
            "layer_5_sst",
            "layer_5_vip",
            "layer_5_lamp5",
            "layer_6_pv",
            "layer_6_sst",
            "layer_6_vip",
            "layer_6_lamp5",
        ],
    )

    return {
        "annotation": VoxelData(
            np.array([[[101, 203, 104, 105, 106]]], dtype=int), voxel_dimensions=[25] * 3
        ),
        "hierarchy": hierarchy,
        "region_map": RegionMap.from_dict(hierarchy),
        "metadata": metadata,
        "molecular_type_densities": {
            "pv": np.array([[[1.0, 0.0, 0.0, 1.0, 1.0]]], dtype=float),
            "sst": np.array([[[0.0, 1.0, 1.0, 0.0, 0.0]]], dtype=float),
            "vip": np.array([[[0.0, 1.0, 0.0, 1.0, 0.0]]], dtype=float),
            "gad67": np.array([[[1.5, 2.0, 1.0, 2.0, 1.0]]], dtype=float),
        },
        "probability_map": probability_map,
        "raw_probability_map": raw_probability_map,
    }


class Test_create_from_probability_map:
    def setup_method(self, method):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data = create_from_probability_map_data()
        tested.create_from_probability_map(
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
        no_layers_filepaths = {
            Path.resolve(f).name for f in Path(tmpdir, "no_layers").glob("*.nrrd")
        }
        assert no_layers_filepaths == {"CHC_densities.nrrd", "LAC_densities.nrrd"}
        with_layers_filepaths = {
            Path.resolve(f).name for f in Path(tmpdir, "with_layers").glob("*.nrrd")
        }
        assert with_layers_filepaths == {
            "L1_LAC_densities.nrrd",
            "L4_LAC_densities.nrrd",
            "L23_LAC_densities.nrrd",
            "L6_LAC_densities.nrrd",
            "L6_CHC_densities.nrrd",
            "L5_CHC_densities.nrrd",
        }

    def test_output_consistency(self):
        sum_ = {
            "CHC": np.zeros(self.data["annotation"].shape, dtype=float),
            "LAC": np.zeros(self.data["annotation"].shape, dtype=float),
        }
        tmpdir = self.tmpdir.name
        for filename in ["L5_CHC_densities.nrrd", "L6_CHC_densities.nrrd"]:
            filepath = str(Path(tmpdir) / "with_layers" / filename)
            sum_["CHC"] += VoxelData.load_nrrd(filepath).raw

        for filename in [
            "L1_LAC_densities.nrrd",
            "L23_LAC_densities.nrrd",
            "L4_LAC_densities.nrrd",
            "L6_LAC_densities.nrrd",
        ]:
            filepath = str(Path(tmpdir) / "with_layers" / filename)
            sum_["LAC"] += VoxelData.load_nrrd(filepath).raw

        for mtype in ["CHC", "LAC"]:
            filepath = str(Path(tmpdir) / "no_layers" / f"{mtype}_densities.nrrd")
            density = VoxelData.load_nrrd(filepath)
            npt.assert_array_equal(
                density.voxel_dimensions, self.data["annotation"].voxel_dimensions
            )
            assert density.raw.dtype == float
            npt.assert_array_almost_equal(sum_[mtype], density.raw)

    def test_output_values(self):
        tmpdir = self.tmpdir.name
        expected_densities = {
            "CHC": np.array([[[0.0, 0.0, 0.0, 2.0, 0.4]]], dtype=float),
            "LAC": np.array([[[1.5, 2.0, 1.0, 0.0, 0.6]]], dtype=float),
        }
        for mtype in ["CHC", "LAC"]:
            filepath = str(Path(tmpdir) / "no_layers" / f"{mtype}_densities.nrrd")
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
        tested.create_from_probability_map(
            self.data["annotation"],
            self.data["region_map"],
            self.data["metadata"],
            self.data["molecular_type_densities"],
            self.data["probability_map"],
            self.tmpdir.name,
        )

    def test_probability_map_sanity_negative_probability(self):
        self.data["probability_map"].loc["layer_1_gad67", "chc"] = -0.0025
        with pytest.raises(AtlasDensitiesError):
            self.create_densities()

    def test_probability_map_sanity_row_sum_is_1(self):
        self.data["probability_map"].loc["layer_1_gad67", "chc"] = 2.0
        with pytest.raises(AtlasDensitiesError):
            self.create_densities()

    def test_labels_sanity(self):
        self.data["probability_map"].rename(str.upper, axis="columns", inplace=True)
        with pytest.raises(AtlasDensitiesError):
            self.create_densities()

    def test_probability_map_layer_1_gad67_exists(self):
        self.data["probability_map"].drop("layer_1_gad67", axis="rows", inplace=True)
        with pytest.raises(AtlasDensitiesError):
            self.create_densities()

    def test_consisitent_layer_names(self):
        self.data["probability_map"].rename(index={"layer_1_gad67": "layer_7_gad67"}, inplace=True)
        with pytest.raises(AtlasDensitiesError):
            self.create_densities()

    def test_consisitent_molecular_types(self):
        self.data["probability_map"].rename(
            index={
                "layer_23_pv": "layer_23_gad67",
                "layer_4_pv": "layer_4_gad67",
                "layer_5_pv": "layer_5_gad67",
                "layer_6_pv": "layer_6_gad67",
            },
            inplace=True,
        )
        with pytest.raises(AtlasDensitiesError):
            self.create_densities()
