"""test app/mtype_densities"""
import json
import os
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest
import yaml
from click.testing import CliRunner
from voxcell import VoxelData  # type: ignore

import atlas_densities.app.mtype_densities as tested
from atlas_densities.exceptions import AtlasDensitiesError
from tests.densities.test_mtype_densities_from_map import create_from_probability_map_data
from tests.densities.test_mtype_densities_from_profiles import (
    DATA_PATH,
    create_excitatory_neuron_density,
    create_expected_cell_densities,
    create_inhibitory_neuron_density,
    create_slicer_data,
)
from tests.utils import write_json


def get_result_from_profiles(runner, td):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return runner.invoke(
            tested.app,
            [
                # fmt: off
                "--log-output-path", td,
                "create-from-profile",
                "--annotation-path", "annotation.nrrd",
                "--hierarchy-path", "hierarchy.json",
                "--metadata-path", "metadata.json",
                "--direction-vectors-path", "direction_vectors.nrrd",
                "--mtypes-config-path", "config.yaml",
                "--output-dir", "output_dir",
                # fmt: on
            ],
        )


def test_mtype_densities_from_profiles(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        create_excitatory_neuron_density().save_nrrd("excitatory_neuron_density.nrrd")
        create_inhibitory_neuron_density().save_nrrd("inhibitory_neuron_density.nrrd")
        data = create_slicer_data()
        data["annotation"].save_nrrd("annotation.nrrd")
        data["annotation"].with_data(data["direction_vectors"]).save_nrrd("direction_vectors.nrrd")
        write_json("metadata.json", data["metadata"])
        write_json("hierarchy.json", data["hierarchy"])
        with open("config.yaml", "w", encoding="utf-8") as file_:
            config = {
                "mtypeToProfileMapPath": str(DATA_PATH / "meta" / "mapping.tsv"),
                "layerSlicesPath": str(DATA_PATH / "meta" / "layers.tsv"),
                "densityProfilesDirPath": str(DATA_PATH / "mtypes"),
                "excitatoryNeuronDensityPath": "excitatory_neuron_density.nrrd",
                "inhibitoryNeuronDensityPath": "inhibitory_neuron_density.nrrd",
            }
            yaml.dump(config, file_)

        result = get_result_from_profiles(runner, td)
        assert result.exit_code == 0
        expected_cell_densities = create_expected_cell_densities()
        for mtype, expected_cell_density in expected_cell_densities.items():
            created_cell_density = VoxelData.load_nrrd(
                str(Path("output_dir", f"{mtype}_density.nrrd"))
            ).raw
            npt.assert_array_equal(created_cell_density, expected_cell_density)

        # No input density nrrd files
        with open("config.yaml", "w", encoding="utf-8") as file_:
            config = {
                "mtypeToProfileMapPath": str(DATA_PATH / "meta" / "mapping.tsv"),
                "layerSlicesPath": str(DATA_PATH / "meta" / "layers.tsv"),
                "densityProfilesDirPath": str(DATA_PATH / "mtypes"),
            }
            yaml.dump(config, file_)

        result = get_result_from_profiles(runner, td)
        assert result.exit_code == 1
        assert "neuron density file" in str(result.exception)


def test_mtype_densities_from_probability_map(tmp_path):
    data = create_from_probability_map_data()
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        td = Path(td)

        data["annotation"].save_nrrd(td / "annotation.nrrd")
        write_json("hierarchy.json", data["hierarchy"])
        data["probability_map01"].to_csv(td / "probability_map01.csv", index=True)
        data["probability_map02"].to_csv(td / "probability_map02.csv", index=True)

        for molecular_type, raw in data["molecular_type_densities"].items():
            VoxelData(
                raw,
                voxel_dimensions=data["annotation"].voxel_dimensions,
            ).save_nrrd(td / f"{molecular_type}.nrrd")

        result = runner.invoke(
            tested.app,
            [
                # fmt: off
                    "--log-output-path", str(td),
                    "create-from-probability-map",
                    "--annotation-path", "annotation.nrrd",
                    "--hierarchy-path", "hierarchy.json",
                    "--probability-map", "probability_map01.csv",
                    "--probability-map", "probability_map02.csv",
                    "--marker", "pv", "pv.nrrd",
                    "--marker", "sst", "sst.nrrd",
                    "--marker", "vip", "vip.nrrd",
                    "--marker", "gad67", "gad67.nrrd",
                    "--marker", "approx_lamp5", "approx_lamp5.nrrd",
                    "--synapse-class", "EXC",
                    "--output-dir", "output_dir",
                # fmt: on
            ],
        )
        assert result.exit_code == 0

        BPbAC = VoxelData.load_nrrd(Path("output_dir") / "BP|bAC_EXC_densities.nrrd")
        assert BPbAC.raw.dtype == float
        npt.assert_array_equal(BPbAC.voxel_dimensions, data["annotation"].voxel_dimensions)

        with open(Path("output_dir") / "metadata.json", "r") as file:
            metadata = json.load(file)

        assert "BP" in metadata["density_files"]
        assert "bAC" in metadata["density_files"]["BP"]
        assert "EXC" == metadata["synapse_class"]


class Test_mtype_densities_from_composition:
    @pytest.fixture(scope="session")
    def class_tmpdir(self, tmpdir_factory):
        """Create a session scoped temp dir using the class name"""
        return tmpdir_factory.mktemp(type(self).__name__)

    @pytest.fixture(scope="session")
    def annotation_path(self, class_tmpdir):
        path = class_tmpdir.join("annotation.nrrd")
        VoxelData(
            np.array([[[101, 102, 103, 104, 105, 106]]], dtype=np.int32),
            voxel_dimensions=[25] * 3,
        ).save_nrrd(str(path))
        return path

    @pytest.fixture(scope="session")
    def hierarchy_path(self, class_tmpdir):
        path = class_tmpdir.join("hierarchy.json")

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
                            "id": 102,
                            "acronym": "MO2",
                            "name": "Somatomotor areas, Layer 2",
                            "children": [],
                        },
                        {
                            "id": 103,
                            "acronym": "MO3",
                            "name": "Somatomotor areas, Layer 3",
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
        write_json(path, hierarchy)
        return path

    @pytest.fixture(scope="session")
    def metadata_path(self, class_tmpdir):
        path = class_tmpdir.join("metadata.json")
        metadata = {
            "region": {
                "name": "Isocortex",
                "query": "Isocortex",
                "attribute": "acronym",
                "with_descendants": True,
            },
            "layers": {
                "names": ["layer_1", "layer_2", "layer_3", "layer_4", "layer_5", "layer_6"],
                "queries": [
                    "@.*1[ab]?$",
                    "@.*2[ab]?$",
                    "@.*[^/]3[ab]?$",
                    "@.*4[ab]?$",
                    "@.*5[ab]?$",
                    "@.*6[ab]?$",
                ],
                "attribute": "acronym",
                "with_descendants": True,
            },
        }
        write_json(path, metadata)
        return path

    @pytest.fixture(scope="session")
    def density(self):
        return VoxelData(
            np.array([[[0.3, 0.3, 0.3, 0.3, 0.3, 0.3]]], dtype=np.float32),
            voxel_dimensions=[25] * 3,
        )

    @pytest.fixture(scope="session")
    def density_path(self, density, class_tmpdir):
        path = class_tmpdir.join("density.nrrd")
        density.save_nrrd(str(path))

        return path

    @pytest.fixture(scope="session")
    def taxonomy(self):
        return pd.DataFrame(
            {
                "mtype": ["L3_TPC:A", "L3_TPC:B", "L23_MC", "L4_TPC", "L4_LBC", "L4_UPC"],
                "mClass": ["PYR", "PYR", "INT", "PYR", "INT", "PYR"],
                "sClass": ["EXC", "EXC", "INH", "EXC", "INH", "EXC"],
            },
            columns=["mtype", "mClass", "sClass"],
        )

    @pytest.fixture(scope="session")
    def taxonomy_path(self, taxonomy, class_tmpdir):
        """Creates a taxonomy file and returns its path"""
        path = class_tmpdir.join("neurons-mtype-taxonomy.tsv")
        taxonomy.to_csv(path, sep="\t")

        return path

    @pytest.fixture(scope="session")
    def composition(self):
        return pd.DataFrame(
            {
                "density": [51750.099, 14785.743, 2779.081, 62321.137, 2103.119, 25921.181],
                "layer": ["layer_3", "layer_3", "layer_3", "layer_4", "layer_4", "layer_4"],
                "mtype": ["L3_TPC:A", "L3_TPC:B", "L23_MC", "L4_TPC", "L4_LBC", "L4_UPC"],
            },
            columns=["density", "layer", "mtype"],
        )

    @pytest.fixture(scope="session")
    def composition_path(self, composition, class_tmpdir):
        path = class_tmpdir.join("composition.yaml")
        out_dict = {"neurons": []}
        for row in composition.itertuples():
            out_dict["neurons"].append(
                {
                    "density": row.density,
                    "traits": {
                        "layer": int(row.layer.replace("layer_", "")),
                        "mtype": row.mtype,
                    },
                }
            )
        with open(path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(out_dict, yaml_file)

        return path

    def test_load_neuronal_mtype_taxonomy(self, taxonomy, taxonomy_path):
        pdt.assert_frame_equal(tested._load_neuronal_mtype_taxonomy(taxonomy_path), taxonomy)

    def test_validate_mtype_taxonomy(self, taxonomy):
        tested._validate_mtype_taxonomy(taxonomy)

        wrong_taxonomy = taxonomy.rename(columns={"sClass": "John"})
        with pytest.raises(AtlasDensitiesError):
            tested._validate_mtype_taxonomy(wrong_taxonomy)

        wrong_taxonomy = taxonomy.drop(columns=["mtype"])
        with pytest.raises(AtlasDensitiesError):
            tested._validate_mtype_taxonomy(wrong_taxonomy)

        wrong_taxonomy = deepcopy(taxonomy)
        wrong_taxonomy.loc[1, "sClass"] = "OTHER"
        with pytest.raises(AtlasDensitiesError):
            tested._validate_mtype_taxonomy(wrong_taxonomy)

    def test_load_neuronal_mtype_composition(self, composition, composition_path):
        pdt.assert_frame_equal(
            tested._load_neuronal_mtype_composition(composition_path), composition
        )

    def test_validate_density(self, density):
        tested._validate_density(density)

        wrong_density = density.with_data(np.zeros_like(density.raw))
        with pytest.raises(AtlasDensitiesError):
            tested._validate_density(wrong_density)

        wrong_density = deepcopy(density)
        wrong_density.raw[0, 0, 2] = -10.0
        with pytest.raises(AtlasDensitiesError):
            tested._validate_density(wrong_density)

    def test_validate_neuronal_mtype_composition(self, composition):
        tested._validate_neuronal_mtype_composition(composition)

        wrong_composition = composition.copy(deep=True)
        wrong_composition[["density", 2]] = -1.0
        with pytest.raises(AtlasDensitiesError):
            tested._validate_neuronal_mtype_composition(wrong_composition)

    def test_check_taxonomy_composition_congruency(self, taxonomy, composition):
        tested._check_taxonomy_composition_congruency(taxonomy, composition)

        with pytest.raises(AtlasDensitiesError):
            tested._check_taxonomy_composition_congruency(taxonomy.drop([1]), composition)

        with pytest.raises(AtlasDensitiesError):
            tested._check_taxonomy_composition_congruency(taxonomy, composition.drop([2]))

    def test_create_from_composition(
        self,
        annotation_path,
        hierarchy_path,
        metadata_path,
        density_path,
        taxonomy_path,
        composition_path,
        class_tmpdir,
    ):
        output_dir = class_tmpdir.mkdir("output")

        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=class_tmpdir):
            result = runner.invoke(
                tested.create_from_composition,
                [
                    # fmt: off
                    "--annotation-path", annotation_path,
                    "--hierarchy-path", hierarchy_path,
                    "--metadata-path", metadata_path,
                    "--excitatory-neuron-density-path", density_path,
                    "--taxonomy-path", taxonomy_path,
                    "--composition-path", composition_path,
                    "--output-dir", output_dir,
                    # fmt: on
                ],
            )

            assert result.exit_code == 0, result.output

            expected_filenames = {
                "L3_TPC:A_densities.nrrd",
                "L3_TPC:B_densities.nrrd",
                "L4_TPC_densities.nrrd",
                "L4_UPC_densities.nrrd",
            }

            filenames = set(os.listdir(output_dir))

            assert filenames == expected_filenames

            for filename in filenames:
                data = VoxelData.load_nrrd(str(Path(output_dir, filename)))
                assert data.shape == (1, 1, 6)
                assert not np.allclose(data.raw, 0.0)
