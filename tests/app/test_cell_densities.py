"""test cell_densities"""
import json
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import yaml  # type: ignore
from click.testing import CliRunner
from voxcell import VoxelData  # type: ignore

import atlas_densities.app.cell_densities as tested
from atlas_densities.densities.cell_counts import (
    extract_inhibitory_neurons_dataframe,
    glia_cell_counts,
    inhibitory_data,
)
from atlas_densities.densities.measurement_to_density import remove_non_density_measurements
from tests.densities.test_excel_reader import (
    check_columns_na,
    check_non_negative_values,
    get_invalid_region_names,
)
from tests.densities.test_fitting import get_fitting_input_data
from tests.densities.test_glia_densities import get_glia_input_data
from tests.densities.test_inhibitory_neuron_density import get_inhibitory_neuron_input_data
from tests.densities.test_measurement_to_density import (
    get_expected_output as get_average_densities_expected_output,
)
from tests.densities.test_measurement_to_density import (
    get_input_data as get_measurement_to_density_input_data,
)

TEST_PATH = Path(Path(__file__).parent.parent)
DATA_PATH = Path(TEST_PATH.parent, "atlas_densities", "app", "data")
MEASUREMENTS_PATH = DATA_PATH / "measurements"


def _get_cell_density_result(runner):
    args = [
        "cell-density",
        "--hierarchy-path",
        str(Path(TEST_PATH, "1.json")),
        "--annotation-path",
        "annotation.nrrd",
        "--nissl-path",
        "nissl.nrrd",
        "--output-path",
        "overall_cell_density.nrrd",
    ]

    return runner.invoke(tested.app, args)


def test_cell_density():
    input_ = {
        "annotation": np.array(
            [
                [[512, 512, 1143]],
                [[512, 512, 1143]],
                [[477, 56, 485]],
            ]
        ),
        "nissl": np.array(
            [
                [[0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0]],
                [[0.0, 1.0, 0.0]],
            ]
        ),
    }
    voxel_dimensions = [25] * 3
    runner = CliRunner()
    with runner.isolated_filesystem():
        for name, array in input_.items():
            VoxelData(array, voxel_dimensions=voxel_dimensions).save_nrrd(f"{name}.nrrd")
        result = _get_cell_density_result(runner)

        assert result.exit_code == 0
        voxel_data = VoxelData.load_nrrd("overall_cell_density.nrrd")
        assert voxel_data.raw.dtype == float

        # An error should be raised if annotation and nissl don't use the same voxel dimensions
        VoxelData(np.ones((3, 1, 3)), voxel_dimensions=[10] * 3).save_nrrd("nissl.nrrd")
        result = _get_cell_density_result(runner)
        assert "voxel_dimensions" in str(result.exception)


def _get_glia_cell_densities_result(runner):
    args = [
        "glia-cell-densities",
        "--annotation-path",
        "annotation.nrrd",
        "--hierarchy-path",
        str(Path(TEST_PATH, "1.json")),
        "--cell-density-path",
        "cell_density.nrrd",
        "--glia-density-path",
        "glia_density.nrrd",
        "--astrocyte-density-path",
        "astrocyte_density.nrrd",
        "--oligodendrocyte-density-path",
        "oligodendrocyte_density.nrrd",
        "--microglia-density-path",
        "microglia_density.nrrd",
        "--glia-proportions-path",
        "glia_proportions.json",
        "--output-dir",
        "densities",
    ]

    return runner.invoke(tested.app, args)


def test_glia_cell_densities():
    glia_cell_count = sum(glia_cell_counts().values())
    input_ = get_glia_input_data(glia_cell_count)
    runner = CliRunner()
    with runner.isolated_filesystem():
        voxel_dimensions = (25, 25, 25)
        VoxelData(input_["annotation"], voxel_dimensions=voxel_dimensions).save_nrrd(
            "annotation.nrrd"
        )
        VoxelData(input_["cell_density"], voxel_dimensions=voxel_dimensions).save_nrrd(
            "cell_density.nrrd"
        )
        for (glia_type, unconstrained_density) in input_["glia_densities"].items():
            VoxelData(unconstrained_density, voxel_dimensions=voxel_dimensions).save_nrrd(
                glia_type + "_density.nrrd"
            )
        with open("glia_proportions.json", "w", encoding="utf-8") as out:
            json.dump(input_["glia_proportions"], out)
        result = _get_glia_cell_densities_result(runner)
        assert result.exit_code == 0

        neuron_density = VoxelData.load_nrrd("densities/neuron_density.nrrd")
        assert neuron_density.raw.dtype == np.float64
        npt.assert_array_equal(neuron_density.shape, input_["annotation"].shape)
        assert np.all(neuron_density.raw >= 0.0)

        oligodendrocyte_density = VoxelData.load_nrrd("densities/oligodendrocyte_density.nrrd")
        assert oligodendrocyte_density.raw.dtype == np.float64
        npt.assert_array_equal(neuron_density.shape, input_["annotation"].shape)

        # Check that an exception is thrown if voxel dimensions aren't consistent
        VoxelData(input_["cell_density"], voxel_dimensions=(10, 10, 10)).save_nrrd(
            "cell_density.nrrd"
        )
        result = _get_glia_cell_densities_result(runner)
        assert "voxel_dimensions" in str(result.exception)

        # Check that an exception is thrown if the input cell density has negative values
        input_["cell_density"][0, 0, 0] = -1.0
        VoxelData(input_["cell_density"], voxel_dimensions=(10, 10, 10)).save_nrrd(
            "cell_density.nrrd"
        )
        result = _get_glia_cell_densities_result(runner)
        assert "Negative density value" in str(result.exception)


def _get_inh_and_exc_neuron_densities_result(runner):
    args = [
        "inhibitory-and-excitatory-neuron-densities",
        "--annotation-path",
        "annotation.nrrd",
        "--hierarchy-path",
        str(Path(TEST_PATH, "1.json")),
        "--gad1-path",
        "gad1.nrrd",
        "--nrn1-path",
        "nrn1.nrrd",
        "--neuron-density-path",
        "neuron_density.nrrd",
        "--output-dir",
        "densities",
    ]

    return runner.invoke(tested.app, args)


def test_inhibitory_and_excitatory_neuron_densities():
    inhibitory_df = extract_inhibitory_neurons_dataframe(Path(MEASUREMENTS_PATH, "mmc1.xlsx"))
    neuron_count = inhibitory_data(inhibitory_df)["neuron_count"]
    input_ = get_inhibitory_neuron_input_data(neuron_count)
    runner = CliRunner()
    with runner.isolated_filesystem():
        voxel_dimensions = (25, 25, 25)
        for name in ["annotation", "neuron_density", "gad1", "nrn1"]:
            VoxelData(input_[name], voxel_dimensions=voxel_dimensions).save_nrrd(name + ".nrrd")

        result = _get_inh_and_exc_neuron_densities_result(runner)
        assert result.exit_code == 0

        inh_neuron_density = VoxelData.load_nrrd("densities/inhibitory_neuron_density.nrrd")
        assert inh_neuron_density.raw.dtype == np.float64
        npt.assert_array_equal(inh_neuron_density.shape, input_["annotation"].shape)
        assert np.all(inh_neuron_density.raw >= 0.0)

        exc_neuron_density = VoxelData.load_nrrd("densities/excitatory_neuron_density.nrrd")
        assert exc_neuron_density.raw.dtype == np.float64
        npt.assert_array_equal(exc_neuron_density.shape, input_["annotation"].shape)
        assert np.all(exc_neuron_density.raw >= 0.0)

        # Check that an exception is thrown if voxel dimensions aren't consistent
        VoxelData(input_["neuron_density"], voxel_dimensions=(10, 10, 10)).save_nrrd(
            "neuron_density.nrrd"
        )
        result = _get_inh_and_exc_neuron_densities_result(runner)
        assert "voxel_dimensions" in str(result.exception)

        # Check that an exception is thrown if the input neuron density has negative values
        input_["neuron_density"][0, 0, 0] = -1.0
        VoxelData(input_["neuron_density"], voxel_dimensions=(25, 25, 25)).save_nrrd(
            "neuron_density.nrrd"
        )
        result = _get_inh_and_exc_neuron_densities_result(runner)
        assert "Negative density value" in str(result.exception)


def _get_compile_measurements_result(runner):
    args = [
        "compile-measurements",
        "--measurements-output-path",
        "measurements.csv",
        "--homogenous-regions-output-path",
        "homogenous_regions.csv",
    ]

    return runner.invoke(tested.app, args)


def test_compile_measurements():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = _get_compile_measurements_result(runner)
        assert result.exit_code == 0

        dataframe = pd.read_csv("measurements.csv")

        assert get_invalid_region_names(dataframe) == {
            "Nucleus of reunions",
            "Nucleus accumbens, core",
            "Nucleus accumbens, shell",
            "Kolliker-Fuse subnucleus",
            "Periaqueductal gray, dorsal lateral",
            "Periaqueductal gray, dorsal medial",
            "Periaqueductal gray, lateral",
            "Periaqueductal gray, ventral lateral",
            "Somatosensory cortex",
            "Superior Colliculus",
            "Prelimbic area, layer 4",
            "Medulla, unassigned",
            "Midbrain, motor related, other",
        }

        check_columns_na(dataframe)
        check_non_negative_values(dataframe)

        dataframe = pd.read_csv("homogenous_regions.csv")
        assert set(dataframe["cell_type"]) == {"inhibitory", "excitatory"}


def _get_measurements_to_average_densities_result(runner, hierarchy_path, measurements_path):
    args = [
        "measurements-to-average-densities",
        "--hierarchy-path",
        hierarchy_path,
        "--annotation-path",
        "annotation.nrrd",
        "--cell-density-path",
        "cell_density.nrrd",
        "--neuron-density-path",
        "neuron_density.nrrd",
        "--measurements-path",
        measurements_path,
        "--output-path",
        "average_densities.csv",
    ]

    return runner.invoke(tested.app, args)


def test_measurements_to_average_densities():
    runner = CliRunner()
    with runner.isolated_filesystem():
        input_ = get_measurement_to_density_input_data()
        voxel_dimensions = input_["voxel_dimensions"]
        for name in ["annotation", "cell_density", "neuron_density"]:
            VoxelData(input_[name], voxel_dimensions=voxel_dimensions).save_nrrd(name + ".nrrd")
        input_["measurements"].to_csv("measurements.csv", index=False)
        with open("hierarchy.json", "w", encoding="utf-8") as out:
            json.dump(input_["hierarchy"], out, indent=1, separators=(",", ": "))

        result = _get_measurements_to_average_densities_result(
            runner, hierarchy_path="hierarchy.json", measurements_path="measurements.csv"
        )

        assert result.exit_code == 0

        actual = pd.read_csv("average_densities.csv")
        expected = get_average_densities_expected_output()
        remove_non_density_measurements(expected)  # One volume measurement to remove
        pdt.assert_frame_equal(actual, expected)

        result = _get_measurements_to_average_densities_result(
            runner,
            hierarchy_path=DATA_PATH / "1.json",
            measurements_path=MEASUREMENTS_PATH / "measurements.csv",
        )
        assert result.exit_code == 0
        actual = pd.read_csv("average_densities.csv")
        assert np.all(actual["measurement_type"] == "cell density")
        assert np.all(actual["measurement_unit"] == "number of cells per mm^3")


def _get_fitting_result(runner):
    args = [
        "fit-average-densities",
        "--hierarchy-path",
        "hierarchy.json",
        "--annotation-path",
        "annotation.nrrd",
        "--neuron-density-path",
        "neuron_density.nrrd",
        "--gene-config-path",
        "gene_config.yaml",
        "--average-densities-path",
        "average_densities.csv",
        "--homogenous-regions-path",
        "homogenous_regions.csv",
        "--fitted-densities-output-path",
        "fitted_densities.csv",
        "--fitting-maps-output-path",
        "fitting_maps.json",
    ]

    return runner.invoke(tested.app, args)


def test_fit_average_densities():
    runner = CliRunner()
    with runner.isolated_filesystem():
        input_ = get_fitting_input_data()
        for name in ["annotation", "neuron_density"]:
            VoxelData(input_[name], voxel_dimensions=[25.0] * 3).save_nrrd(name + ".nrrd")

        input_["homogenous_regions"].to_csv("homogenous_regions.csv", index=False)
        input_["average_densities"].to_csv("average_densities.csv", index=False)

        with open("hierarchy.json", "w", encoding="utf-8") as out:
            json.dump(input_["hierarchy"], out, indent=1, separators=(",", ": "))

        with open("realigned_slices.json", "w", encoding="utf-8") as out:
            json.dump(input_["realigned_slices"], out, indent=1, separators=(",", ": "))

        with open("std_cells.json", "w", encoding="utf-8") as out:
            json.dump(input_["cell_density_stddevs"], out, indent=1, separators=(",", ": "))

        with open("gene_config.yaml", "w", encoding="utf-8") as out:
            gene_config = {
                "inputGeneVolumePath": {},
                "sectionDataSetID": {},
                "realignedSlicesPath": "realigned_slices.json",
                "cellDensityStandardDeviationsPath": "std_cells.json",
            }
            for marker, intensity in input_["gene_marker_volumes"].items():
                VoxelData(intensity["intensity"], voxel_dimensions=[25.0] * 3).save_nrrd(
                    marker + ".nrrd"
                )
                gene_config["inputGeneVolumePath"][marker] = marker + ".nrrd"
                gene_config["sectionDataSetID"][marker] = input_["slice_map"][marker]

            yaml.dump(gene_config, out)

        result = _get_fitting_result(runner)
        assert result.exit_code == 0

        densities = pd.read_csv("fitted_densities.csv")
        assert set(densities.columns) == {
            "brain_region",
            "gad67+",
            "gad67+_standard_deviation",
            "pv+",
            "pv+_standard_deviation",
        }
        densities.set_index("brain_region", inplace=True)
        assert np.allclose(densities.at["Hippocampal formation", "pv+"], 2.0)
        assert densities.at["Hippocampal formation", "pv+_standard_deviation"] >= 0.0

        with open("fitting_maps.json", "r", encoding="utf-8") as file_:
            fitting_maps = json.load(file_)
            assert set(fitting_maps.keys()) == {"Cerebellum group", "Isocortex group", "Rest"}
            assert np.allclose(fitting_maps["Rest"]["pv+"]["coefficient"], 2.0 / 3.0)

        # Check that an exception is thrown if the input neuron density has negative values
        input_["neuron_density"][0, 0, 0] = -1.0
        VoxelData(input_["neuron_density"], voxel_dimensions=(10, 10, 10)).save_nrrd(
            "neuron_density.nrrd"
        )
        result = _get_fitting_result(runner)
        assert "Negative density value" in str(result.exception)
