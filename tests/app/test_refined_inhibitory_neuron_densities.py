"""test cell_densities"""
import json
from pathlib import Path

import numpy as np
import numpy.testing as npt
from click.testing import CliRunner
from voxcell import VoxelData  # type: ignore

import atlas_densities.app.cell_densities as tested
from tests.densities.test_inhibitory_neuron_density_optimization import (
    get_initialization_data as get_optimization_data,
)
from tests.densities.test_refined_inhibitory_neuron_density import (
    get_inhibitory_neuron_densities_data,
)


def _get_inhibitory_neuron_densities_result(runner, algorithm=None):
    args = [
        "inhibitory-neuron-densities",
        "--hierarchy-path",
        "hierarchy.json",
        "--annotation-path",
        "annotation.nrrd",
        "--neuron-density-path",
        "neuron_density.nrrd",
        "--average-densities-path",
        "average_densities.csv",
        "--algorithm",
        "keep-proportions",
        "--output-dir",
        "output_dir",
    ]

    if algorithm is not None:
        args + ["--algorithm", algorithm]

    return runner.invoke(tested.app, args)


def _test(input_, algorithm=None):
    runner = CliRunner()
    with runner.isolated_filesystem():
        voxel_dimensions = [25] * 3
        VoxelData(input_["annotation"], voxel_dimensions=voxel_dimensions).save_nrrd(
            "annotation.nrrd"
        )
        VoxelData(input_["neuron_density"], voxel_dimensions=voxel_dimensions).save_nrrd(
            "neuron_density.nrrd"
        )
        with open("hierarchy.json", "w") as file_:
            json.dump(input_["hierarchy"], file_, indent=1, separators=(",", ": "))
        input_["average_densities"]["brain_region"] = input_["average_densities"].index
        input_["average_densities"].to_csv("average_densities.csv", index=False)

        result = _get_inhibitory_neuron_densities_result(runner, algorithm)

        assert result.exit_code == 0
        gad_data = VoxelData.load_nrrd(str(Path("output_dir") / "gad67+_density.nrrd"))
        assert gad_data.raw.dtype == float
        npt.assert_array_equal(gad_data.voxel_dimensions, [25.0] * 3)

        subsum_density = np.zeros_like(gad_data.raw)

        for cell_subtype in ["pv+", "sst+", "vip+"]:
            voxel_data = VoxelData.load_nrrd(
                str(Path("output_dir") / f"{cell_subtype}_density.nrrd")
            )
            assert voxel_data.raw.dtype == float
            npt.assert_array_equal(voxel_data.voxel_dimensions, [25.0] * 3)
            subsum_density = subsum_density + voxel_data.raw

        assert np.all(input_["neuron_density"] >= gad_data.raw)
        assert np.all(gad_data.raw >= subsum_density)

        # Test assertion in case of invalid hierarchy file
        with open("hierarchy.json", "w", encoding="utf-8") as file_:
            hierarchy = {"msg": [input_["hierarchy"], {}]}
            json.dump(hierarchy, file_, indent=1, separators=(",", ": "))

        result = _get_inhibitory_neuron_densities_result(runner, algorithm)
        assert "Unexpected JSON layout" in str(result.exception)

        # Check that an exception is thrown if the input neuron density has negative values
        input_["neuron_density"][0, 0, 0] = -1.0
        VoxelData(input_["neuron_density"], voxel_dimensions=(10, 10, 10)).save_nrrd(
            "neuron_density.nrrd"
        )
        result = _get_inhibitory_neuron_densities_result(runner, algorithm)
        assert "Negative density value" in str(result.exception)


def test_inhibitory_neuron_densities():
    input_ = get_inhibitory_neuron_densities_data()
    _test(input_, "keep-proportions")


def test_inhibitory_neuron_densities_opt():
    input_ = get_optimization_data()
    _test(input_, "linprog")
