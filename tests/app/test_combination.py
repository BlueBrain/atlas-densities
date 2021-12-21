"""test markers_combinator"""
import json
from pathlib import Path

import numpy as np
import numpy.testing as npt
from click.testing import CliRunner
from voxcell import VoxelData  # type: ignore

import atlas_densities.app.combination as tested

TEST_PATH = Path(Path(__file__).parent.parent)


def test_combine_markers():
    expected_proportions = [4.0 / 23.0, 8.0 / 23.0, 11.0 / 23.0]
    input_ = {
        # Cerebellum ids: 512, 1143
        # Striatum ids: 56, 477, 485
        "annotation": np.array(
            [
                [[512, 512, 1143]],
                [[512, 512, 1143]],
                [[477, 56, 485]],
            ]
        ),
        "cnp": np.array(
            [
                [[0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0]],
                [[0.0, 1.0, 0.0]],
            ]
        ),
        "mbp": np.array(
            [
                [[0.0, 1.0, 0.0]],
                [[9.0, 1.0, 0.0]],
                [[0.0, 1.0, 0.0]],
            ]
        ),
        "gfap": np.array(
            [
                [[0.0, 1.0, 5.0]],
                [[1.0, 4.0, 5.0]],
                [[0.0, 1.0, 0.0]],
            ]
        ),
        "s100b": np.array(
            [
                [[0.0, 1.0, 0.0]],
                [[0.0, 1.0, 0.0]],
                [[0.0, 1.0, 0.0]],
            ]
        ),
        "aldh1l1": np.array(
            [
                [[1.0, 1.0, 0.0]],
                [[2.0, 1.0, 4.0]],
                [[0.0, 0.0, 0.0]],
            ]
        ),
        "tmem119": np.array(
            [
                [[0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0]],
            ]
        ),
    }
    expected_glia_intensities = {
        "astrocyte": np.array(
            [
                [[2.0, 6.0, 15.0]],
                [[7.0, 15.0, 23.0]],
                [[0.0, 4.0, 0.0]],
            ]
        )
        / 6.0,
        "microglia": np.array(
            [
                [[0.0, 3.0, 0.0]],
                [[0.0, 0.0, 3.0]],
                [[3.0, 0.0, 0.0]],
            ]
        )
        / 3.0,
        "oligodendrocyte": np.array(
            [
                [[0.0, 2.0, 0.0]],
                [[18.0, 3.0, 0.0]],
                [[0.0, 3.0, 0.0]],
            ]
        )
        / 3.0,
    }
    expected_glia_intensities["glia"] = np.average(
        list(expected_glia_intensities.values()), axis=0, weights=expected_proportions
    )
    voxel_dimensions = [25] * 3
    runner = CliRunner()
    with runner.isolated_filesystem():
        for name, array in input_.items():
            VoxelData(array, voxel_dimensions=voxel_dimensions).save_nrrd(f"{name}.nrrd")
        result = runner.invoke(
            tested.app,
            [
                "combine-markers",
                "--annotation-path",
                "annotation.nrrd",
                "--hierarchy-path",
                str(Path(TEST_PATH, "1.json")),
                "--config",
                str(Path(TEST_PATH, "markers_config.yaml")),
            ],
        )
        assert result.exit_code == 0
        with open("glia_proportions.json", encoding="utf-8") as file_:
            glia_proportions = json.load(file_)
            assert glia_proportions == {
                "astrocyte": str(4.0 / 23.0),
                "microglia": str(8.0 / 23.0),
                "glia": str(1.0),
                "oligodendrocyte": str(11.0 / 23.0),
            }
        for type_, arr in expected_glia_intensities.items():
            voxel_data = VoxelData.load_nrrd(f"{type_}.nrrd")
            npt.assert_array_almost_equal(voxel_data.raw, arr, decimal=5)
