from pathlib import Path
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from voxcell import RegionMap, VoxelData

import atlas_densities.densities.excitatory_inhibitory_splitting as tested
from atlas_densities.app.utils import DATA_PATH

TESTS_PATH = Path(__file__).parent.parent
HIERARCHY_PATH = Path(TESTS_PATH, "1.json")


def test_scale_excitatory_densities(tmp_path):
    shape = (
        3,
        3,
    )
    brain_regions = VoxelData(
        np.ones(shape),
        voxel_dimensions=(10.0, 10.0),
    )
    mapping = pd.DataFrame()
    layer_ids = pd.DataFrame()
    density = VoxelData(
        np.ones(shape),
        voxel_dimensions=(10.0, 10.0),
    )
    tested.scale_excitatory_densities(tmp_path, brain_regions, mapping, layer_ids, density)

    # empty mapping
    output = tmp_path.glob("*_cADpyr.nrrd")
    assert len(list(output)) == 0

    mapping = pd.DataFrame(
        {
            "L2_IPC": {
                "layer_2": 0.1,
                "layer_3": 0,
            },
            "L2_TPC:A": {
                "layer_2": 0.2,
                "layer_3": 0,
            },
            "L2_TPC:B": {
                "layer_2": 0.8,
                "layer_3": 0,
            },
            "L3_TPC:A": {
                "layer_2": 0,
                "layer_3": 0.7,
            },
        }
    )
    mapping.index.name = "layer"
    layer_ids = {"layer_2": [], "layer_3": []}
    tested.scale_excitatory_densities(tmp_path, brain_regions, mapping, layer_ids, density)
    output = list(tmp_path.glob("*_cADpyr.nrrd"))

    filenames = {n.name: n for n in output}
    assert set(filenames) == {
        "L3_TPC:A_cADpyr.nrrd",
        "L2_IPC_cADpyr.nrrd",
        "L2_TPC:A_cADpyr.nrrd",
        "L2_TPC:B_cADpyr.nrrd",
    }
    # no ids specified, all densities should be 0
    assert all(VoxelData.load_nrrd(v).raw.sum() == 0.0 for v in filenames.values())

    layer_ids = {"layer_2": [], "layer_3": [1]}
    tested.scale_excitatory_densities(tmp_path, brain_regions, mapping, layer_ids, density)
    output = list(tmp_path.glob("*_cADpyr.nrrd"))
    values = {k: VoxelData.load_nrrd(v).raw.sum() for k, v in filenames.items()}
    assert values["L2_IPC_cADpyr.nrrd"] == 0.0
    assert values["L2_TPC:B_cADpyr.nrrd"] == 0.0
    assert values["L2_TPC:A_cADpyr.nrrd"] == 0.0
    assert values["L3_TPC:A_cADpyr.nrrd"] == pytest.approx(0.7 * 3 * 3)


def test_set_ids_to_zero_and_save(tmp_path):
    shape = (
        3,
        3,
    )
    raw = np.ones(shape)
    path = tmp_path / "zero.nrrd"
    brain_regions = VoxelData(
        raw.copy(),
        voxel_dimensions=(10.0, 10.0),
    )
    nrrd = VoxelData(
        raw.copy(),
        voxel_dimensions=(10.0, 10.0),
    )
    tested.set_ids_to_zero_and_save(
        path,
        brain_regions,
        nrrd,
        [
            1,
        ],
    )

    output = VoxelData.load_nrrd(path)
    assert output.shape == nrrd.shape
    assert np.sum(output.raw) == 0.0

    raw[0, 0] = 10
    raw[0, 1] = 11
    brain_regions = VoxelData(raw, voxel_dimensions=(10.0, 10.0))
    tested.set_ids_to_zero_and_save(path, brain_regions, nrrd, [1, 10])
    output = VoxelData.load_nrrd(path)

    assert np.sum(output.raw) == 1.0


def test_make_excitatory_density():
    shape = (
        3,
        3,
    )
    neuron_density = VoxelData(
        2 * np.ones(shape),
        voxel_dimensions=(10.0, 10.0),
    )
    inhibitory_density = VoxelData(
        np.ones(shape),
        voxel_dimensions=(10.0, 10.0),
    )
    res = tested.make_excitatory_density(neuron_density, inhibitory_density)

    assert res.shape == neuron_density.shape
    assert np.sum(res.raw) == np.prod(neuron_density.shape)

    # this would create negative densities; make sure they are clipped to zero
    res = tested.make_excitatory_density(inhibitory_density, neuron_density)
    assert res.shape == neuron_density.shape
    assert np.sum(res.raw) == 0.0


def test_gather_isocortex_ids_from_metadata():
    region_map = RegionMap.load_json(HIERARCHY_PATH)
    metadata_path = DATA_PATH / "metadata" / "excitatory-inhibitory-splitting.json"
    res = tested.gather_isocortex_ids_from_metadata(region_map, metadata_path)
    acronyms = {
        "layer_1": [
            "ACA1",
            "ACAd1",
            "ACAv1",
            "AId1",
            "AIp1",
            "AIv1",
            "AUDd1",
            "AUDp1",
            "AUDpo1",
            "AUDv1",
            "ECT1",
            "FRP1",
            "GU1",
            "ILA1",
            "MO1",
            "MOp1",
            "MOs1",
            "ORB1",
            "ORBl1",
            "ORBm1",
            "ORBvl1",
            "PERI1",
            "PL1",
            "PTLp1",
            "RSPagl1",
            "RSPd1",
            "RSPv1",
            "SS1",
            "SSp-bfd1",
            "SSp-ll1",
            "SSp-m1",
            "SSp-n1",
            "SSp-tr1",
            "SSp-ul1",
            "SSp-un1",
            "SSp1",
            "SSs1",
            "TEa1",
            "VIS1",
            "VISC1",
            "VISa1",
            "VISal1",
            "VISam1",
            "VISl1",
            "VISli1",
            "VISlla1",
            "VISm1",
            "VISmma1",
            "VISmmp1",
            "VISp1",
            "VISpl1",
            "VISpm1",
            "VISpor1",
            "VISrl1",
            "VISrll1",
        ],
        "layer_2": ["PL2", "ORBm2", "RSPv2", "ILA2"],
        "layer_3": [],
        "layer_4": [
            "AUDd4",
            "AUDp4",
            "AUDpo4",
            "AUDv4",
            "GU4",
            "PTLp4",
            "RSPd4",
            "SS4",
            "SSp-bfd4",
            "SSp-ll4",
            "SSp-m4",
            "SSp-n4",
            "SSp-tr4",
            "SSp-ul4",
            "SSp-un4",
            "SSp4",
            "SSs4",
            "TEa4",
            "VIS4",
            "VISC4",
            "VISa4",
            "VISal4",
            "VISam4",
            "VISl4",
            "VISli4",
            "VISlla4",
            "VISm4",
            "VISmma4",
            "VISmmp4",
            "VISp4",
            "VISpl4",
            "VISpm4",
            "VISpor4",
            "VISrl4",
            "VISrll4",
        ],
        "layer_5": [
            "ACA5",
            "ACAd5",
            "ACAv5",
            "AId5",
            "AIp5",
            "AIv5",
            "AUDd5",
            "AUDp5",
            "AUDpo5",
            "AUDv5",
            "ECT5",
            "FRP5",
            "GU5",
            "ILA5",
            "MO5",
            "MOp5",
            "MOs5",
            "ORB5",
            "ORBl5",
            "ORBm5",
            "ORBvl5",
            "PERI5",
            "PL5",
            "PTLp5",
            "RSPagl5",
            "RSPd5",
            "RSPv5",
            "SS5",
            "SSp-bfd5",
            "SSp-ll5",
            "SSp-m5",
            "SSp-n5",
            "SSp-tr5",
            "SSp-ul5",
            "SSp-un5",
            "SSp5",
            "SSs5",
            "TEa5",
            "VIS5",
            "VISC5",
            "VISa5",
            "VISal5",
            "VISam5",
            "VISl5",
            "VISli5",
            "VISlla5",
            "VISm5",
            "VISmma5",
            "VISmmp5",
            "VISp5",
            "VISpl5",
            "VISpm5",
            "VISpor5",
            "VISrl5",
            "VISrll5",
        ],
        "layer_6": [
            "ACA6a",
            "ACA6b",
            "ACAd6a",
            "ACAd6b",
            "ACAv6a",
            "ACAv6b",
            "AId6a",
            "AId6b",
            "AIp6a",
            "AIp6b",
            "AIv6a",
            "AIv6b",
            "AUDd6a",
            "AUDd6b",
            "AUDp6a",
            "AUDp6b",
            "AUDpo6a",
            "AUDpo6b",
            "AUDv6a",
            "AUDv6b",
            "ECT6a",
            "ECT6b",
            "FRP6a",
            "FRP6b",
            "GU6a",
            "GU6b",
            "ILA6a",
            "ILA6b",
            "MO6a",
            "MO6b",
            "MOp6a",
            "MOp6b",
            "MOs6a",
            "MOs6b",
            "ORB6a",
            "ORB6b",
            "ORBl6a",
            "ORBl6b",
            "ORBm6a",
            "ORBm6b",
            "ORBvl6a",
            "ORBvl6b",
            "PERI6a",
            "PERI6b",
            "PL6a",
            "PL6b",
            "PTLp6a",
            "PTLp6b",
            "RSPagl6a",
            "RSPagl6b",
            "RSPd6a",
            "RSPd6b",
            "RSPv6a",
            "RSPv6b",
            "SS6a",
            "SS6b",
            "SSp-bfd6a",
            "SSp-bfd6b",
            "SSp-ll6a",
            "SSp-ll6b",
            "SSp-m6a",
            "SSp-m6b",
            "SSp-n6a",
            "SSp-n6b",
            "SSp-tr6a",
            "SSp-tr6b",
            "SSp-ul6a",
            "SSp-ul6b",
            "SSp-un6a",
            "SSp-un6b",
            "SSp6a",
            "SSp6b",
            "SSs6a",
            "SSs6b",
            "TEa6a",
            "TEa6b",
            "VIS6a",
            "VIS6b",
            "VISC6a",
            "VISC6b",
            "VISa6a",
            "VISa6b",
            "VISal6a",
            "VISal6b",
            "VISam6a",
            "VISam6b",
            "VISl6a",
            "VISl6b",
            "VISli6a",
            "VISli6b",
            "VISlla6a",
            "VISlla6b",
            "VISm6a",
            "VISm6b",
            "VISmma6a",
            "VISmma6b",
            "VISmmp6a",
            "VISmmp6b",
            "VISp6a",
            "VISp6b",
            "VISpl6a",
            "VISpl6b",
            "VISpm6a",
            "VISpm6b",
            "VISpor6a",
            "VISpor6b",
            "VISrl6a",
            "VISrl6b",
            "VISrll6a",
            "VISrll6b",
        ],
    }

    for layer, ids in res.items():
        assert set(region_map.get(i, "acronym") for i in ids) == set(acronyms[layer])
