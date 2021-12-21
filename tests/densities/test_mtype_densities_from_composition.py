import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from numpy import testing as npt
from pandas import testing as pdt
from voxcell import RegionMap, VoxelData  # type: ignore

from atlas_densities.densities import mtype_densities_from_composition as tested


@pytest.fixture
def taxonomy():
    return pd.DataFrame(
        {
            "mtype": ["L3_TPC:A", "L3_TPC:B", "L32_MC", "L4_TPC", "L4_LBC", "L4_UPC"],
            "mClass": ["PYR", "PYR", "INT", "PYR", "INT", "PYR"],
            "sClass": ["EXC", "EXC", "INH", "EXC", "INH", "EXC"],
        },
        columns=["mtype", "mClass", "sClass"],
    )


def test_excitatory_mtypes_from_taxonomy(taxonomy):
    result = tested._excitatory_mtypes_from_taxonomy(taxonomy)
    assert result == ("L3_TPC:A", "L3_TPC:B", "L4_TPC", "L4_UPC")


@pytest.fixture
def composition():
    return pd.DataFrame(
        {
            "density": [51750.099, 14785.743, 2779.081, 62321.137, 2103.119, 25921.181],
            "layer": ["layer_3", "layer_3", "layer_3", "layer_4", "layer_4", "layer_4"],
            "mtype": ["L3_TPC:A", "L3_TPC:B", "L23_MC", "L4_TPC", "L4_LBC", "L4_UPC"],
        },
        columns=["density", "layer", "mtype"],
    )


@pytest.fixture
def annotation():
    return VoxelData(
        np.array([[[101, 102, 103, 104, 105, 106]]], dtype=int), voxel_dimensions=[25] * 3
    )


@pytest.fixture
def region_map():
    return RegionMap.from_dict(
        {
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
                            "name": "Somatomotor areas, Layer 2/3",
                            "children": [],
                        },
                        {
                            "id": 103,
                            "acronym": "MO3",
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
    )


@pytest.fixture
def metadata():
    return {
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


@pytest.fixture
def density(annotation):
    return np.full_like(annotation.raw, fill_value=0.3, dtype=np.float32)


def test_create_from_composition(annotation, region_map, metadata, density, taxonomy, composition):

    per_mtype_density = {
        mtype: mtype_density
        for mtype, mtype_density in tested.create_from_composition(
            annotation, region_map, metadata, density, taxonomy, composition
        )
    }

    expected_mtypes = ("L3_TPC:A", "L3_TPC:B", "L4_TPC", "L4_UPC")

    assert per_mtype_density.keys() == set(expected_mtypes)

    expected_layers = (3, 3, 4, 4)

    for mtype, layer in zip(expected_mtypes, expected_layers):

        layer_sum = composition[
            (composition["layer"] == f"layer_{layer}")
            & (composition["mtype"].isin(expected_mtypes))
        ]["density"].sum()

        mtype_average_density = composition[composition["mtype"] == mtype]["density"].values[0]

        # starting from zero positional layer
        pos_layer = layer - 1

        expected_density = np.zeros_like(density)
        expected_density[0, 0, pos_layer] = (mtype_average_density / layer_sum) * density[
            ..., pos_layer
        ]

        mtype_density = per_mtype_density[mtype]

        assert not np.allclose(mtype_density, 0.0)
        npt.assert_allclose(mtype_density, expected_density)
