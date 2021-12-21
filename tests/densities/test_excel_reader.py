"""
Unit tests for excel_reader
"""

import re
import warnings
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
from voxcell import RegionMap

import atlas_densities.densities.excel_reader as tested
from atlas_densities.exceptions import AtlasDensitiesWarning

TEST_PATH = Path(__file__).parent.parent
MEASUREMENTS_PATH = Path(
    Path(__file__).parent.parent.parent, "atlas_densities", "app", "data", "measurements"
)

REGULAR_COLUMNS = {
    "brain_region": np.dtype("O"),
    "measurement": float,
    "standard_deviation": float,
    "measurement_type": np.dtype("O"),
    "measurement_unit": np.dtype("O"),
    "cell_type": np.dtype("O"),
}

SOURCE_COLUMNS = {
    "source_title": np.dtype("O"),
    "comment": np.dtype("O"),
    "specimen_age": np.dtype("O"),
}

REGION_MAP = RegionMap.load_json(Path(TEST_PATH, "1.json"))
AIBS_REGION_IDS = REGION_MAP.find(
    "Basic cell groups and regions", attr="name", with_descendants=True
)
AIBS_REGIONS = set(map(lambda id_: REGION_MAP.get(id_, "name"), AIBS_REGION_IDS))


def test_compute_kim_et_al_neuron_densities():
    with warnings.catch_warnings(record=True) as warnings_:
        dataframe = tested.compute_kim_et_al_neuron_densities(Path(MEASUREMENTS_PATH, "mmc3.xlsx"))
        warnings_ = [w for w in warnings_ if isinstance(w.message, AtlasDensitiesWarning)]
        regions_with_nd = []
        regions_with_invalid_full_name = []
        nd_warning_regexp = re.compile(r'Region (.*) with acronym (.*) has "N/D" values')
        full_name_regexp = re.compile(r"Region with acronym (.*) has no valid full name")
        for warning in warnings_:
            found = nd_warning_regexp.search(str(warning))
            if found is not None:
                regions_with_nd.append((found.group(2), found.group(1)))
            found = full_name_regexp.search(str(warning))
            if found is not None:
                regions_with_invalid_full_name.append(found.group(1))
        npt.assert_array_equal(
            regions_with_nd,
            [
                ("SF", "Septofimbrial nucleus"),
                ("TRS", "Triangular nucleus of septum"),
                ("PVH", "Paraventricular hypothalamic nucleus"),
                (
                    "PVHm",
                    "Paraventricular hypothalamic nucleus, magnocellular division",
                ),
                (
                    "PVHmm",
                    "Paraventricular hypothalamic nucleus, magnocellular division, medial magnocellular part",
                ),
                (
                    "PVHpm",
                    "Paraventricular hypothalamic nucleus, magnocellular division, posterior magnocellular part",
                ),
                (
                    "PVHpml",
                    "Paraventricular hypothalamic nucleus, magnocellular division, posterior magnocellular part, lateral zone",
                ),
                (
                    "PVHpmm",
                    "Paraventricular hypothalamic nucleus, magnocellular division, posterior magnocellular part, medial zone",
                ),
                (
                    "PVHp",
                    "Paraventricular hypothalamic nucleus, parvicellular division",
                ),
                (
                    "PVHap",
                    "Paraventricular hypothalamic nucleus, parvicellular division, anterior parvicellular part",
                ),
                (
                    "PVHmpd",
                    "Paraventricular hypothalamic nucleus, parvicellular division, medial parvicellular part, dorsal zone",
                ),
                (
                    "PVHpv",
                    "Paraventricular hypothalamic nucleus, parvicellular division, periventricular part",
                ),
                ("PVa", "Periventricular hypothalamic nucleus, anterior part"),
                ("PVi", "Periventricular hypothalamic nucleus, intermediate part"),
                ("KF", "Kolliker-Fuse subnucleus"),
            ],
        )
        npt.assert_array_equal(regions_with_invalid_full_name, ["IB"])
        expected_columns = [
            "full_name",
            "PV",
            "PV_stddev",
            "SST",
            "SST_stddev",
            "VIP",
            "VIP_stddev",
        ]
        npt.assert_array_equal(dataframe.columns, expected_columns)
        assert len(dataframe.index) == 824
        assert len({"grey", "CH", "CTX", "IB", "SF", "TRS"} - set(dataframe.index)) == 0
        assert (
            len(
                {"Whole brain", "Cerebrum", "Cerebral cortex", "Olfactory areas"}
                - set(dataframe["full_name"])
            )
            == 0
        )
        assert not np.any(np.isnan(dataframe["SST"]))
        assert not np.any(np.isnan(dataframe["SST_stddev"]))
        assert not np.any(np.isnan(dataframe["VIP"]))
        assert not np.any(np.isnan(dataframe["VIP_stddev"]))
        assert np.count_nonzero(np.isnan(dataframe["PV"])) == 15
        assert np.count_nonzero(np.isnan(dataframe["PV_stddev"])) == 15

        for column in expected_columns[1:]:
            mask = ~np.isnan(dataframe[column])
            assert np.all(dataframe[column].to_numpy()[mask] >= 0.0)

        assert np.allclose(dataframe["PV"][0], (6000.52758114279 + 5831.1654080927) / 2.0)
        assert np.allclose(dataframe["SST"][1], (5083.93779315662 + 5422.81140933669) / 2.0)
        assert np.allclose(dataframe["VIP"][2], (1935.39495313035 + 2013.81692802911) / 2.0)


def check_column_names(dataframe, has_source: bool = True):
    columns = set(dataframe.columns)
    expected = set(REGULAR_COLUMNS.keys())
    if has_source:
        expected |= set(SOURCE_COLUMNS)

    assert columns == expected


def check_columns_na(dataframe):
    assert np.all(~dataframe["brain_region"].isna())
    assert np.all(~dataframe["measurement"].isna())
    assert np.all(~dataframe["cell_type"].isna())
    assert np.all(~dataframe["measurement_type"].isna())


def check_column_types(dataframe, has_source: bool = True) -> bool:
    """
    Check the expected dtype of each column.
    """
    for column, type_ in REGULAR_COLUMNS.items():
        assert dataframe[column].dtype == type_

    if has_source:
        for column, type_ in SOURCE_COLUMNS.items():
            assert dataframe[column].dtype == type_


def get_invalid_region_names(dataframe):
    return set(dataframe["brain_region"]) - AIBS_REGIONS


def check_non_negative_values(dataframe):
    for column in ["measurement", "standard_deviation"]:
        mask = ~dataframe[column].isna()
        assert np.all(dataframe.loc[mask, column] >= 0.0)


def test_read_kim_et_al_neuron_densities():
    warnings.simplefilter("ignore")
    dataframe = tested.read_kim_et_al_neuron_densities(
        REGION_MAP, Path(MEASUREMENTS_PATH, "mmc3.xlsx")
    )

    check_column_names(dataframe)
    check_column_types(dataframe)
    check_columns_na(dataframe)
    check_non_negative_values(dataframe)

    unhandled_invalid_names = get_invalid_region_names(dataframe)
    assert unhandled_invalid_names == {
        "Nucleus of reunions",
        "Nucleus accumbens, core",
        "Nucleus accumbens, shell",
        "Kolliker-Fuse subnucleus",
        "Periaqueductal gray, dorsal lateral",
        "Periaqueductal gray, dorsal medial",
        "Periaqueductal gray, lateral",
        "Periaqueductal gray, ventral lateral",
    }

    pv_count = np.count_nonzero(dataframe["cell_type"] == "PV+")
    sst_count = np.count_nonzero(dataframe["cell_type"] == "SST+")
    vip_count = np.count_nonzero(dataframe["cell_type"] == "VIP+")

    # 849 = 824 - 15 + 40:
    # * 824 initial region names
    # * 15 rows with 'N/D' measurement values for PV are removed
    # * 40 additional region names due to layer 6 -> layer 6a, layer 6b expansions
    assert pv_count == 849
    # 864 = 824 + 40
    assert sst_count == 864
    assert vip_count == 864
    assert len(dataframe.index) == 849 + 2 * 864

    for value in ["measurement", "standard_deviation"]:
        assert np.all(~dataframe[value].isna())


def test_read_inhibitory_neuron_measurement_compilation():
    dataframe = tested.read_inhibitory_neuron_measurement_compilation(
        Path(MEASUREMENTS_PATH, "gaba_papers.xlsx")
    )
    check_column_names(dataframe, has_source=True)
    check_column_types(dataframe, has_source=True)

    assert get_invalid_region_names(dataframe) == {
        "Somatosensory cortex",
        "Prelimbic area, layer 4",
    }

    check_columns_na(dataframe)
    check_non_negative_values(dataframe)

    assert np.count_nonzero(dataframe["measurement"].isna()) == 0
    assert np.count_nonzero(dataframe["standard_deviation"].isna()) == 10

    assert np.all(dataframe["cell_type"] == "inhibitory neuron")
    mask = dataframe["brain_region"] == "Field CA1"
    assert set(dataframe["source_title"][mask]) == {
        "Quantitative analysis of GABAergic neurons in the mouse hippocampus, with optical "
        "disector using confocal laser scanning microscope",
        "Selective populations of hippocampal interneurons express ErbB4 and their number and "
        "distribution is altered in ErbB4 knockout mice",
        "GABA concentration and GABAergic neuron populations in limbic areas are differentially "
        "altered by brain serotonin deficiency in Tph2 knockout mice",
    }

    assert len(set(dataframe["source_title"])) == 21
    assert len(set(dataframe["comment"])) == 25


def test_read_pv_sst_vip_measurement_compilation():
    dataframe = tested.read_pv_sst_vip_measurement_compilation(
        Path(MEASUREMENTS_PATH, "gaba_papers.xlsx")
    )
    check_column_names(dataframe, has_source=True)
    check_column_types(dataframe, has_source=True)

    invalid_region_names = get_invalid_region_names(dataframe)
    assert invalid_region_names == {
        "Somatosensory cortex",
        "Superior Colliculus",
        "Prelimbic area, layer 4",
        "Medulla, unassigned",
        "Midbrain, motor related, other",
    }

    check_columns_na(dataframe)
    check_non_negative_values(dataframe)

    assert np.count_nonzero(dataframe["measurement"].isna()) == 0
    assert np.count_nonzero(dataframe["standard_deviation"].isna()) == 0

    mask = dataframe["brain_region"] == "Field CA1"
    assert set(dataframe["source_title"][mask]) == {
        "Age‐dependent loss of parvalbumin‐expressing hippocampal interneurons in mice deficient"
        " in CHL1, a mental retardation and schizophrenia susceptibility gene",
        "Cellular architecture of the mouse hippocampus: A quantitative aspect of chemically "
        "defined GABAergic neurons with stereology",
        "Comparative density of CCK- and PV-GABA cells within the cortex and hippocampus",
        "Deletion of Selenoprotein M Leads to Obesity without Cognitive Deficits",
        "Densities and numbers of calbindin and parvalbumin positive neurons across the rat and "
        "mouse brain",
        "Early neuropathology of somatostatin/NPY GABAergic cells in thehippocampus of a PS1×APP"
        " transgenic model of Alzheimer’s disease",
        "GABA concentration and GABAergic neuron populations in limbic areas are differentially "
        "altered by brain serotonin deficiency in Tph2 knockout mice",
        "Selective populations of hippocampal interneurons express ErbB4 and their number and "
        "distribution is altered in ErbB4 knockout mice",
    }

    mask = dataframe["brain_region"] == "Prelimbic area"
    assert set(dataframe["specimen_age"][mask]) == {
        "Adult mice",
        "3–8 months old",
        "P25",
        "6  months  old",
        "3 months",
    }
    assert set(dataframe["cell_type"]) == {"PV+", "SST+", "VIP+"}


def test_read_non_density_measurements():
    dataframe = pd.read_csv(Path(MEASUREMENTS_PATH, "non_density_measurements.csv"))

    check_column_names(dataframe, has_source=True)
    assert np.all(dataframe["measurement_unit"].isna())
    dataframe["measurement_unit"] = "None"
    check_column_types(dataframe, has_source=True)
    check_columns_na(dataframe)
    check_non_negative_values(dataframe)

    assert np.count_nonzero(dataframe["measurement"].isna()) == 0
    assert np.count_nonzero(dataframe["standard_deviation"].isna()) == 9
    assert np.count_nonzero(dataframe["specimen_age"].isna()) == 2

    mask = dataframe["measurement_type"] != "cell count per slice"
    assert np.all(dataframe["measurement"][mask] <= 1.0)  # only proportions are expected

    inhibitory_neuron_count = np.count_nonzero(dataframe["cell_type"] == "inhibitory neuron")
    pv_count = np.count_nonzero(dataframe["cell_type"] == "PV+")
    sst_count = np.count_nonzero(dataframe["cell_type"] == "SST+")
    vip_count = np.count_nonzero(dataframe["cell_type"] == "VIP+")
    assert inhibitory_neuron_count == 12
    assert pv_count == 9
    assert sst_count == 10
    assert vip_count == 2
    assert len(dataframe.index) == 33


def test_read_measurements():
    warnings.simplefilter("ignore")
    dataframe = tested.read_measurements(
        REGION_MAP,
        Path(MEASUREMENTS_PATH, "mmc3.xlsx"),
        Path(MEASUREMENTS_PATH, "gaba_papers.xlsx"),
        Path(MEASUREMENTS_PATH, "non_density_measurements.csv"),
    )
    check_column_names(dataframe)
    check_non_negative_values(dataframe)
    check_columns_na(dataframe)

    assert set(dataframe["cell_type"]) == {"PV+", "SST+", "VIP+", "inhibitory neuron"}
    assert set(dataframe["measurement_type"]) == {
        "cell density",
        "neuron proportion",
        "cell proportion",
        "cell count per slice",
    }
    mask = (dataframe["measurement_type"] == "neuron proportion") | (
        dataframe["measurement_type"] == "cell proportion"
    )
    assert np.all(dataframe["measurement"][mask] <= 1.0)

    # TODO: one duplicate title because of a trailing endline character
    assert len(set(dataframe["source_title"])) == 48
    assert len(set(dataframe["comment"])) >= 48
