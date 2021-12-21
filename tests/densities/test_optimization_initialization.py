"""
Unit tests for inhibitory cell density computation
(linear program minimizing distances to initial estimates)
"""

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest

import atlas_densities.densities.inhibitory_neuron_densities_optimization as tested
from atlas_densities.exceptions import AtlasDensitiesError


@pytest.fixture
def counts_data_1():
    hierarchy_info = pd.DataFrame(
        {"brain_region": ["A", "B"], "descendant_id_set": [{1}, {2}]}, index=[1, 2]
    )
    region_counts = pd.DataFrame(
        {
            "gad67+": [9.0, 12.0],
            "gad67+_standard_deviation": [0.9, 1.2],
            "pv+": [1.0, 2.0],
            "pv+_standard_deviation": [0.1, 0.2],
            "sst+": [3.0, 4.0],
            "sst+_standard_deviation": [0.3, 0.4],
            "vip+": [5.0, 6.0],
            "vip+_standard_deviation": [0.5, 0.6],
        },
        index=["A", "B"],
    )
    id_counts = pd.DataFrame(
        {
            "gad67+": [9.0, 12.0],
            "gad67+_standard_deviation": [0.0, 1.2],
            "pv+": [1.0, 2.0],
            "pv+_standard_deviation": [0.1, 0.0],
            "sst+": [3.0, 4.0],
            "sst+_standard_deviation": [0.3, 0.4],
            "vip+": [5.0, 6.0],
            "vip+_standard_deviation": [0.5, 0.6],
        },
        index=[1, 2],
    )

    return {
        "region_counts": region_counts,
        "id_counts": id_counts,
        "hierarchy_info": hierarchy_info,
    }


def test_set_known_values_1(counts_data_1):
    # Two disjoint regions A and B
    x_result, deltas = tested.set_known_values(
        counts_data_1["region_counts"], counts_data_1["id_counts"], counts_data_1["hierarchy_info"]
    )

    pdt.assert_frame_equal(
        x_result,
        pd.DataFrame(
            {
                "gad67+": [np.nan] * 2,
                "pv+": [np.nan] * 2,
                "sst+": [np.nan] * 2,
                "vip+": [np.nan] * 2,
            },
            index=[1, 2],
        ),
    )

    pdt.assert_frame_equal(
        deltas,
        pd.DataFrame(
            {
                "gad67+": [np.nan] * 2,
                "pv+": [np.nan] * 2,
                "sst+": [np.nan] * 2,
                "vip+": [np.nan] * 2,
            },
            index=["A", "B"],
        ),
    )


def test_set_known_values_2(counts_data_1):
    # Some of the standard deviations associated to cell counts are zero
    # The corresponding entries in `x_result` must be set with the estimates of `average_densities`.
    # The corresponding entries in `deltas` must be set with 0.0.
    counts_data_1["region_counts"].loc["A", "gad67+_standard_deviation"] = 0.0
    counts_data_1["region_counts"].loc["B", "pv+_standard_deviation"] = 0.0
    x_result, deltas = tested.set_known_values(
        counts_data_1["region_counts"], counts_data_1["id_counts"], counts_data_1["hierarchy_info"]
    )

    pdt.assert_frame_equal(
        x_result,
        pd.DataFrame(
            {
                "gad67+": [9.0, np.nan],
                "pv+": [np.nan, 2.0],
                "sst+": [np.nan] * 2,
                "vip+": [np.nan] * 2,
            },
            index=[1, 2],
        ),
    )

    pdt.assert_frame_equal(
        deltas,
        pd.DataFrame(
            {
                "gad67+": [0.0, np.nan],
                "pv+": [np.nan, 0.0],
                "sst+": [np.nan] * 2,
                "vip+": [np.nan] * 2,
            },
            index=["A", "B"],
        ),
    )


def test_set_known_values_3(counts_data_1):
    # Some std deviations of the region cell counts are np.nan, which means that the estimates are missing.
    # The corresponding entries in `deltas` must be set with 0.0.
    counts_data_1["region_counts"].loc["A", "gad67+_standard_deviation"] = np.nan
    counts_data_1["region_counts"].loc["B", "pv+_standard_deviation"] = np.nan
    x_result, deltas = tested.set_known_values(
        counts_data_1["region_counts"], counts_data_1["id_counts"], counts_data_1["hierarchy_info"]
    )

    pdt.assert_frame_equal(
        x_result,
        pd.DataFrame(
            {
                "gad67+": [np.nan] * 2,
                "pv+": [np.nan] * 2,
                "sst+": [np.nan] * 2,
                "vip+": [np.nan] * 2,
            },
            index=[1, 2],
        ),
    )

    pdt.assert_frame_equal(
        deltas,
        pd.DataFrame(
            {
                "gad67+": [np.inf, np.nan],
                "pv+": [np.nan, np.inf],
                "sst+": [np.nan] * 2,
                "vip+": [np.nan] * 2,
            },
            index=["A", "B"],
        ),
    )


@pytest.fixture
def counts_data_2():
    hierarchy_info = pd.DataFrame(
        # A is now a parent region of B
        {"brain_region": ["A", "B"], "descendant_id_set": [{1, 2}, {2}]},
        index=[1, 2],
    )
    region_counts = pd.DataFrame(
        {
            "gad67+": [0.0, 1.0],  # zero cell count for A
            "gad67+_standard_deviation": [0.0, 1.2],  # zero standard deviation for A
            "pv+": [2.0, 1.0],
            "pv+_standard_deviation": [0.2, 0.1],
            "sst+": [4.0, 3.0],
            "sst+_standard_deviation": [0.4, 0.3],
            "vip+": [6.0, 5.0],
            "vip+_standard_deviation": [0.6, 0.5],
        },
        index=["A", "B"],
    )
    id_counts = pd.DataFrame(
        {
            "gad67+": [0.0, 1.0],
            "gad67+_standard_deviation": [0.0, 1.2],
            "pv+": [1.0, 1.0],
            "pv+_standard_deviation": [0.1, 0.0],
            "sst+": [1.0, 3.0],
            "sst+_standard_deviation": [0.1, 0.3],
            "vip+": [1.0, 5.0],
            "vip+_standard_deviation": [0.1, 0.5],
        },
        index=[1, 2],
    )

    return {
        "region_counts": region_counts,
        "id_counts": id_counts,
        "hierarchy_info": hierarchy_info,
    }


def test_set_known_values_exception(counts_data_2):
    # The region A is the parent of B.
    # Non-zero estimate is given with certainty for pv+ in B whereas the A has a zero estimate
    # given also for certain.
    counts_data_2["region_counts"].at["B", "pv+_standard_deviation"] = 0.0
    with pytest.raises(AtlasDensitiesError, match=".*pv\+.*non-zero estimate.*ancestor region"):
        tested.set_known_values(
            counts_data_2["region_counts"],
            counts_data_2["id_counts"],
            counts_data_2["hierarchy_info"],
        )


def test_set_known_values_4(counts_data_2):
    # The region A is the parent of B.
    # Some of the standard deviations associated to gad67+ counts are zero and the gad67+ counts are zero too.
    # The corresponding entries in `x_result` must be set with 0.0 but the cell counts of descendants must be set
    # as well The corresponding entries in `deltas` must be set with 0.0.
    x_result, deltas = tested.set_known_values(
        counts_data_2["region_counts"], counts_data_2["id_counts"], counts_data_2["hierarchy_info"]
    )

    pdt.assert_frame_equal(
        x_result,
        pd.DataFrame(
            {
                "gad67+": [0.0, 0.0],
                "pv+": [0.0, 0.0],
                "sst+": [0.0, 0.0],
                "vip+": [0.0, 0.0],
            },
            index=[1, 2],
        ),
    )

    pdt.assert_frame_equal(
        deltas,
        pd.DataFrame(
            {
                "gad67+": [0.0, 0.0],
                "pv+": [0.0, 0.0],
                "sst+": [0.0, 0.0],
                "vip+": [0.0, 0.0],
            },
            index=["A", "B"],
        ),
    )


def test_set_known_values_5(counts_data_2):
    # Some std deviations of region cell counts are NaNs, meaning that the corresponding estimates are missing.
    # The corresponding entries in `deltas` must be set with np.inf.
    counts_data_2["region_counts"].loc["A", "gad67+_standard_deviation"] = np.nan
    x_result, deltas = tested.set_known_values(
        counts_data_2["region_counts"], counts_data_2["id_counts"], counts_data_2["hierarchy_info"]
    )

    pdt.assert_frame_equal(
        x_result,
        pd.DataFrame(
            {
                "gad67+": [np.nan] * 2,
                "pv+": [np.nan] * 2,
                "sst+": [np.nan] * 2,
                "vip+": [np.nan] * 2,
            },
            index=[1, 2],
        ),
    )

    pdt.assert_frame_equal(
        deltas,
        pd.DataFrame(
            {
                "gad67+": [np.inf, np.nan],  # standard deviation is NaN for A
                "pv+": [np.nan] * 2,
                "sst+": [np.nan] * 2,
                "vip+": [np.nan] * 2,
            },
            index=["A", "B"],
        ),
    )


def test_set_known_values_6(counts_data_2):
    # Some std deviations of region cell counts are NaNs, implying that the corresponding estimates are missing.
    # The corresponding entries in `deltas` must be set with np.inf.
    # Besides, one of the standard deviations associated to the overall gad67+ count is zero and a gad67+
    # leaf id count is zero too. The corresponding entry in `x_result` must be set with 0.0.

    counts_data_2["region_counts"].loc["A", "gad67+"] = 1.0
    # The std deviation below is NaN: the corresponding entries in `deltas` must be set with np.inf.
    counts_data_2["region_counts"].loc["A", "pv+_standard_deviation"] = np.nan
    x_result, deltas = tested.set_known_values(
        counts_data_2["region_counts"], counts_data_2["id_counts"], counts_data_2["hierarchy_info"]
    )

    pdt.assert_frame_equal(
        x_result,
        pd.DataFrame(
            {
                "gad67+": [
                    0.0,
                    np.nan,
                ],  # standard deviation is 0.0 for A and the id cell count is zero for id = 1
                "pv+": [np.nan] * 2,
                "sst+": [np.nan] * 2,
                "vip+": [np.nan] * 2,
            },
            index=[1, 2],
        ),
    )

    pdt.assert_frame_equal(
        deltas,
        pd.DataFrame(
            {
                "gad67+": [0.0, np.nan],  # standard deviation is 0.0 for A
                "pv+": [np.inf, np.nan],
                "sst+": [np.nan] * 2,
                "vip+": [np.nan] * 2,
            },
            index=["A", "B"],
        ),
    )


def neuron_counts():
    return pd.DataFrame(
        {"cell_count": [1.0, 3.0, 3.0], "brain_region": ["A", "B", "C"]},
        index=[1, 2, 3],
    )


def counts_data_3():
    hierarchy_info = pd.DataFrame(
        {"brain_region": ["A", "B", "C"], "descendant_id_set": [{1, 2, 3}, {2}, {3}]},
        index=[1, 2, 3],
    )
    region_counts = pd.DataFrame(
        {
            "gad67+": [5.5, 3.0, 2.0],
            "gad67+_standard_deviation": [1.0, 0.1, 0.1],
            "pv+": [2.0, 1.0, 1.0],
            "pv+_standard_deviation": [0.1, 0.1, 0.0],
            "sst+": [2.0, 1.0, 1.0],
            "sst+_standard_deviation": [0.0, 0.0, 0.0],
            "vip+": [1.0, 1.0, 0.0],
            "vip+_standard_deviation": [0.1, 0.1, 0.1],
        },
        index=["A", "B", "C"],
    )

    id_counts = pd.DataFrame(
        {
            "gad67+": [1.0, 3.0, 2.0],
            "gad67+_standard_deviation": [1.0, 0.1, 0.1],
            "pv+": [1.0, 1.0, 1.0],
            "pv+_standard_deviation": [0.1, 0.1, 0.0],
            "sst+": [0.0, 1.0, 1.0],
            "sst+_standard_deviation": [0.0, 0.0, 0.0],
            "vip+": [0.0, 1.0, 0.0],
            "vip+_standard_deviation": [0.1, 0.1, 0.1],
        },
        index=[1, 2, 3],
    )

    return {
        "region_counts": region_counts,
        "id_counts": id_counts,
        "hierarchy_info": hierarchy_info,
        "neuron_counts": neuron_counts(),
    }


def expected_x_result():
    return pd.DataFrame(
        {
            "gad67+": [np.nan] * 3,
            "pv+": [np.nan, np.nan, 1.0],
            "sst+": [0.0, 1.0, 1.0],
            "vip+": [np.nan] * 3,
        },
        index=[1, 2, 3],
    )


def expected_deltas():
    return pd.DataFrame(
        {
            "gad67+": [np.nan] * 3,
            "pv+": [np.nan, np.nan, 0.0],
            "sst+": [0.0] * 3,
            "vip+": [np.nan] * 3,
        },
        index=["A", "B", "C"],
    )


def test_set_known_values_7():
    counts_data = counts_data_3()
    x_result, deltas = tested.set_known_values(
        counts_data["region_counts"], counts_data["id_counts"], counts_data["hierarchy_info"]
    )

    pdt.assert_frame_equal(x_result, expected_x_result())
    pdt.assert_frame_equal(deltas, expected_deltas())


@pytest.fixture
def bounds_data():
    x_result = pd.DataFrame(
        {
            "gad67+": [0.0, np.nan],
            "pv+": [0.0, 0.0],
            "sst+": [np.nan] * 2,
            "vip+": [np.nan] * 2,
        },
        index=[1, 2],
    )

    deltas = pd.DataFrame(
        {
            "gad67+": [0.0, np.nan],
            "pv+": [0.0, 0.0],
            "sst+": [np.nan] * 2,
            "vip+": [np.nan] * 2,
        },
        index=["A", "B"],
    )

    neuron_counts = pd.DataFrame(
        {"cell_count": [5.0, 1.0], "brain_region": ["A", "B"]},
        index=[1, 2],
    )

    return {"x_result": x_result, "deltas": deltas, "neuron_counts": neuron_counts}


def test_create_bounds(bounds_data):
    bounds, x_map, deltas_map = tested.create_bounds(
        bounds_data["x_result"], bounds_data["deltas"], bounds_data["neuron_counts"]
    )

    expected_x_map = {
        (1, "sst+"): 0,
        (1, "vip+"): 1,
        (2, "gad67+"): 2,
        (2, "sst+"): 3,
        (2, "vip+"): 4,
    }

    assert x_map == expected_x_map

    expected_deltas_map = {
        ("A", "sst+"): 5,
        ("A", "vip+"): 6,
        ("B", "gad67+"): 7,
        ("B", "sst+"): 8,
        ("B", "vip+"): 9,
    }

    assert expected_deltas_map == deltas_map

    expected_bounds = np.array(
        [[0.0] * 10, [5.0, 5.0, 1.0, 1.0, 1.0, np.inf, np.inf, np.inf, np.inf, np.inf]]
    ).T

    npt.assert_array_almost_equal(expected_bounds, bounds)


def test_create_bounds_with_one_less_delta(bounds_data):
    bounds_data["deltas"]["sst+"] = [
        np.inf,
        np.nan,
    ]  # np.inf means that no delta is required for region A and cell type sst+.
    bounds, x_map, deltas_map = tested.create_bounds(
        bounds_data["x_result"], bounds_data["deltas"], bounds_data["neuron_counts"]
    )

    expected_x_map = {
        (1, "sst+"): 0,
        (1, "vip+"): 1,
        (2, "gad67+"): 2,
        (2, "sst+"): 3,
        (2, "vip+"): 4,
    }

    assert x_map == expected_x_map

    expected_deltas_map = {
        ("A", "vip+"): 5,
        ("B", "gad67+"): 6,
        ("B", "sst+"): 7,
        ("B", "vip+"): 8,
    }

    assert expected_deltas_map == deltas_map

    expected_bounds = np.array(
        [[0.0] * 9, [5.0, 5.0, 1.0, 1.0, 1.0, np.inf, np.inf, np.inf, np.inf]]
    ).T

    npt.assert_array_almost_equal(expected_bounds, bounds)


def expected_maps():
    x_map = {
        (1, "gad67+"): 0,
        (1, "pv+"): 1,
        (1, "vip+"): 2,
        (2, "gad67+"): 3,
        (2, "pv+"): 4,
        (2, "vip+"): 5,
        (3, "gad67+"): 6,
        (3, "vip+"): 7,
    }

    deltas_map = {
        ("A", "gad67+"): 8,
        ("A", "pv+"): 9,
        ("A", "vip+"): 10,
        ("B", "gad67+"): 11,
        ("B", "pv+"): 12,
        ("B", "vip+"): 13,
        ("C", "gad67+"): 14,
        ("C", "vip+"): 15,
    }

    return x_map, deltas_map


def test_create_bounds_2():
    bounds, x_map, deltas_map = tested.create_bounds(
        expected_x_result(), expected_deltas(), neuron_counts()
    )

    expected_x_map, expected_deltas_map = expected_maps()

    assert x_map == expected_x_map
    assert expected_deltas_map == deltas_map

    expected_bounds = np.asarray(
        [
            [0.0] * 16,
            [
                1.0,
                1.0,
                1.0,
                3.0,
                3.0,
                3.0,
                3.0,
                3.0,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
            ],
        ],
    ).T

    npt.assert_array_almost_equal(expected_bounds, bounds)


@pytest.fixture
def aub_and_bub_data():
    hierarchy_info = pd.DataFrame(
        {"brain_region": ["A", "B", "C"], "descendant_id_set": [{1, 2, 3}, {2}, {3}]},
        index=[1, 2, 3],
    )
    x_map, deltas_map = expected_maps()

    return {
        "x_result": expected_x_result(),
        "region_counts": counts_data_3()["region_counts"],
        "x_map": x_map,
        "deltas_map": deltas_map,
        "hierarchy_info": hierarchy_info,
    }


def expected_aub_bub():
    A_ub = np.array(
        [
            [
                1.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                1.0,
                0.0,  # (3c, "A", gad67+)
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                -1.0,
                0.0,
                0.0,
                -1.0,
                0.0,
                0.0,
                -1.0,
                0.0,  # (3d, "A", gad67+)
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                1.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,  # (3c, "A", pv+)
                0.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                -1.0,
                0.0,
                0.0,
                -1.0,
                0.0,
                0.0,
                0.0,  # (3d, "A", pv+)
                0.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                1.0,
                0.0,
                1.0,  # (3c, "A", vip+)
                0.0,
                0.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                -1.0,
                0.0,
                0.0,
                -1.0,
                0.0,
                -1.0,  # (3d, "A", vip+)
                0.0,
                0.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                -1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,  # (3e, id = 1)
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,  # (3c, "B", gad67+)
                0.0,
                0.0,
                0.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,  # (3d, "B", gad67+)
                0.0,
                0.0,
                0.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,  # (3c, "B", pv+)
                0.0,
                0.0,
                0.0,
                0.0,
                -1.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                -1.0,
                0.0,
                0.0,
                0.0,  # (3d, "B", pv+)
                0.0,
                0.0,
                0.0,
                0.0,
                -1.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,  # (3c, "B", vip+)
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.0,
                0.0,
                0.0,  # (3d, "B", vip+)
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                -1.0,
                1.0,
                1.0,
                0.0,
                0.0,  # (3e, id = 2)
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,  # (3c, "C", gad67+)
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.0,
                0.0,  # (3d, "C", gad67+)
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,  # (3c, "C", vip+)
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.0,  # (3d, "C", vip+)
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.0,
                1.0,  # (3e, id = 2)
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        ]
    )

    b_ub = np.array(
        [
            5.5,  # (3c, "A", gad67+)
            -5.5,  # (3d, "A", gad67+)
            1.0,  # (3c, "A", pv+)
            -1.0,  # (3d, "A", pv+)
            1.0,  # (3c, "A", vip+)
            -1.0,  # (3d, "A", vip+)
            0.0,  # (3e, id = 1),
            3.0,  # (3c, "B", gad67+)
            -3.0,  # (3d, "B", gad67+)
            1.0,  # (3c, "B", pv+)
            -1.0,  # (3d, "B", pv+)
            1.0,  # (3c, "B", vip+)
            -1.0,  # (3d, "B", vip+)
            -1.0,  # (3e, id = 2)
            2.0,  # (3c, "C", gad67+)
            -2.0,  # (3d, "C", gad67+)
            0.0,  # (3c, "C", vip+)
            0.0,  # (3d, "C", vip+)
            -2.0,  # (3e, id = 3)
        ]
    )

    return A_ub, b_ub


def test_create_aub_and_bub(aub_and_bub_data):
    A_ub, b_ub = tested.create_aub_and_bub(
        aub_and_bub_data["x_result"],
        aub_and_bub_data["region_counts"],
        aub_and_bub_data["x_map"],
        aub_and_bub_data["deltas_map"],
        aub_and_bub_data["hierarchy_info"],
    )

    expected_A_ub, expected_b_ub = expected_aub_bub()

    npt.assert_array_almost_equal(b_ub, expected_b_ub)
    npt.assert_array_almost_equal(A_ub, expected_A_ub)
