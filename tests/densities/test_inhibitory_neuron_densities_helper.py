"""
Unit tests for the inhibitory densities helper functions
"""

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import atlas_densities.densities.inhibitory_neuron_densities_helper as tested
from atlas_densities.exceptions import AtlasDensitiesError


def test_average_densities_to_cell_counts():
    average_densities = pd.DataFrame(
        {
            "pv+": [1.0, 0.0],
            "pv+_standard_deviation": [1.0, 0.0],
            "sst+": [np.nan, 1.0],
            "sst+_standard_deviation": [1.0, 1.0],
        },
        index=["A", "B"],
    )

    region_volumes = pd.DataFrame({"brain_region": ["A", "B"], "volume": [1.0, 2.0]}, index=[6, 66])

    expected = pd.DataFrame(
        {
            "pv+": [1.0, 0.0],
            "pv+_standard_deviation": [1.0, 0.0],
            "sst+": [np.nan, 2.0],
            "sst+_standard_deviation": [1.0, 2.0],
        },
        index=["A", "B"],
    )

    actual = tested.average_densities_to_cell_counts(average_densities, region_volumes)
    pdt.assert_frame_equal(actual, expected)


def test_average_densities_to_cell_counts_2():
    # Same test as before but with different numerical values.
    average_densities = pd.DataFrame(
        {
            "pv+": [1.0, 2.0],
            "pv+_standard_deviation": [0.1, 0.2],
            "sst+": [3.0, 4.0],
            "sst+_standard_deviation": [0.3, 0.4],
        },
        index=["A", "B"],
    )
    region_volumes = pd.DataFrame(
        {"volume": [10.0, 20.0], "brain_region": ["A", "B"]}, index=[1, 2]
    )
    actual = tested.average_densities_to_cell_counts(average_densities, region_volumes)
    pdt.assert_frame_equal(
        actual,
        pd.DataFrame(
            {
                "pv+": [10.0, 40.0],
                "pv+_standard_deviation": [1.0, 4.0],
                "sst+": [30.0, 80.0],
                "sst+_standard_deviation": [3.0, 8.0],
            },
            index=["A", "B"],
        ),
    )


def test_average_densities_to_cell_counts_with_reverse_index_order():
    # Same test as before but with the index of `average_densities`
    # in reverse order. We check that the `tested.average_densities_to_cell_counts`
    # does not sort or change in any way the index order.
    average_densities = pd.DataFrame(
        {
            "pv+": [1.0, 2.0],
            "pv+_standard_deviation": [0.1, 0.2],
            "sst+": [3.0, 4.0],
            "sst+_standard_deviation": [0.3, 0.4],
        },
        index=["B", "A"],  # Inverse lexico-graphical order
    )
    region_volumes = pd.DataFrame(
        {"volume": [10.0, 20.0], "brain_region": ["B", "A"]}, index=[1, 2]
    )
    actual = tested.average_densities_to_cell_counts(average_densities, region_volumes)
    pdt.assert_frame_equal(
        actual,
        pd.DataFrame(
            {
                "pv+": [10.0, 40.0],
                "pv+_standard_deviation": [1.0, 4.0],
                "sst+": [30.0, 80.0],
                "sst+_standard_deviation": [3.0, 8.0],
            },
            index=["B", "A"],
        ),
    )


def test_resize_average_densities():
    average_densities = pd.DataFrame(
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
    hierarchy_info = pd.DataFrame(
        {"brain_region": ["A", "B", "C"], "descendant_id_set": [{1}, {2}, {1, 2, 3}]},
        index=[1, 2, 3],
    )

    expected = pd.DataFrame(
        {
            "gad67+": [9.0, 12.0, np.nan],
            "gad67+_standard_deviation": [0.9, 1.2, np.nan],
            "pv+": [1.0, 2.0, np.nan],
            "pv+_standard_deviation": [0.1, 0.2, np.nan],
            "sst+": [3.0, 4.0, np.nan],
            "sst+_standard_deviation": [0.3, 0.4, np.nan],
            "vip+": [5.0, 6.0, np.nan],
            "vip+_standard_deviation": [0.5, 0.6, np.nan],
        },
        index=hierarchy_info["brain_region"],
    )
    actual = tested.resize_average_densities(average_densities, hierarchy_info)
    pdt.assert_frame_equal(actual, expected)


def test_get_cell_types():
    average_densities = pd.DataFrame(
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

    assert tested.get_cell_types(average_densities) == ["gad67+", "pv+", "sst+", "vip+"]


def test_check_region_counts_consistency():
    region_counts = pd.DataFrame(
        {
            "gad67+": [9.0, 12.0, 2.0],
            "gad67+_standard_deviation": [0.9, 1.2, 1.0],
            "pv+": [1.0, 1.0, 1.0],
            "pv+_standard_deviation": [0.1, 0.2, 0.3],
            "sst+": [3.0, 4.0, 5.0],
            "sst+_standard_deviation": [0.3, 0.4, 0.5],
            "vip+": [5.0, 6.0, 7.0],
            "vip+_standard_deviation": [0.5, 0.6, 0.7],
        },
        index=["A", "B", "C"],
    )
    hierarchy_info = pd.DataFrame(
        {"brain_region": ["A", "B", "C"], "descendant_id_set": [{1, 2, 3}, {2}, {3}]},
        index=[1, 2, 3],
    )

    # Uncertain or consistent
    tested.check_region_counts_consistency(region_counts, hierarchy_info)
    region_counts.loc["A", "gad67+_standard_deviation"] = 0.0
    tested.check_region_counts_consistency(region_counts, hierarchy_info)

    # Inconsistent parent-child count inequality
    region_counts.loc["B", "gad67+_standard_deviation"] = 0.0
    with pytest.raises(AtlasDensitiesError, match=r"gad67\+.*exceed"):
        tested.check_region_counts_consistency(region_counts, hierarchy_info)

    # The sum of the cell counts of the children regions B and C exceeds the count
    # of the parent region A.
    region_counts.loc["B", "gad67+_standard_deviation"] = 1.2  # restores valid value
    region_counts.loc["A", "pv+_standard_deviation"] = 0.0
    region_counts.loc["B", "pv+_standard_deviation"] = 0.0
    region_counts.loc["C", "pv+_standard_deviation"] = 0.0
    with pytest.raises(AtlasDensitiesError, match=r"sum of the counts.*pv\+.*exceeds.*region A"):
        tested.check_region_counts_consistency(region_counts, hierarchy_info)


def test_replace_inf_with_none():
    bounds = np.array([[0.1, 1.0], [-1.0, np.inf]])
    result = [(0.1, 1.0), (-1.0, None)]

    assert repr(result) == repr(tested.replace_inf_with_none(bounds))
