"""Utility functions to retrieve cell counts from the scientific literature.

Lexicon: AIBS stands for Allen Institute for Brain Science
    https://alleninstitute.org/what-we-do/brain-science/
"""

from typing import TYPE_CHECKING, Dict, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragme: no cover
    from pathlib import Path

    from voxcell import RegionMap, VoxelData  # type: ignore


def cell_counts() -> Dict[str, int]:
    """
    Cell counts of different region groups of the mouse brain.


    Groups are supposed not to overlap and to cover the entire brain.
    The sum of the group cell counts is the total number of cells in the whole mouse brain.

    Cell counts have been extracted from
        * Table 1 of "Updated Neuronal Scaling Rules for the Brains of Glires (Rodents/Lagomorphs)"
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3237106/ by Herculano-Houzel et al., 2011
        * Table 1 of "Distribution of neurons in functional areas of the mouse cerebral cortex
         reveals quantitatively different cortical zones" by Herculano-Houzel et al., 2013
         https://www.frontiersin.org/articles/10.3389/fnana.2013.00035/full

    Returns:
        dict whose keys are region group names including
            * 'Cerebellum group': Cerebellum (CB) and arbor vitae (arb)
            * 'Isocortex group': Isocortex plus the Entorhinal (ENT) and Piriform areas (PIR)
            * 'Rest': rest of the mouse brain
        and whose values are the corresponding cell counts.
    """
    counts = {
        # Table 1 of Herculano-Houzel et al., 2011,
        "Cerebellum group": 42220000 + 6950000,  # 'Cerebellar neurons' + 'Cerebellar other cells'
        # Table 1 of Herculano-Houzel et al., 2013, Line 'Total'
        "Isocortex group": 2 * (5048837 + 6640234),  # 'neurons' + 'other cells', 2 hemispheres
    }
    # Table 1 of Herculano-Houzel et al., 2011,
    # ('Brain neurons' + 'Brain other cells') + ('Olf bulb neurons' + 'Olf other cells')
    total = (67870000 + 33860000) + (3890000 + 5460000)
    counts["Rest"] = total - sum(counts.values())

    return counts


def neuron_counts() -> Dict[str, int]:
    """
    Neuron counts of different region groups of the mouse brain.


    Groups are supposed not to overlap and to cover the entire brain.
    The sum of the group cell counts is the total number of cells in the whole mouse brain.

    Neuron counts have been extracted from
        * Table 1 of "Updated Neuronal Scaling Rules for the Brains of Glires (Rodents/Lagomorphs)"
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3237106/ by Herculano-Houzel et al., 2011
        * Table 1 of "Distribution of neurons in functional areas of the mouse cerebral cortex
         reveals quantitatively different cortical zones" by Herculano-Houzel et al., 2013
         https://www.frontiersin.org/articles/10.3389/fnana.2013.00035/full

    Returns:
        dict whose keys are region group names including
            * 'Cerebellum group': Cerebellum (CB) and arbor vitae (arb)
            * 'Isocortex group': Isocortex plus the Entorhinal (ENT) and Piriform areas (PIR)
            * 'Rest': rest of the mouse brain
        and whose values are the corresponding neuron counts.
    """
    counts = {
        # Table 1 of Herculano-Houzel et al., 2011,
        "Cerebellum group": 42220000,  # 'Cerebellar neurons'
        # Table 1 of Herculano-Houzel et al., 2013, Line 'Total'
        "Isocortex group": 2 * 5048837,  # 'neurons', 2 hemispheres
    }
    # Table 1 of Herculano-Houzel et al., 2011,
    # ('Brain neurons' + 'Brain other cells') + ('Olf bulb neurons' + 'Olf other cells')
    total = 67870000 + 3890000
    counts["Rest"] = total - sum(counts.values())

    return counts


def glia_cell_counts() -> Dict[str, int]:
    """Number of glial cells in the whole mouse brain.

    Glia cell counts have been extracted from
        * Table 1 of "Updated Neuronal Scaling Rules for the Brains of Glires (Rodents/Lagomorphs)"
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3237106/ by Herculano-Houzel et al., 2011
        * Table 1 of "Distribution of neurons in functional areas of the mouse cerebral cortex
         reveals quantitatively different cortical zones" by Herculano-Houzel et al., 2013
         https://www.frontiersin.org/articles/10.3389/fnana.2013.00035/full

    Returns:
        dict whose keys are region group names including
            * 'Cerebellum group': Cerebellum (CB) and
            * 'Isocortex group': Isocortex plus the Entorhinal (ENT) and Piriform areas (PIR)
            * 'Rest': rest of the mouse brain
        and whose values are the corresponding glia cell counts.
    """
    return {group: cell_counts()[group] - neuron_counts()[group] for group in cell_counts()}


def inhibitory_neuron_counts(
    inhibitory_neurons_dataframe: pd.DataFrame,
) -> Dict[str, int]:
    """Number of inhibitory neurons in the whole mouse brain.

    Inhibitory neuron counts have been extracted from the suplementary materials of
        'Brain-wide Maps Reveal Stereotyped Cell-Type-Based Cortical Architecture '
        'and Subcortical Sexual Dimorphism' by Kim et al., 2017.
        https://ars.els-cdn.com/content/image/1-s2.0-S0092867417310693-mmc1.xlsx

    Returns:
        dict whose keys are region group names including
            * 'Cerebellum group': Cerebellum (CB) and arbor vitae (arb)
            * 'Isocortex group': Isocortex plus the Entorhinal (ENT) and Piriform areas (PIR)
            * 'Rest': rest of the mouse brain
        and whose values are the corresponding inhibitory neuron cell counts (int).
    """
    cerebellum_group_count = np.sum(inhibitory_neurons_dataframe.loc["CB"][1:])
    isocortex_group_count = sum(
        np.sum(inhibitory_neurons_dataframe.loc[acronym][1:])
        for acronym in ["Isocortex", "ENT", "PIR"]
    )
    rest_count = (
        np.sum(inhibitory_neurons_dataframe.loc["grey"][1:])
        - cerebellum_group_count
        - isocortex_group_count
    )
    return {
        "Cerebellum group": round(cerebellum_group_count),
        "Isocortex group": round(isocortex_group_count),
        "Rest": round(rest_count),
    }


def inhibitory_data(
    inhibitory_neurons_dataframe: pd.DataFrame,
) -> Dict[str, Union[int, Dict[str, float]]]:
    """
    Number of inhibitory cells for different region groups of the mouse brain.

    Groups are supposed not to overlap and to cover the entire brain.
    The sum of the group cell counts is the total number of inhibitory cells in the whole mouse
     brain.

    Taken from the suplementary materials of
    'Brain-wide Maps Reveal Stereotyped Cell-Type-Based Cortical Architecture '
    'and Subcortical Sexual Dimorphism' by Kim et al., 2017.
    https://ars.els-cdn.com/content/image/1-s2.0-S0092867417310693-mmc1.xlsx

    Args:
        inhibitory_neurons_dataframe: pandas.DataFrame of the form
                                                fullName          PV        SST        VIP
            ROI
            grey                             Whole brain  2.8914e+06  2357781.0  451561.50
            CH                                  Cerebrum      660590  1383927.6  427575.90
            CTX                          Cerebral cortex      627281  1068325.2  424446.60
            CTXpl                         Cortical plate      619825   999929.7  414437.40
            Isocortex                          Isocortex      533366   594769.8  280774.05
            ...                                      ...         ...        ...        ...
            The columns of PV, SST and VIP contain the counts of the corresponding immunoreactive
            cells for each AIBS acronym listed in the DataFrame index.

    Returns:
        dict of the form
        {
            'proportions': {
                'Cerebellum group': <float value>,
                'Isocortex group': <float value>,
                'Rest': <float value>
            }
            'neuron_count': <int value>
        }
        where the value corresponding to each group is a float in (0, 1) indicating the proportion
        of inhibitory cells among neurons in that group and the value corresponding to
        'neuron_count' is the number of inhibitory neurons in the whole mouse brain.
    """

    return {
        "proportions": {
            "Cerebellum group": inhibitory_neuron_counts(inhibitory_neurons_dataframe)[
                "Cerebellum group"
            ]
            / neuron_counts()["Cerebellum group"],
            "Isocortex group": inhibitory_neuron_counts(inhibitory_neurons_dataframe)[
                "Isocortex group"
            ]
            / neuron_counts()["Isocortex group"],
            "Rest": inhibitory_neuron_counts(inhibitory_neurons_dataframe)["Rest"]
            / neuron_counts()["Rest"],
        },
        "neuron_count": round(sum(inhibitory_neuron_counts(inhibitory_neurons_dataframe).values())),
    }


def extract_inhibitory_neurons_dataframe(
    inhibitory_neuron_counts_path: Union[str, "Path"]
) -> pd.DataFrame:
    """
    Extract from excel file a pandas.DataFrame containing the counts of the cells reacting to
    PV, SST and VIP in every AIBS region of the mouse brain.

    Note: The following markers have been used by Kim et al., in
    'Brain-wide Maps Reveal Stereotyped Cell-Type-Based Cortical Architecture and '
    'Subcortical Sexual Dimorphism' to detect inhibitory neurons:
    parvalbumin (PV), somatostatin (SST), and vasoactive intestinal peptide (VIP).
    PV, SST and VIP are only expressed in neurons.
    The cell populations reacting respectively to PV, SST and VIP are not overlapping,
    i.e., there is no cell in the brain that is co-expressing a combination of these markers.
    This assumption is supported in the Cortex and Hippocampus by transcriptomic studies.

    Args:
        inhibitory_neuron_counts_path: path to the excel document mm1c.xls of the supplementary
            materials of 'Brain-wide Maps Reveal Stereotyped Cell-Type-Based Cortical Architecture
            and Subcortical Sexual Dimorphism' by Kim et al., 2017.
            https://ars.els-cdn.com/content/image/1-s2.0-S0092867417310693-mmc1.xlsx

    Returns: pandas.DataFrame of the form
                                            fullName          PV        SST        VIP
        ROI
        grey                             Whole brain  2.8914e+06  2357781.0  451561.50
        CH                                  Cerebrum      660590  1383927.6  427575.90
        CTX                          Cerebral cortex      627281  1068325.2  424446.60
        CTXpl                         Cortical plate      619825   999929.7  414437.40
        Isocortex                          Isocortex      533366   594769.8  280774.05
        ...                                      ...         ...        ...        ...
        The columns of PV, SST and VIP contain the counts of the corresponding immunoreactive
        cells for each AIBS acronym listed in the DataFrame index.

    """
    return pd.read_excel(
        str(inhibitory_neuron_counts_path),
        sheet_name="count",
        header=None,
        names=["ROI", "fullName", "PV", "SST", "VIP"],
        usecols="A,B,D,F,H",
        skiprows=[0, 1],
        engine="openpyxl",
    ).set_index("ROI")
