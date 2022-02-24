"""
Module performing the combination of several genetic marker datasets based on literature estimates.

This module implements the glia differentiation process of 'A Cell Atlas for the Mouse Brain' by
by C. Eroe et al. 2018,
https://www.frontiersin.org/articles/10.3389/fninf.2018.00084/full.

This consists first in combining the gene expression intensities of the different glia types in
order to get a mean dataset for each type.
The resulting mean datasets of oligodendrocytes, astrocytes and microglia are then averaged together
to produce an overall glia mean dataset.

We work under the assumption that the image intensity of a genetic marker is proportional to the
density of marked cells.

Note: the output glia intensities will be further constrained to derive the final glia densities
in a subsequent step, see cell_density module.

"""
import numpy as np
import pandas as pd
import voxcell
from atlas_commons.typing import AnnotationT


def combine(
    region_map: voxcell.RegionMap,
    annotation_raw: AnnotationT,
    glia_celltype_densities: pd.DataFrame,
    combination_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Average the glia marker intensities based on ratios from the scientific literature.

    Group the respective markers of astrocytes, oligodendrocytes and microglia and average each
    resulting group with respect to its ratio obtained from the scienfitic literature.
    The computed oligodendrocyte, astrocyte and microglia mean arrays are then averaged together
    using other ratios provided for the Cerebellum and the Striatum. The weights used in this last
    step are inferred from literature figures and from the annotation file. These weights are also
     returned.

    Note: the returned intensity arrays can be interpreted in terms of cell densities
    (up to a constant factor).
    At this stage, they do not respect any prescribed cell count from the scientific literature.
    This is why they are subsequently referred to as 'unconstrained cell densities' or simply
    'intensities'.

    Args:
        region_map: object to navigate the brain regions hierarchy.
        annotation_raw: annotated 3D volume of the whole mouse brain.
        glia_celltype_densities: a DataFrame of the following form:
                               cerebellum    striatum
            oligodendrocyte       13750.0      9950.0
                  astrocyte        1512.0      9867.0
                  microglia        8624.0     12100.0
            The rows are the different glia cell types and the columns are
            regions for which the glia type average density is known.
            Densities are expressed in number of cells per mm^3.
            Note: Densities entries come from
            'Cell Densities in the Mouse Brain: A Systematic Review' by D. Keller et al. 2018,
            Tables 4 and 5, https://www.frontiersin.org/articles/10.3389/fnana.2018.00083/full
        combination_data: a DataFrame of the following form
                      cellType     gene  averageExpressionIntensity          volume
               oligodendrocyte      cnp                   35.962800  NDArray[float]
               oligodendrocyte      mbp                    3.304965  NDArray[float]
                     astrocyte     gfap                    3.209790  NDArray[float]
                     astrocyte  aldh1l1                    1.326080  NDArray[float]
                     microglia  tmem119                    1.326080  NDArray[float]

            The `cellType` columns contains the different glia cell types under consideration.
            The `gene` column indicates which gene markers should be combined to get the expression
            intensity of each glia cell type.
            The `averageExpressionIntensity` column contains the average expression intensity of
            each gene marker with respect to each glia cell type under consideration.
            Note: These numbers are the E_marker of the 'Glia differentiation' section in
            'A Cell Atlas for the Mouse Brain' by by C. Eroe et al. 2018,
            https://www.frontiersin.org/articles/10.3389/fninf.2018.00084/full.
            They have been extracted from the Supplementary Materials of
            'Cell types in the mouse cortex and hippocampus revealed by single-cell RNA-seq'
            by Zeisel et al. 2015. Unfortunately, the relevant data are not available anymore.
            See rather the Excell file attachement of
            https://bbpteam.epfl.ch/project/issues/browse/BBPP82-313

    Returns:
        A dataframe of the following form:
                           intensity  proportion
               cellType
              astrocyte      [ ... ]    0.161766
              microglia      [ ... ]    0.368282
        oligodendrocyte      [ ... ]    0.469952
                   glia      [ ... ]    1.000000

        The arrays of the `intensity` columns are float volumetric arrays of the same shape
        (W, H, D). They are assumed to reflect the densities of the corresponding glia cell
        types, up to a constant and uniform factor (an assumption we are working with).
        The `glia` array is a weighted average of the astrocyte, microglia and oligodendrocyte
        arrays where the weights are the cell type proportions. These proportions are inferred
        from literature data together with voxel counts of the annotated volume. The proportions
        are referred to as the cell type global scaling factors (S_celltype) in
        'A Cell Atlas for the Mouse Brain', see 'Glia differentiation section'.
         https://www.frontiersin.org/articles/10.3389/fninf.2018.00084/full.
    """

    def compute_voxel_count(
        region_name: str,
    ) -> int:
        """
        Compute the voxel count of `region_name`.

        Args:
            region_name: name of the region whose voxel count is queried.

        Returns:
            voxel count of `region_name`.
        """
        ids = list(
            region_map.find(region_name, attr="name", with_descendants=True, ignore_case=True)
        )
        return np.count_nonzero(np.isin(annotation_raw, ids))

    glia_celltype_densities.sort_index(inplace=True)
    weights = [compute_voxel_count(region) for region in glia_celltype_densities.columns]
    average_densities = np.average(glia_celltype_densities, weights=weights, axis=1)
    proportions = average_densities / average_densities.sum()

    # We diverge from the formula of the 'Glia Differentiation' section in
    # 'A Cell Atlas for the Mouse Brain' in that we use the weights 1.0 / E_marker normalized by
    # their sums instead of the coefficients C_marker = 1.0 / (E_marker * N_marker).
    combination_data.sort_index(inplace=True)
    intensities = combination_data.groupby("cellType").apply(
        lambda x: np.average(
            np.array(list(x.volume)),
            weights=1.0 / np.array(list(x["averageExpressionIntensity"])),
            axis=0,
        )
    )
    intensities = pd.DataFrame(intensities, columns=["intensity"])
    intensities["proportion"] = proportions

    # Average overall glia intensity
    glia = np.average(np.array(list(intensities.intensity)), axis=0, weights=proportions)
    intensities.loc["glia"] = [glia, 1.0]

    return intensities
