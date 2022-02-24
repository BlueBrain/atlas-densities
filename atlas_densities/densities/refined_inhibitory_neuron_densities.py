"""Functions to create volumetric neuron densities for the following inhibitory neuron types of the
mouse brain:
- PV+
- SST+
- VIP+
- GAD67+ (a superset of the union of the previous three)

The algorithm is presented in "Atlas of inhibitory neurons in the mouse brain" by D. Dimitri
Rodarie et al., to appear in <journal>.

This code supersedes the inhibitory_neuron_densities module, which is maintained to allow
comparisons with the former approach of "A Cell Atlas for the Mouse Brain" by C. Eroe et
al., 2018.
"""

import copy
import logging
from typing import TYPE_CHECKING, Dict, List

import numpy as np
from atlas_commons.typing import AnnotationT, BoolArray, FloatArray
from tqdm import tqdm
from voxcell import RegionMap

from atlas_densities.densities.inhibitory_neuron_densities_helper import (
    average_densities_to_cell_counts,
)
from atlas_densities.densities.utils import compute_region_volumes, get_hierarchy_info

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

L = logging.getLogger(__name__)


class VolumetricDensityHelper:
    """
    Helper class to compute volumetric cell densities out of
    cell counts.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        annotation: AnnotationT,
        voxel_volume: float,
        neuron_density: FloatArray,
        region_counts: "pd.DataFrame",
        cell_type: str,
        cell_subtypes: List[str],
    ) -> None:
        """
        Args:
            annotation: int array of shape (W, H, D) holding the annotation of the whole
                brain model. (The integers W, H and D are the dimensions of the array).
            voxel_volume: volume expressed in mm^3 of a voxel. The input volumetric data (annotation
                and neuron density) are assumed to have voxels of the same dimensions.
            neuron_density: non-negative float array of shape (W, H, D) where W, H and D are
                integer dimensions. This array holds the volumetric neuron density of the brain
                model expressed in number of neurons per mm^3.
            region_counts: data frame with a pair of columns T and T_standard_deviation for each
                cell type T. The possible T are `cell_type` (e.g., "gad67+") or a cell subtype in
                `cell_subtypes` (e.g., "pv+", "sst+" or "vip+"). The index of `region_counts` is
                a list of brain region names. The columns hold the counts of the corresponding
                cell types and their standard deviations.
            cell_type: name of a cell type (e.g., "gad67+") which can be composed of several cell
                subtypes.
            cell_subtypes: names of cell subtypes of `cell_type`, e.g., "pv+", "sst+" or "vip+"
                if `cell_type` is "gad67+". The list `cell_subtypes` does not necessrily exhausts
                all the subtype of `cell_type`. (For example, there are cells reacting to gad67
                which don't react to any of pv, sst or vip).
        """
        self.annotation = annotation
        self.voxel_volume = voxel_volume
        self.neuron_density = neuron_density
        self.region_counts = region_counts
        self.cell_type = cell_type
        self.cell_subtypes = cell_subtypes
        self.cell_types = [cell_type] + cell_subtypes
        self.volumetric_densities: Dict[str, FloatArray] = {}

    def initialize_volumetric_densities(self) -> None:
        """
        Initialize with 0.0 each volumetric density array to be filled.
        """
        if not bool(self.volumetric_densities):
            self.volumetric_densities = {
                cell_type: np.zeros_like(self.annotation, dtype=float)
                for cell_type in self.cell_types
            }

    def get_zero_counts(self) -> Dict[str, float]:
        """
        Return a dict with key-value pairs (cell_type, cell_count) with cell_count set to 0.0
        """
        return {cell_type: 0.0 for cell_type in self.cell_types}

    def get_neuron_count(self, region_mask: BoolArray) -> float:
        """
        Returns the neuron count of the region in `self.annotation` defined by `region_mask`.

        Args:
            region_mask: 3D boolean mask of shape `self.annotation.shape`.

        Returns:
            neuron count of type float
        """

        return np.sum(self.neuron_density[region_mask]) * self.voxel_volume

    def fill_volumetric_densities(self, region_id: int, region_counts: Dict[str, float]) -> None:
        """
        Fill the volumetric density of each cell type in `self.cell_types` in the region
        with id = `region_id` with prescribed cell counts given by `region_counts`.

        The arrays `self.volumetric_densities` are modified in-place.

        Args:
            region_id: identifier of the 3D region where density values will be set.
            region_counts: dict with key-value pairs of the form (cell_type, cell_count)
                where cell_count is presribed number of cells (float) of the specified type.
        """
        region_mask = self.annotation == region_id
        for cell_type in self.volumetric_densities.keys():
            self.volumetric_densities[cell_type][region_mask] = self.neuron_density[region_mask]
            overall_density = np.sum(self.neuron_density[region_mask])
            if np.isclose(overall_density, 0.0):
                factor = 0.0
            else:
                factor = region_counts[cell_type] / (overall_density * self.voxel_volume)
            self.volumetric_densities[cell_type][region_mask] *= factor

    def compute_consistent_region_counts(
        self,
        region_id: int,
        cell_count: float,
        deviation: float,
        subcounts: List[float],
        subsum_deviation: float,
    ) -> Dict[str, float]:
        """
        Modify `cell_count` and `subcounts` to ensure cell counts consistency.

        Modify `cell_count` and `subcounts` so that:
        - the sum of `subcounts` does not exceed `cell_count`
        - `cell_count`does not exceed the neuron count of the region with id
        `region_id`
        - `cell_count` and `subcounts` both remain in prescribed ranges defined by means of
        `deviation` and `subsum_deviation` whenever possible. If not, preserve the proportions
        of `subcounts`.

        Returns:
            a dict {type_: <count> for type_ in self.cell_types} with consistent <count> values.
            The value of `cell_count` correponds to `self.cell_type`, the `subcounts` to
            `self.cell_subtypes`.

        Raises:
            AtlasDensitiesError if the modified `cell_count` exceeds the neuron cell count plus
            `tolerance`.

        """
        tolerance = 1e-2
        # Zero cell counts below tolerance
        if cell_count <= tolerance:
            return self.get_zero_counts()

        subsum = np.sum(subcounts)
        neuron_count = self.get_neuron_count(self.annotation == region_id)
        inconsistent = cell_count < subsum
        if inconsistent:
            if subsum_deviation == 0.0 and deviation == 0.0:
                cell_count = 0.5 * (cell_count + subsum)
            else:
                ratio = (subsum - cell_count) / (subsum_deviation + deviation)
                # If ratio <= 1.0, then a solution is found in
                # [count - std deviation, count + std deviation]
                # for every cell count. Otherwise the new value of cell_count is located in
                # [cell_count + deviation, subsum - subsum_deviation]
                cell_count += ratio * deviation

        cell_count = min(cell_count, neuron_count)

        # If cell counts were inconsistent, we still preserve proportions
        # of the cell subtypes of `self.cell_type`.
        if inconsistent and subsum > 0.0:
            subcounts = (subcounts / subsum) * cell_count

        result = dict(zip(self.cell_subtypes, subcounts))
        result[self.cell_type] = cell_count

        return result

    def compute_consistent_leaf_counts_(
        self,
        region_name: str,
        region_id: int,
    ) -> Dict[str, float]:
        """
        Helper function to make cell counts of leaf regions consistent across cell types.

            Args:
                region_name: name of the leaf region whose cell counts are modified
                region_id: identifier of the region `region_name`.

            Returns:
                A dict with the form of the return value of
                :func:`compute_consistent_region_counts`.
        """
        cell_count = self.region_counts.at[region_name, self.cell_type]
        subcounts = [
            self.region_counts.at[region_name, cell_subtype] for cell_subtype in self.cell_subtypes
        ]
        subsum_deviation = np.sum(
            [
                self.region_counts.at[region_name, cell_subtype + "_standard_deviation"]
                for cell_subtype in self.cell_subtypes
            ]
        )
        deviation = self.region_counts.at[region_name, self.cell_type + "_standard_deviation"]

        return self.compute_consistent_region_counts(
            region_id, cell_count, deviation, subcounts, subsum_deviation
        )

    def compute_consistent_leaf_counts(
        self, hierarchy_leaf_info: "pd.DataFrame"
    ) -> Dict[str, Dict[str, float]]:
        """
        Modify cell counts of leaf regions to make them consistent.

        This function modifies the initial leaf counts in `self.region_counts` so that:
        - (1) the cell count of `self.cell_type` does not exceed the sum of the cell counts
        of its `self.cell_subtypes` in every leaf region.
        - (2) the cell count of `self.cell_type` does not exceed the neuron count in every leaf
        region.

        If (2) is violated, the function raises an AtlasDensitiesError.

        The algorithm attempts to modify the counts so that every count remains in the range
        [count - standard_deviation, count + standard_deviation] for the deviation defined in
        self.region_counts. The subtype counts and deviations are summed and the subcounts sum is
        considered as the cell count of unique virtual cell subtype with unique range
        [subsum - subsum_deviation, subsum + subsum_deviation]. If the range of the
        `self.cell_type` count intersects the previous range, we apply a modification which
        respects each individual subtype range. Otherwise, we only preserve the proportions of the
        subtype cell counts.

        Args:
            hierarchy_leaf_info: data frame with a "brain_region" columns holding the names
                the regions with ids listed in hierarchy_leaf_info.index.

        Returns:
            dict whose keys are leaf region names and values are dict of the form:
                {"gad67+": 100.0, "pv+": 30.0, "sst+": 30.0, "vip+": 35.0}.
                (Values are fake.) The float values are the modified cell counts.
                In this example, `self.cell_type` is "gad67+" and `self.cell_subtypes`
                is ["pv+", "sst+", "vip+"].
        """
        leaf_region_counts = {}
        for id_ in hierarchy_leaf_info.index:
            region_name = hierarchy_leaf_info.at[id_, "brain_region"]
            # TODO: address duplicate region names  # pylint: disable=fixme
            # such as "Prelimbic area, layer 2" created after the splitting of layer 2/3.
            leaf_region_counts[region_name] = self.compute_consistent_leaf_counts_(
                region_name,
                id_,
            )

        return leaf_region_counts

    def compute_consistent_counts_helper_(
        self,
        updated_counts: Dict[str, Dict[str, float]],
        nomansland_counts: Dict[str, Dict[str, float]],
        root: dict,
    ) -> None:
        """
        Recursive helper function to re-estimate region cell counts, starting from modified leaf
        counts.

        The dictionaries `updated_counts` and `nomansland_counts` are modified in-place.

        This function recursively re-evaluates the region cell counts for every cell type from
        bottom to top. The difference between the initial estimate in `self.region_counts` and the
        new estimate in `updated_counts` (computed recursively) is used to provide a cell count for
        each nomans'land, i.e., each 3D non-leaf region whose identifier labels voxels in
        `self.annotation`.
        """
        region_name = root["name"]
        if "children" in root and root["children"]:
            updated_counts[region_name] = self.get_zero_counts()
            for child in root["children"]:
                self.compute_consistent_counts_helper_(updated_counts, nomansland_counts, child)
                if child["name"] in updated_counts:
                    for type_ in self.cell_types:
                        updated_counts[region_name][type_] += updated_counts[child["name"]][type_]
            region_mask = self.annotation == root["id"]
            if np.any(region_mask):  # no man's land
                # We use the difference between two estimates in order to assign a cell count value
                # to the set of voxels labeled by root["id"]:
                # - the initial estimate reported in the scientific literature or obtained after
                # a linear fitting
                # - the sum of the estimates of the child regions which have been made consistent
                # in an earlier step (recursion through the brain region hierarchy)
                neuron_count = self.get_neuron_count(region_mask)
                cell_count = self.region_counts.loc[region_name, self.cell_type]
                diff_count = cell_count - updated_counts[region_name][self.cell_type]
                cell_count = min(max(diff_count, 0.0), neuron_count)
                deviation = self.region_counts.loc[
                    region_name, self.cell_type + "_standard_deviation"
                ]
                subsum_deviation = np.sum(
                    [
                        self.region_counts.at[region_name, cell_subtype + "_standard_deviation"]
                        for cell_subtype in self.cell_subtypes
                    ]
                )
                subcounts = [
                    max(
                        self.region_counts.loc[region_name, cell_subtype]
                        - updated_counts[region_name][cell_subtype],
                        0.0,
                    )
                    for cell_subtype in self.cell_subtypes
                ]
                nomansland_counts[region_name] = self.compute_consistent_region_counts(
                    root["id"], cell_count, deviation, subcounts, subsum_deviation
                )
                for type_ in self.cell_types:
                    updated_counts[region_name][type_] += nomansland_counts[region_name][type_]

    def compute_consistent_counts(
        self,
        leaf_region_counts: Dict[str, Dict[str, float]],
        hierarchy: dict,
    ) -> Dict[str, Dict[str, float]]:
        """
        Re-estimate recursively region cell counts, starting from modified leaf counts.

        This function recursively re-evaluates the region cell counts for every cell type from
        bottom to top. The difference between the initial estimate in `self.region_counts` and the
        new estimate in (computed recursively) is used to provide a cell count for each
        nomans'land, i.e., each 3D non-leaf region whose identifier labels voxels in
        `self.annotation`.

        Args:
            leaf_region_counts: dict whose keys are leaf region names and values are dict
                of the form {"gad67+": 100.0, "pv+": 30.0, "sst+": 30.0, "vip+": 35.0} with
                consistent cell counts.
            hierarchy: dict holding the tree of the brain regions hierarchy. Children regions
                are accessed via the "children" attribute. (dict counterpart of 1.json or
                hierarchy.json)

        Returns:
            dict whose keys are names of nomans'lands and whose values are dicts holding the
            cell count estimates for such regions for every cell type in `self.cell_types`.
        """
        nomansland_counts: Dict[str, Dict[str, float]] = {}
        self.compute_consistent_counts_helper_(
            copy.deepcopy(leaf_region_counts), nomansland_counts, hierarchy
        )

        return nomansland_counts


def create_inhibitory_neuron_densities(  # pylint: disable=too-many-locals
    hierarchy: dict,
    annotation: AnnotationT,
    voxel_volume: float,
    neuron_density: FloatArray,
    average_densities: "pd.DataFrame",
) -> Dict[str, FloatArray]:
    """
    Create a 3D float array for each cell type labelling the columns of `average_densities`.

    Voxel values are cell density values expressed in number of cells per mm^3.
    For a given `annotation` array, a non-zero annotated voxel
    - either belongs to a leaf region (no child subregion in the brain region tree hierarchy)
    - or is in a nomans'land.

    The function modifies the estimates in `average_densities` to make them consistent
    accross cell types: the average density of "gad67+" cells in a leaf region L should be
    at most the sum of the average densities of its subtypes under scrutiny (e.g. "pv+", "sst+" and
    "vip+") and should not exceed the neuron density of L.

    Whenever possible, modified densities are kept in the range [initial value - std] deviation,
    initial value + std deviation] for the standard deviations specified in `average_densities`.

    The modified leaf densities are used to re-evaluate the average densities of higher
    regions. The difference between the new estimate and the original estimate in
    `average_densities` is used to determine the average densities of nomans'lands.

    Note:
        The density values of regions which are not listed in `average_densities` are
        set to 0.0. Idem for regions whose initial estimates contain a NaN value.

    Args:
        hierarchy: dict holding the tree of the brain regions hierarchy. Children regions
            are accessed via the "children" attribute. (dict counterpart of 1.json or
            hierarchy.json)
        annotation: int array of shape (W, H, D) holding the annotation of the whole
            brain model. (The integers W, H and D are the dimensions of the array).
        voxel_volume: volume expressed in mm^3 of a voxel. The input volumetric data (annotation
            and neuron density) are assumed to have voxels of the same dimensions.
        neuron_density: non-negative float array of shape (W, H, D). This array holds the
            volumetric neuron density of the brain model expressed in number of neurons per mm^3.
        average_densities: a data frame whose columns are described in
            :func:`atlas_densities.densities.fitting.linear_fitting` containing the average
            cell densities of brain regions and their associated standard deviations. Columns are
            labelled by T and T_standard_deviation for various cell types T. The index of
            `average_densities` is a list of region names.

    Returns:
        a dict whose keys are cell types (e.g., "gad67+", "pv+", "sst+" and "vip+") and whose
        values are float arrays of shape (W, H, D) holding the cell density of each cell type
        expressed in number of cells per mm^3.

    """

    region_map = RegionMap.from_dict(hierarchy)
    hierarchy_info = get_hierarchy_info(region_map, root="root")
    L.info("Computing region volumes ...")
    volumes = compute_region_volumes(annotation, voxel_volume, hierarchy_info)

    # We compute cell counts from average densities as we will
    # modify cell densities recursively starting from the leaves
    L.info("Computing region cell counts ...")
    region_counts = average_densities_to_cell_counts(average_densities, volumes)
    cell_types = {column.replace("_standard_deviation", "") for column in region_counts.columns}

    density_helper = VolumetricDensityHelper(
        annotation,
        voxel_volume,
        neuron_density,
        region_counts,
        "gad67+",
        list(cell_types - {"gad67+"}),
    )

    L.info("Making leaf region cell counts consistent ...")
    leaf_ids = [id_ for id_ in hierarchy_info.index if region_map.is_leaf_id(id_)]
    hierarchy_leaf_info = hierarchy_info[hierarchy_info.index.isin(leaf_ids)]
    leaf_region_counts = density_helper.compute_consistent_leaf_counts(hierarchy_leaf_info)
    L.info("Making remaining region cell counts consistent ...")
    nomansland_counts = density_helper.compute_consistent_counts(
        leaf_region_counts,
        hierarchy,
    )

    L.info("Initializing volumetric densities ...")
    density_helper.initialize_volumetric_densities()

    L.info("Filling leaf region volumetric densities ...")
    for region_id in tqdm(hierarchy_leaf_info.index):
        region_name = hierarchy_leaf_info.brain_region[region_id]
        if region_name in leaf_region_counts:  # Some leaf regions may be missing
            density_helper.fill_volumetric_densities(region_id, leaf_region_counts[region_name])

    L.info("Filling remaining volumetric densities ...")
    for region_name in tqdm(nomansland_counts):
        region_id = region_map.find(region_name, attr="name").pop()
        density_helper.fill_volumetric_densities(region_id, nomansland_counts[region_name])

    return density_helper.volumetric_densities
