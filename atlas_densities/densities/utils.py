"""Utility functions for cell density computation."""

from typing import TYPE_CHECKING, Dict, Set
from warnings import warn

import numpy as np
import pandas as pd
import scipy.misc
import scipy.ndimage
from atlas_commons.typing import AnnotationT, BoolArray, FloatArray
from tqdm import tqdm

from atlas_densities.exceptions import AtlasDensitiesError, AtlasDensitiesWarning
from atlas_densities.utils import copy_array

if TYPE_CHECKING:  # pragma: no cover
    from voxcell import RegionMap  # type: ignore


MEASUREMENT_TYPES = {
    # For a given brain region R and a given cell type T:
    0: "cell density",  # number of cells of type T per mm^3 in R
    1: "neuron proportion",  # number of cells of type T / number of neurons in R
    2: "cell proportion",  # number of cells of type T / number of cells in R
    3: "cell count per slice",  # number of cells of type T per slice of R, see MEASUREMENT_UNITS
}

MEASUREMENT_UNITS = {
    "cell density": "number of cells per mm^3",
    "neuron proportion": "None",
    "cell proportion": "None",
    "cell count per slice": "number of cells per 5 micrometer-thick slice",
}


def normalize_intensity(
    marker_intensity: FloatArray,
    annotation: AnnotationT,
    threshold_scale_factor: float = 3.0,
    region_id: int = 0,
    copy: bool = True,
) -> FloatArray:
    """
    Subtract a positive constant from the marker intensity and constraint intensity in [0, 1].

    This function
        * subtracts a so-called threshold obtained as the average positive intensity of the marker
            within `region_id` times `scale_factor`. Typically, `region_id` is zero, the background
            annotation associated to any out-of-brain voxel.
        * zeroes negative intensity values and divides intensity by its maximum.

    This function is used to filter out cells expressing a genetic marker below a
    certain threshold. For instance, as PV+ cells react greatly to GAD because of their size,
    they should be filtered in.
    The function also clears the remaining expression of supposed non-expressing regions, e.g.,
    the outside of the annotated brain.

    Args:
        marker_intensity: 3D float array holding the marker intensity.
        annotation: 3D integer array of region identifiers.
        threshold_scale_factor: Scale factor for the threshold.
        region_id: (Optional) identifier of the region over which an intensity average is computed.
            The latter constant is then subtracted from the `marker_intensity`.
            Defaults to 0, the background identifier, which identifies voxels lying out of the
            whole brain annotated volume.
        copy: If True, a deepcopy of the input is normalized and returned. Otherwise, the input is
             normalized in place. Defaults to True.

    Returns:
        3D array, the normalized marker intensity array.
    """

    outside_mean = np.mean(
        marker_intensity[np.logical_and((annotation == region_id), (marker_intensity > 0.0))]
    )
    output_intensity = copy_array(marker_intensity, copy=copy)
    output_intensity -= outside_mean * threshold_scale_factor
    output_intensity[output_intensity < 0.0] = 0.0
    max_intensity = np.max(output_intensity)
    if max_intensity > 0:
        output_intensity /= max_intensity
    else:
        warn(
            f"The thresholding of the dataset according to the intensity in region id={region_id} "
            f"with scaling factor={threshold_scale_factor} resulted in the whole dataset being "
            f"set to 0.",
            AtlasDensitiesWarning,
        )

    return output_intensity


def compensate_cell_overlap(
    marker_intensity: FloatArray,
    annotation: AnnotationT,
    copy: bool = True,
    gaussian_filter_stdv: float = 0.0,
) -> FloatArray:
    """
    Transform, in place, the marker intensity I into - A * log(1 - I) to compensate cell overlap.

    This function, referred to as 'transfer function', compensates the possible cell overlap, see
    'Estimating the Volumetric Cell Density' section in 'A Cell Atlas for the Mouse Brain',
     (Eröe et al. 2018) and Appendix 1.2 of the Supplementary Material.
     https://www.frontiersin.org/articles/10.3389/fninf.2018.00084/full#supplementary-material

    The transfer function
        f:I -> -log(1.0 - I / (1.0 + epsilon))
    is applied to marker intensity where epsilon is choosen close to zero.

    Before applying the transfer function,
        * optionally, applies a Gaussian filter of standard deviation `gaussian_filter_stdv`,
        * intensity values are zeroed outside the annotated volume,
        * the intensity is divided by its maximum value.

    After applying the transfer function,
        * negative values are zeroed,
        * the intensity is divided by its maximum value.

    Arguments:
        marker: intensity scalar field. 3D float array holding the marker intensity over the
            annotated volume.
        annotation: 3D of region identifiers.
        gaussian_filter_stdv: standard deviation value for the gaussian filter to apply on the
            intensity scalar field before transformation. Defaults to 0.0, i.e., no filtering.
        copy: If True, makes deep copy of the input. Otherwise, transform in place. Defaults
            to True.

    Returns:
        The transformed marker intensity array.
    """
    marker_intensity = marker_intensity.copy() if copy else marker_intensity
    if gaussian_filter_stdv > 0.0:
        marker_intensity = scipy.ndimage.gaussian_filter(
            marker_intensity, sigma=gaussian_filter_stdv, output=marker_intensity
        )
    marker_intensity[annotation == 0] = 0.0
    marker_intensity /= np.max(marker_intensity)  # pre-normalize
    marker_intensity[marker_intensity < 0.0] = 0.0
    epsilon = 1e-4  # Used as a safety for intensity values too close to 1.0.
    marker_intensity = -np.log(1.0 - marker_intensity / (1.0 + epsilon))

    marker_intensity /= np.max(marker_intensity)  # post-normalize
    return marker_intensity


# pylint: disable=fixme
# TODO: Re-assess and underline each density validation criterion. Design an actual optimization
# strategy if appropriate.
def optimize_distance_to_line(  # pylint: disable=too-many-arguments
    line_direction_vector: FloatArray,
    upper_bounds: FloatArray,
    sum_constraint: float,
    threshold: float = 1e-7,
    max_iter: int = 45,
    copy: bool = True,
) -> FloatArray:
    """
    Find inside a box the closest point to a line with prescribed coordinate sum.

    This function aims at solving the following (convex quadratic) optmization problem:

    Given a sum S >= 0.0, a line D in the non-negative orthant of the Euclidean N-dimensional
    space and a box B in this orthant (an N-dimensional vector with non-negative coordinates),
    find, if it exists, the point P in B which is the closest to D and whose coordinate sum is S.

    The point P exists if and only if the coordinate sum of B is not less than S.
    The proposed algorithm is iterative and starts with the end point of a direction vector of D.
    First, we uniformly rescale the point coordinates so that it belongs to the plane
     defined by the equation 'coordinate sum = S'. Second, we project the new point on the box
    boundary. The algorithm iterates the two previous steps until we get sufficiently close to B.

    Note: The convergence to the optimal solution is obvious in 2D, but can already fail in 3D.
    At the moment, the function only returns a feasible point.

    Args:
        line_direction_vector: N-dimensional float vector with non-negative coordinates.
        upper_bounds: N-dimensional float vector with non-negative coordinates. Defines the
            the box constraining the optimization process.
        sum_constraints: non-negative float number. The coordinate sum constraints imposed on
            the point P we are looking for.
        threshold: non-negative float value. If the coordinate sum of the current point
            is below `threshold`, the function returns the current point.
        max_iter: maximum number of iterations.
        copy: If True, the function makes a copy of the input `line_direction_vector`. Otherwise
            `line_direction_vector` is modified in-place and holds the optimal value.

    Returns: N-dimensional float vector with non-negative coordinates. The solution point of the
        optimization problem, if it exists, up to inacurracy due to threshold size or early
        termination of the algorithm. Otherwise a point on the boundary of the box B defined by
        `upper_bounds`.
    """
    diff = float("inf")
    iter_ = 0
    point = line_direction_vector.copy() if copy else line_direction_vector
    while diff > threshold and iter_ < max_iter:
        point *= sum_constraint / np.sum(point)
        point = np.min([point, upper_bounds], axis=0)
        diff = np.abs(np.sum(point) - sum_constraint)
        iter_ += 1

    return point


def constrain_cell_counts_per_voxel(  # pylint: disable=too-many-arguments, too-many-branches
    target_sum: float,
    cell_counts: FloatArray,
    cell_counts_upper_bound: FloatArray,
    max_cell_counts_mask: FloatArray = None,
    zero_cell_counts_mask: FloatArray = None,
    epsilon: float = 1e-3,
    copy: bool = True,
):
    """
    Modify `cell_counts` so that it sums to `target_sum` while respecting bound constraints.

    The output array is kept as close as possible to the input in the following sense:
    the algorithm aims at minimizing the distance to the line defined by the input vector
    under the upper bounds and sum constraints.

    Each voxel value of the output is bounded from above by the corresponding value of
    `cell_counts_upper_bound`.

    Additional constraints can be imposed:
        * the voxels in the optional `max_cell_counts_mask` are assigned their maximum values.
        * the voxels in the optional `zero_cell_counts_mask` are assigned the zero value.

    Args:
        target_sum: the value constraining the sum of all voxel values.
        cell_counts: float array of shape (W, H, D) with non-negative values where each voxel
            holds a (float) cell count. This is the array to modify.
        cell_counts_upper_bound: float array of shape (W, H, D) with non-negative values.
            The bounds imposed upon the voxel values of the ouput array.
        max_cell_counts_mask: Optional boolean array of shape (W, H, D) indicating which voxels
            should be assigned their maximum values.
        zero_cell_counts_mask: Optional boolean array of shape (W, H, D) indicating which voxels
            should be assigned the zero value.
        epsilon: tolerated error between the sum of the output and `target_sum`.

    Returns:
        float array of shape (W, H, D) with non-negative values.
        The output array values should not exceed those of `cell_counts_upper_bound`.
        The sum of array values should be `epsilon`-close to `target_sum`.

    Raises:
       AtlasBuildingError if the problem is not feasible, i.e,
       if the target sum is greater than the sum of the voxels of `maximum_density` or if
       the largest possible contribution of voxels with non-zero density is less than `target_sum`.
    """
    if target_sum < epsilon:
        if not copy:
            cell_counts[...] = 0.0
        else:
            cell_counts = np.zeros_like(cell_counts)

        return cell_counts

    max_subsum = 0
    if max_cell_counts_mask is not None:
        max_subsum = np.sum(cell_counts_upper_bound[max_cell_counts_mask])

    if target_sum < max_subsum - epsilon:
        raise AtlasDensitiesError(
            "The contribution of voxels with prescribed maximum density, that is"
            f" {max_subsum - epsilon}"
            f" exceeds the target sum, that is, {target_sum}. One of the two constraints cannot be"
            " fulfilled."
        )

    zero_indices_subsum = 0
    if zero_cell_counts_mask is not None:
        zero_indices_subsum = np.sum(cell_counts_upper_bound[zero_cell_counts_mask])

    if np.sum(cell_counts_upper_bound) - zero_indices_subsum < target_sum - epsilon:
        raise AtlasDensitiesError(
            "The maximum contribution of voxels with non-zero density"
            " is less than the target sum. The target sum cannot be reached."
        )

    cell_counts = cell_counts.copy() if copy else cell_counts
    complement = None
    if max_cell_counts_mask is not None:
        cell_counts[max_cell_counts_mask] = cell_counts_upper_bound[max_cell_counts_mask]
        complement = max_cell_counts_mask.copy()

    if zero_cell_counts_mask is not None:
        cell_counts[zero_cell_counts_mask] = 0.0
        if complement is None:
            complement = zero_cell_counts_mask.copy()
        complement = np.logical_or(complement, zero_cell_counts_mask)

    if complement is None:
        complement = tuple([slice(0, None)] * 3)
    else:
        complement = np.invert(complement)

    line_direction_vector = cell_counts[complement]
    upper_bound = cell_counts_upper_bound[complement]

    # Find a cell count field respecting all the constraints and which is as close
    # as possible to the line defined by the input cell counts wrt to Euclidean norm.
    cell_counts[complement] = optimize_distance_to_line(
        line_direction_vector,
        upper_bound,
        target_sum - max_subsum,
        threshold=epsilon,
        max_iter=50,
        copy=copy,
    )

    abs_error = np.abs(np.sum(cell_counts) - target_sum)
    if abs_error > epsilon:
        raise AtlasDensitiesError(
            f"The target sum could not be reached. "
            f"The absolute error is {abs_error}. It is larger than the tolerance "
            f"epsilon = {epsilon}"
        )

    return cell_counts


def get_group_ids(region_map: "RegionMap", cleanup_rest: bool = False) -> Dict[str, Set[int]]:
    """
    Get AIBS structure ids for several region groups of interest.

    The groups below have been selected because specific count information is available
    in the scientific literature.

    Args:
        region_map: object to navigate the mouse brain regions hierarchy
            (instantied from AIBS 1.json).
        cleanup_rest: (Optional) If True, the id of any ascendant region of the Cerebellum and
            Isocortex groups are removed from the ids of the Rest group. This makes sure that
            the set of ids of the Rest group is closed under taking descendants.

    Returns:
        A dictionary whose keys are region group names and whose values are
        sets of structure identifiers.
    """
    cerebellum_group_ids = region_map.find(
        "Cerebellum", attr="name", with_descendants=True
    ) | region_map.find("arbor vitae", attr="name", with_descendants=True)
    isocortex_group_ids = (
        region_map.find("Isocortex", attr="name", with_descendants=True)
        | region_map.find("Entorhinal area", attr="name", with_descendants=True)
        | region_map.find("Piriform area", attr="name", with_descendants=True)
    )
    purkinje_layer_ids = region_map.find("@.*Purkinje layer", attr="name", with_descendants=True)
    fiber_tracts_ids = (
        region_map.find("fiber tracts", attr="name", with_descendants=True)
        | region_map.find("grooves", attr="name", with_descendants=True)
        | region_map.find("ventricular systems", attr="name", with_descendants=True)
        | region_map.find("Basic cell groups and regions", attr="name")
        | region_map.find("Cerebellum", attr="name")
    )
    cerebellar_cortex_ids = region_map.find("Cerebellar cortex", attr="name", with_descendants=True)
    molecular_layer_ids = cerebellar_cortex_ids & region_map.find(
        "@.*molecular layer", attr="name", with_descendants=True
    )
    rest_ids = region_map.find("root", attr="name", with_descendants=True)
    rest_ids -= cerebellum_group_ids | isocortex_group_ids

    if cleanup_rest:
        # Make the Rest group stable under taking descendants
        isocortex_id = region_map.find("Isocortex", attr="name").pop()
        cerebellum_id = region_map.find("Cerebellum", attr="name").pop()
        ascendant_ids = set(region_map.get(isocortex_id, attr="id", with_ascendants=True))
        ascendant_ids |= set(region_map.get(cerebellum_id, attr="id", with_ascendants=True))
        rest_ids -= ascendant_ids

    return {
        "Cerebellum group": cerebellum_group_ids,
        "Isocortex group": isocortex_group_ids,
        "Fiber tracts group": fiber_tracts_ids,
        "Purkinje layer": purkinje_layer_ids,
        "Molecular layer": molecular_layer_ids,
        "Cerebellar cortex": cerebellar_cortex_ids,
        "Rest": rest_ids,
    }


def get_group_names(region_map: "RegionMap", cleanup_rest: bool = False) -> Dict[str, Set[str]]:
    """
    Get AIBS names for regions in several region groups of interest.

    Args:
        region_map: object to navigate the mouse brain regions hierarchy
            (instantied from AIBS 1.json).
        cleanup_rest: (Optional) If True, the name of any ascendant region of the Cerebellum and
            Isocortex groups are removed from the names of the Rest group. This makes sure that
            the set of names of the Rest group is closed under taking descendants.

    Returns:
        A dictionary whose keys are region group names and whose values are
        sets of brain region names.
    """

    group_ids = get_group_ids(region_map, cleanup_rest)

    return {
        group_name: {region_map.get(id_, attr="name") for id_ in group_ids[group_name]}
        for group_name in ["Cerebellum group", "Isocortex group", "Rest"]
    }


def get_region_masks(
    group_ids: Dict[str, Set[int]], annotation: FloatArray
) -> Dict[str, BoolArray]:
    """
    Get the boolean masks of several region groups of interest.

    The groups below have been selected because specific count information is available
    in the scientific literature.

    Args:
        group_ids: a dictionary whose keys are group names and whose values are
            sets of AIBS structure identifiers.
        annotation: integer array of shape (W, H, D) enclosing the AIBS annotation of
            the whole mouse brain.

    Returns:
        A dictionary whose keys are region group names and whose values are
        the boolean masks of these groups. Each boolean array is of shape (W, H, D) and
        encodes which voxels belong to the corresponding group.
    """

    return {
        group_name: np.isin(annotation, list(group_ids[group_name]))
        for group_name in ["Cerebellum group", "Isocortex group", "Rest"]
    }


def get_aibs_region_names(region_map: "RegionMap") -> Set[str]:
    """
    Retrieve the names of every region in `region_map`.

    Args:
        region_map: RegionMap object to navigate the brain regions hierarchy
            instantiated with the 1.json hierarchy file from AIBS.

    Returns:
        Set of strings containing the names of all regions represented in
        `region_map`.

    """
    aibs_region_ids = region_map.find(
        "Basic cell groups and regions", attr="name", with_descendants=True
    )

    return {region_map.get(id_, "name") for id_ in aibs_region_ids}


def get_hierarchy_info(
    region_map: "RegionMap",
    root: str = "Basic cell groups and regions",
) -> "pd.DataFrame":
    """
    Returns the name and the descendant_id_set of each region that can be found by `region_map`.

    Note: We assume that the hierarchy file has unique brain region names.

    Args:
        region_map: RegionMap object to navigate the brain regions hierarchy.
        root: (Optional) root of the hierarchy tree used by `region_map`. Defaults to
            "Basic cell groups and regions" which corresponds to the AIBS whole mouse
            brain in AIBS 1.json.

    Returns:
        returns a dataframe with index a list of unique region ids and two columns
        (values are fake)
                 descendant_id_set    brain_region
            1    {1, 3}               "Cerebellum"
            2    {2, 4, 10}           "Isocortex"
                 ...             ...
        The index consists in the sorted list of the identifiers of every region recorded in
        `region_map` under root. The column `descendant_id_set` holds for each region the set of
        identifiers of the descendants including the region itself. `brain region` is the list of
        every region name.

    """
    region_ids = list(region_map.find(root, attr="name", with_descendants=True))
    region_ids.sort()
    descendant_id_sets = [
        region_map.find(id_, attr="id", with_descendants=True) for id_ in region_ids
    ]
    region_names = [region_map.get(id_, attr="name") for id_ in region_ids]
    data_frame = pd.DataFrame(
        {"brain_region": region_names, "descendant_id_set": descendant_id_sets}, index=region_ids
    )

    return data_frame


def compute_region_volumes(
    annotation: AnnotationT,
    voxel_volume: float,
    hierarchy_info: "pd.DataFrame",
) -> "pd.DataFrame":
    """
    Compute the volume in mm^3 of every 3D brain region of `annotation` with an identifier
    in `hierarchy_info.index`.

    Args:
        annotation: int array of shape (W, H, D) holding the annotation of the whole AIBS
            mouse brain. (The integers W, H and D are the dimensions of the array).
        voxel_volume: volume in mm^3 of a voxel in any of the volumetric input arrays.
            This is (25 * 1e-3) ** 3 for an AIBS atlas nrrd file with 25um resolution.
        hierarchy_info: data frame returned by
            :func:`atlas_densities.densities.utils.get_hierarchy_info`.

    Returns:
        DataFrame of the following form (values are fake):
             brain_region                    volume  id_volume
        10   Basic cell groups and regions   2000    55
        123  Cerebrum                        700     0
             ...                             ...
        The index is the sorted list of all region identifiers.
        The column `id_volume` holds the volumes of the brain regions of `annotation`
        which are labeled by a single identifier (descendant subregions are excluded).
        The latter column is created only if `hierarchy_info` has no `id_set` column,
        in which case its index is made of unique integer identifiers.
    """
    id_volumes = []
    for id_ in tqdm(hierarchy_info.index):
        id_volumes.append(np.count_nonzero(annotation == id_) * voxel_volume)

    result = pd.DataFrame(
        {
            "brain_region": hierarchy_info["brain_region"],
            "id_volume": id_volumes,
        },
        index=hierarchy_info.index,
    )

    volumes = []
    for id_ in tqdm(hierarchy_info.index):
        id_set = hierarchy_info.loc[id_, "descendant_id_set"]
        volume = result.loc[list(id_set), "id_volume"].sum()
        volumes.append(volume)
    result["volume"] = volumes

    return result


def compute_region_cell_counts(
    annotation: AnnotationT,
    density: FloatArray,
    voxel_volume: float,
    hierarchy_info: "pd.DataFrame",
    with_descendants: bool = True,
) -> "pd.DataFrame":
    """
    Compute the cell count of every 3D region in `annotation` labeled by a unique
    id in `ids` wrt cell `density`.

    Args:
        annotation: int array of shape (W, H, D) holding the annotation of the whole AIBS
            mouse brain. (The integers W, H and D are the dimensions of the array).
        density: float array of shape `annotation.shape` holding the volumetric density of a
            given cell type. The value assigned to a voxel represents the cell density in this
            voxel expressed in number of cells per mm^3.
        voxel_volume: volume in mm^3 of a voxel in any of the volumetric input arrays.
            This is (25 * 1e-6) ** 3 for an AIBS atlas nrrd file with 25um resolution.
        hierarchy_info: data frame returned by
            :func:`atlas_densities.densities.utils.get_hierarchy_info`.
        with_descendants: if True, a computed cell count refers to the entire 3D region
            designated by a region name, including all descendants subregions. Otherwise, the
            computed cell count is the cell count of the 3D region labeled by the unique region
            identifier. Defaults to True.

    Returns:
        DataFrame of the following form (values are fake):
             brain_region                    cell_count
        10   Basic cell groups and regions   200.30
        123  Cerebrum                         75.05
             ...                             ...
        The index is the sorted list of all region identifiers.
    """
    id_counts = []
    for id_ in tqdm(hierarchy_info.index):
        mask = annotation == id_
        id_counts.append(np.sum(density[mask]) * voxel_volume)

    result = pd.DataFrame(
        {"brain_region": hierarchy_info["brain_region"], "cell_count": id_counts},
        index=hierarchy_info.index,
    )

    if with_descendants:
        counts = []
        for id_set in tqdm(hierarchy_info["descendant_id_set"]):
            count = result.loc[list(id_set), "cell_count"].sum()
            counts.append(count)
        result["cell_count"] = counts

    return result


def zero_negative_values(array: FloatArray) -> None:
    """
    Zero negative values resulting from round-off errors.

    Modifies `array` in place.

    Args:
        array: float numpy array. We expect most of the `array` values to be non-negative.
            Negative values should be negligible when compared to positive values.

    Raises:
        AtlasDensitiesError if the absolute value of the sum of all negative values exceeds
            1 percent of the sum of all positive values, or if the smallest negative value is
            not negligible wrt to the mean of the non-negative values.
    """
    negative_mask = array < 0.0
    if np.count_nonzero(negative_mask) == 0:
        return

    non_negative_mask = np.invert(negative_mask)

    if np.abs(np.sum(array[negative_mask])) / np.sum(array[non_negative_mask]) > 0.01:
        raise AtlasDensitiesError(
            "The absolute value of the sum of all negative values exceeds"
            " 1 percent of the sum of all positive values"
        )
    ratio = np.abs(np.min(array[negative_mask])) / np.mean(array[non_negative_mask])
    if not np.isclose(ratio, 0.0, atol=1e-08):
        raise AtlasDensitiesError(
            "The smallest negative value is not negligible wrt to "
            "the mean of all non-negative values."
        )

    array[negative_mask] = 0.0
