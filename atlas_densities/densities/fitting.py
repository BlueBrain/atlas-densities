"""Functions to estimate average cell densities of various cell types in brain regions based
on measurements from the scientific literature.

Known density values are used to perform a linear fitting for each cell type of interest in
the following three groups:

- isocortex
- cerebellum
- the rest

The cerebellum was singled out because of its very high cell densities wrt the rest of the mouse
brain. The isocortex was singled out because its layer densities and cell type compositions are
quite similar across its subregions.

For each cell type, there are hence three linear fittings.
To estimate the average density of a cell type T of a brain region R, we select the linear mapping
obtained for T and the group R it belongs to.
"""
from __future__ import annotations

import itertools as it
import logging
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from atlas_commons.typing import AnnotationT, BoolArray, FloatArray
from scipy.optimize import curve_fit
from tqdm import tqdm
from voxcell import RegionMap, voxel_data

from atlas_densities.densities import utils
from atlas_densities.densities.measurement_to_density import remove_unknown_regions
from atlas_densities.exceptions import AtlasDensitiesError, AtlasDensitiesWarning

L = logging.getLogger(__name__)

MarkerVolumes = Dict[str, Dict[str, Union[FloatArray, List[int]]]]


def create_dataframe_from_known_densities(
    region_names: list[str],
    average_densities: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a data frame with `region_names` as index and with two columns for each cell type in
    `average_densities["cell_type"]`.

    The data frame is partially filled with the values of `average_densities`.

    If one or more average density values are recorded in `average_densities` for a given region
    R (e.g., Isocortex) and a given cell type T (e.g., PV+), then the function writes the average
    of these values at the intersection of the row of R and the column of T (T is lower cased in the
    output).

    Similarly, the column T_standard_deviation is filled with the average of the standard
    deviations of the recorded densities corresponding to R and T.

    Unknown density values and standard deviations are set with np.nan.

    Args:
        region_names: names of brain regions, used as index of the output data frame.
        average_densities: a data frame whose columns are described in
            :func:`atlas_densities.app.densities.compile_measurements` and where all
            measurements are of type "cell density" in number of cells per mm^3.

    Returns:
        data frame whose index is `region_names` and with a pair of columns labeled by T and
        T_standard_deviation for each cell type T in `average_cell_densities["cell_type"] (e.g.,
        T = "pv+", "inhibitory neuron" or "sst+"). Column labels are lower cased.
    """

    assert len(average_densities.index) > 0, (
        "No rows found in `average_densities` data frame."
        " Expected non-empty data frame filled with density values from the scientific literature."
    )

    region_count = len(region_names)
    columns = {}
    cell_types = np.unique(average_densities["cell_type"])
    for cell_type in cell_types:
        cell_type = cell_type.lower().replace(" ", "_")
        columns[cell_type] = np.full((region_count,), np.nan)
        columns[cell_type + "_standard_deviation"] = np.full((region_count,), np.nan)

    result = pd.DataFrame(columns, index=region_names)

    for region_name in result.index:
        density_mask = average_densities["brain_region"] == region_name
        for cell_type in cell_types:
            cell_type_mask = density_mask & (average_densities["cell_type"] == cell_type)
            cell_type = cell_type.lower().replace(" ", "_")
            result.at[region_name, cell_type] = np.mean(
                average_densities[cell_type_mask]["measurement"]
            )
            result.at[region_name, cell_type.lower() + "_standard_deviation"] = np.mean(
                average_densities[cell_type_mask]["standard_deviation"]
            )
    result.sort_index(inplace=True, axis=1)  # sort columns in lexico-graphical order

    return result


def fill_in_homogenous_regions(
    homogenous_regions: pd.DataFrame,
    annotation: AnnotationT,
    neuron_density: FloatArray,
    densities: pd.DataFrame,
    hierarchy_info: pd.DataFrame,
    cell_density_stddevs: Optional[dict[str, float]] = None,
) -> None:
    """
    Fill the average density values of every region where all neurons are either inhibitory or
    excitatory.

    The `densities` data frame is modified in-place.

    The "inhibitory" column of regions whose neurons are inhibitory only will be filled by means of
    of the volumetric `neuron_density`.

    The "inhibitory" column of regions whose neurons are excitatory only will be filled with 0.0.
    The "inhibitory_standard_deviation" will be set with 0.0 for those regions.

    Args:
        homogenous_regions: data frame with two columns, brain_region and cell_type.
            The brain_region column holds region names and the values of the cell_type column are
            "inhibitory" or "excitatory" depending on whether every neuron of the region is
            inhibitory or excitatory.
        annotation: int array of shape (W, H, D) holding the annotation of the whole
            brain model. (The integers W, H and D are the dimensions of the array).
        neuron_density: non-negative float array of shape (W, H, D) where W, H and D are integer
            dimensions. This array holds the volumetric neuron density of the brain model expressed
            in number of neurons per mm^3.
        densities: the data frame returned by
            :fun:`atlas_densities.densities.fitting.create_dataframe_from_known_densities`.
        hierarchy_info: data frame with columns "descendant_id_set" (set[int]) and "brain_region"
            (str) and whose index is a list of region ids.
            See :fun:`atlas_densities.densities.utils.get_hierarchy_info`.
        cell_density_stddevs: (Optional) dict whose keys are brain regions names and whose values
            are standard deviations of average cell densities of the corresponding regions.
            Defaults to None, in which case the output standard deviation is set with the density
            value.
    """

    hierarchy_info = hierarchy_info.copy()
    hierarchy_info["id"] = hierarchy_info.index
    hierarchy_info.set_index("brain_region", inplace=True)
    inhibitory_mask = homogenous_regions["cell_type"] == "inhibitory"

    for region_name in homogenous_regions[inhibitory_mask]["brain_region"]:
        desc_id_set = hierarchy_info.at[region_name, "descendant_ids"]
        id_mask = [id_ in desc_id_set for id_ in hierarchy_info["id"]]
        for child_region_name in hierarchy_info[id_mask].index:
            region_mask = np.isin(
                annotation, list(hierarchy_info.at[child_region_name, "descendant_ids"])
            )
            density = np.mean(neuron_density[region_mask])
            densities.at[child_region_name, "inhibitory_neuron"] = density
            if cell_density_stddevs is None:
                densities.at[child_region_name, "inhibitory_neuron_standard_deviation"] = density
            else:
                densities.at[
                    child_region_name, "inhibitory_neuron_standard_deviation"
                ] = cell_density_stddevs[child_region_name]

    excitatory_mask = homogenous_regions["cell_type"] == "excitatory"

    for region_name in homogenous_regions[excitatory_mask]["brain_region"]:
        desc_id_set = hierarchy_info.at[region_name, "descendant_ids"]
        id_mask = [id_ in desc_id_set for id_ in hierarchy_info["id"]]
        for child_region_name in hierarchy_info[id_mask].index:
            densities.at[child_region_name, "inhibitory_neuron"] = 0.0
            densities.at[child_region_name, "inhibitory_neuron_standard_deviation"] = 0.0


def compute_average_intensity(
    intensity: FloatArray, volume_mask: BoolArray, slices: Optional[list[int]] = None
) -> float:
    """
    Compute the average of `intensity` within the volume defined by `volume_mask`.

    If `slices` is a non-empty list of slice indices along the x-axis, then the average is
    restricted to the subvolume of `volume_mask` enclosed by these slices. Slices indices outside
    the `volume_mask` will be ignored.

    Args:
        intensity: a float array of shape (W, H, D) where W, H and D are integer dimensions.
        volume_mask: boolean mask of the same shape as `instensity` defining the volume on which
            the average intensity is computed.
        slices (optional): list of integer indices along the x-axis of `intensity`. Indices range in
            [0, W - 1] with W = `intensity.shape[0]`. Defaults to None, in which case the average is
            taken over `volume_mask` without restriction.
    Returns:
        the average intensity over `volume_mask`, restricted to the specified `slices` if
        any. If the restricted subvolume is empty (e.g., the volume does not intersect any slice),
        the returned value is 0.0.
    """

    if slices is None:
        restricted_mask = volume_mask
    else:
        restricted_mask = np.zeros_like(volume_mask)
        # remove slices indices outside the volume_mask
        slices_ = [slice_ for slice_ in slices if 0 <= slice_ < volume_mask.shape[0]]
        restricted_mask[slices_] = True
        restricted_mask = np.logical_and(restricted_mask, volume_mask)

    if np.any(restricted_mask):
        return np.mean(intensity[restricted_mask])

    return 0.0


def _add_depths(region_map_df):
    """Rdd a `depth` column to the `region_map_df` with how deep it is within the hierarchy."""
    region_map_df["depth"] = 0
    parent_ids = [
        -1,
    ]
    depth = 1
    while parent_ids:
        children_ids = region_map_df[np.isin(region_map_df["parent_id"], list(parent_ids))].index
        region_map_df.loc[children_ids, "depth"] = depth
        parent_ids = set(children_ids)
        depth += 1


def _fill_densities(region_map, region_map_df, df):
    """Fill all `voxel_count` and `density` for each region within `def`.

    This assumes the leaf nodes (ie: the deapest nodes in region_map) have had
    their values filled in, and thus each of their parents can recursively be
    filled in.
    """
    order_idx = np.argsort(region_map_df.depth.to_numpy())

    ids = region_map_df.index[order_idx[::-1]]
    for id_ in ids:
        if id_ not in region_map._children:  # pylint: disable=protected-access
            continue

        children = region_map._children[id_]  # pylint: disable=protected-access

        if not children:
            continue

        voxel_count = df.loc[children, "voxel_count"]
        count = voxel_count.sum()
        if count:
            df.loc[id_, "voxel_count"] = count
            df.loc[id_, "density"] = np.average(df.loc[children, "density"], weights=voxel_count)


def _apply_density_slices(gene_marker_volumes):
    """For each marker in `gene_marker_volumes`, apply the slice mask to the indensities volume."""
    ret = {}
    for marker, intensity in gene_marker_volumes.items():
        intensity, slices = intensity["intensity"], intensity["slices"]

        if slices is None:
            mask = np.ones_like(intensity, dtype=bool)
        else:
            mask = np.zeros_like(intensity, dtype=bool)
            slices_ = [slice_ for slice_ in slices if 0 <= slice_ < mask.shape[0]]
            mask[slices_] = True

        ret[marker] = {"intensity": intensity, "slices": slices, "mask": mask}

    return ret


def _compute_per_marker_intensity(annotation, gene_marker_volumes):
    """Compute the average intensity for `id_`, for all makers in `gene_marker_volumes`"""
    vtiv = voxel_data.ValueToIndexVoxels(annotation)

    count_and_density = []
    for id_ in vtiv.values:
        if id_ == 0:
            continue

        voxel_ids = vtiv.value_to_1d_indices(id_)

        res = []
        for marker, intensity in gene_marker_volumes.items():
            mask_voxels = vtiv.ravel(intensity["mask"])[voxel_ids]

            count = mask_voxels.sum()

            if count <= 0:
                continue

            mean_density = vtiv.ravel(intensity["intensity"])[voxel_ids][mask_voxels].sum() / count

            if mean_density == 0.0:
                L.warning("Mean density for id=%s and marker=%s", id_, marker)

            count_and_density.append((marker.lower(), id_, count, mean_density))

    res = (
        pd.DataFrame(count_and_density, columns=["marker", "id", "voxel_count", "density"])
        .set_index("id")
        .pivot(columns="marker")
        .swaplevel(axis=1)
    )
    return res


def compute_average_intensities(
    annotation: AnnotationT,
    gene_marker_volumes: MarkerVolumes,
    hierarchy_info: pd.DataFrame,
    region_map,
) -> pd.DataFrame:
    """
    Compute the average marker intensity of every region in `hierarchy_info` for every marker
    of `gene_marker_volumes`.

    If a region does not intersect any of the slices of a gene marker volume, the average density
    of the marked cell type of this region is set with np.nan.

    Args:
        annotation: int array of shape (W, H, D) holding the annotation of the whole
            brain model. (The integers W, H and D are the dimensions of the array).
        gene_marker_volumes: dict of the form {
                "gad67": {"intensity": <array>, "slices": <list>},
                "pv": {"intensity": <array>, "slices": <list>},
                ...
                }
            where each intensity array is of shape (W, H, D) and where the items of each slice
            list range in [0, W - 1].
        hierarchy_info: data frame with colums "descendant_ids" (set[int]) and "brain_region"
            (str) and whose index is a list of region ids.
            See :fun:`atlas_densities.densities.utils.get_hierarchy_info`.

    Returns:
        A data frame of the following form (values are fake):
                    gad67  pv   sst    vip
        Isocortex   1.5    1.1  0.2    12.0
        Cerebellum  2.5    0.9  0.1    11.0
        ...         ...   ...    ...    ...
        The index of the data frame is the list of regions `hierarchy_info["brain_region"]`.
        The column labels are the keys of `gene_marker_volumes` in lower case.
    """
    gene_marker_volumes = _apply_density_slices(gene_marker_volumes)

    intensity = _compute_per_marker_intensity(annotation, gene_marker_volumes)

    region_map_df = region_map.as_dataframe()
    _add_depths(region_map_df)

    result = pd.DataFrame(
        data=np.nan,
        index=hierarchy_info.index,
        columns=[marker_name.lower() for marker_name in gene_marker_volumes],
    )
    result["brain_region"] = region_map_df.loc[result.index].name

    for marker in gene_marker_volumes:
        df = pd.DataFrame(data=0.0, index=result.index, columns=["voxel_count", "density"])
        df.update(intensity[marker.lower()])
        _fill_densities(region_map, region_map_df, df)
        result[marker.lower()] = df["density"]

    result = result.set_index("brain_region")
    return result


def linear_fitting_xy(
    xdata: list[float],
    ydata: list[float],
    sigma: Union[list[float], FloatArray],
    min_data_points: int,
) -> dict[str, float]:
    """
    Compute the coefficient of the linear least-squares regression of the point cloud
    (`xdata`, `ydata`) and its standard deviation.

    Args:
        xdata: list of float values
        ydata: list of float values with the same length as `xdata`.
        sigma: list of non-negative float values with the same length as `xdata`.
            These are the standard deviations of the values in `ydata`.
            Zero values are replaced by the least positive value if it exists,
            otherwise by 1.0.
        min_data_points: minimum number of datapoints required for running
            the linear regression. If the number of datapoints is less than
            min_data_points then no fitting is done, and np.nan values are
            returned.

    Returns:
        a dict of the form {"coefficient": <float>,
                            "standard_deviation": <float>,
                            "r_square": <float>}.

    Raises:
       AtlasDensitiesError if some of the `sigma`values are negative.
    """
    if len(xdata) < min_data_points:
        return {"coefficient": np.nan, "standard_deviation": np.nan, "r_square": np.nan}

    sigma = np.array(sigma)
    if np.any(sigma < 0.0):
        raise AtlasDensitiesError(
            "Some sigma value provided for linear fitting is negative. "
            "Expected: non-negative values."
        )

    # A zero sigma value forces curve_fit to return always 1.0 as coefficient independently of the
    # input point cloud. We replace zero by half of the least positive sigma value if it exists,
    # 1.0 otherwise.
    zero_mask = np.isclose(sigma, 0.0)
    if np.all(zero_mask):
        sigma[zero_mask] = 1.0
    else:
        least_positive = np.min(sigma[~zero_mask])
        sigma[zero_mask] = least_positive / 2.0

    def _optimize_func(x, coefficient):
        return coefficient * np.array(x)

    parameters = curve_fit(
        _optimize_func,
        xdata=xdata,
        ydata=ydata,
        sigma=sigma,
        absolute_sigma=True,
    )
    ss_reg = np.sum((_optimize_func(xdata, parameters[0][0]) - np.mean(ydata)) ** 2)
    ss_tot = np.sum((np.array(ydata) - _optimize_func(xdata, parameters[0][0])) ** 2) + ss_reg
    L.debug("Length of xdata = %s.  The ss_reg is %s and ss_tot is %s", xdata, ss_reg, ss_tot)
    # if total sum of square is null, no variance can be explained by the fitting
    return {
        "coefficient": parameters[0][0],
        "standard_deviation": np.sqrt(parameters[1][0][0]),
        "r_square": (ss_reg / ss_tot) if ss_tot > 0 else np.nan,
    }


FittingData = Dict[str, Dict[str, Dict[str, float]]]


def compute_fitting_coefficients(
    groups: dict[str, set[str]],
    average_intensities: pd.DataFrame,
    densities: pd.DataFrame,
    min_data_points: int,
) -> FittingData:
    """
    Compute the linear fitting coefficient of the cloud of 2D points (average marker intensity,
    average cell density) in each region group of `groups`.

    The linear regression is a linear least-squares regression with y-intercept equal to zero,
    i.e., we fit a function x -> c * x with parameter c.

    Note: the 2D points such that average marker intensity or the average cell density is 0.0
    are excluded from the fitting.

    Args:
        groups: dict whose keys are region group names (e.g., "Isocortex", "Cerebellum", or "Rest")
            and whose values are sets containing the names of the regions in each group.
        average_intensities: data frame returned by
          :fun:`atlas_densities.densities.fitting.compute_average_intensities`.
        densities: data frame returned by
          :fun:`atlas_densities.densities.fitting.create_dataframe_from_known_densities`.

    Returns:
        dict of the form (values are fake):
        {
            "Isocortex" : {
                "gad67": { "coefficient": 0.85, "standard_deviation": 0.15, "r_square": 0.75},
                "pv": { "coefficient": 1.00, "standard_deviation": 0.5, "r_square": 0.66},
                "sst": { "coefficient": 1.07, "standard_deviation": 0.75, "r_square": 0.80},
                "vip": { "coefficient": 0.95, "standard_deviation": 0.15, "r_square": 0.90},
            },
            "Cerebellum" : {
                "gad67": { "coefficient": 0.75, "standard_deviation": 0.15, "r_square": 0.55},
                "pv": { "coefficient": 1.10, "standard_deviation": 0.55, "r_square": 0.40},
                "sst": { "coefficient": 1.20, "standard_deviation": 0.45, "r_square": 0.58},
                "vip": { "coefficient": 0.95, "standard_deviation": 0.15, "r_square": 0.77},
            },
           ...
        }
        The keys of the dict are the names of the region groups where separate fittings are
        performed. A "coefficient" value is the value of the linear regression coefficient for a
        given region group and a given cell type.
        The "standard_deviation" value is the standard deviation of the coefficient value.
        The "r_square" value is the coefficient of determination of the coefficient value.
    """
    # pylint: disable=too-many-locals
    if len(densities.index) != len(average_intensities.index) or np.any(
        densities.index != average_intensities.index
    ):
        raise AtlasDensitiesError(
            "The data frames `densities` and `average_intensities` have a different index."
        )

    cell_types = [marker + "+" for marker in average_intensities.columns]
    diff_types = set(cell_types) - set(densities.columns)
    if len(diff_types) > 0:
        diff_types = {cell_type[:-1] for cell_type in diff_types}
        raise AtlasDensitiesError(
            f"Some marker names are not represented in the densities data"
            f" frame by the corresponding marked cell types: {diff_types}."
        )

    result = {
        group_name: {
            cell_type: {"coefficient": np.nan, "standard_deviation": np.nan, "r_square": np.nan}
            for cell_type in cell_types
        }
        for group_name in groups
    }

    clouds: dict[str, dict[str, dict[str, list[float]]]] = {
        group_name: {cell_type: {"xdata": [], "ydata": [], "sigma": []} for cell_type in cell_types}
        for group_name in groups
    }
    L.info("Computing fitting coefficients in %d groups ...", len(groups))
    for group_name in groups:
        L.info(
            "Building regression data in group %s for %d regions and %d cell types ...",
            group_name,
            len(groups[group_name]),
            len(cell_types),
        )
        for region_name in tqdm(groups[group_name]):
            if region_name in average_intensities.index:
                for cell_type in cell_types:
                    intensity = average_intensities.at[region_name, cell_type[:-1]]
                    density = densities.at[region_name, cell_type]
                    if not (
                        np.isnan(density)
                        or np.isnan(intensity)
                        or intensity == 0.0
                        or density == 0.0
                    ):
                        assert_msg = f"in region {region_name} for cell type {cell_type}"
                        assert intensity >= 0.0, "Negative average intensity " + assert_msg
                        assert density >= 0.0, "Negative density " + assert_msg
                        standard_deviation = densities.at[
                            region_name, cell_type + "_standard_deviation"
                        ]
                        assert not np.isnan(standard_deviation), (
                            "NaN standard deviation " + assert_msg
                        )
                        assert standard_deviation >= 0.0, (
                            "Negative standard deviation " + assert_msg
                        )

                        clouds[group_name][cell_type]["xdata"].append(intensity)
                        clouds[group_name][cell_type]["ydata"].append(density)
                        clouds[group_name][cell_type]["sigma"].append(standard_deviation)

        L.info("Computing regression coefficients for %d cell types ...", len(cell_types))
        for cell_type in tqdm(cell_types):
            cloud = clouds[group_name][cell_type]
            L.debug(
                "The length of training data for %s and %s is %s",
                group_name,
                cell_type,
                cloud["xdata"],
            )
            result[group_name][cell_type] = linear_fitting_xy(
                cloud["xdata"], cloud["ydata"], cloud["sigma"], min_data_points
            )
            if np.isnan(result[group_name][cell_type]["coefficient"]):
                warnings.warn(
                    f"The region group {group_name} has a NaN fitting coefficient for the cell "
                    f"type {cell_type} due to missing input data for regression.",
                    AtlasDensitiesWarning,
                )

    return result


def fit_unknown_densities(
    groups: dict[str, set[str]],
    average_intensities: pd.DataFrame,
    densities: pd.DataFrame,
    fitting_coefficients: FittingData,
) -> None:
    """
    Estimate unknown cell densities by means of a linear fitting.

    The data frame `densities` is modified in-place.
    Known values (i.e. non-NaN) are left unchanged.

    Args:
        groups: dict whose keys are region group names (e.g., "Isocortex", "Cerebellum", or "Rest")
            and whose values are sets containing the names of the regions in each group.
        average_intensities: data frame returned by
          :fun:`atlas_densities.densities.fitting.compute_average_intensities`.
        densities: data frame returned by
          :fun:`atlas_densities.densities.fitting.create_dataframe_from_known_densities`.
        fitting_coefficients: dict returned by
            fun:`atlas_densities.densities.fitting.compute_fitting_coefficients`.
    """

    for group_name, region_names in groups.items():
        for region_name in region_names:
            for marker in average_intensities.columns:
                if region_name in average_intensities.index:
                    _apply_fitting(
                        region_name,
                        marker,
                        group_name,
                        average_intensities,
                        densities,
                        fitting_coefficients,
                    )


def _apply_fitting(
    region_name, marker, group_name, average_intensities, densities, fitting_coefficients
):
    """
    Apply the fitting function to a given region and marker based on the corresponding marker
    intensity.

    Args:
        region_name: region name
        marker: ISH marker name
        group_name: fitting group name
        average_intensities: pandas data frame returned by
          :fun:`atlas_densities.densities.fitting.compute_average_intensities`.
        densities: data frame returned by
          :fun:`atlas_densities.densities.fitting.create_dataframe_from_known_densities`.
        fitting_coefficients: dict returned by
          :fun:`atlas_densities.densities.fitting.compute_fitting_coefficients`.
    """

    intensity = average_intensities.at[region_name, marker]
    cell_type = marker + "+"
    if np.isnan(densities.at[region_name, cell_type]) and not np.isnan(intensity):
        fitting = fitting_coefficients[group_name][cell_type]
        if np.isnan(fitting["coefficient"]):
            warnings.warn(
                f"Interpolating density of region {region_name} for cell type "
                f"{cell_type} using NaN fitting coefficient of region group "
                f"{group_name}.",
                AtlasDensitiesWarning,
            )
        fitted_value = fitting["coefficient"] * intensity
        standard_deviation = fitted_value * fitting["standard_deviation"]
        densities.at[region_name, cell_type] = fitted_value
        densities.at[region_name, cell_type + "_standard_deviation"] = standard_deviation


def _check_homogenous_regions_sanity(homogenous_regions: pd.DataFrame) -> None:
    """
    Raise an error if `homogenous_regions` has unexpected entries.
    """

    actual_cell_types = set(homogenous_regions["cell_type"])
    supported_cell_types = {"inhibitory", "excitatory"}
    diff = actual_cell_types - supported_cell_types
    if len(diff) > 0:
        raise AtlasDensitiesError(f"`homogenous_regions` has unexpected cell types: {diff}")


def _check_average_densities_sanity(average_densities: pd.DataFrame) -> None:
    """
    Raise an error if `average_densities` has NaN or negative entries.
    """
    actual_measurement_types = set(average_densities["measurement_type"])
    diff = actual_measurement_types - {"cell density"}
    if len(diff) > 0:
        raise AtlasDensitiesError(f"`average_densities` has unexpected measurement types: {diff}")

    nan_mask = (
        average_densities["measurement"].isna() | average_densities["standard_deviation"].isna()
    )

    if np.any(nan_mask):
        raise AtlasDensitiesError(
            f"`average_densities` has a NaN measurement or a NaN standard deviation for the "
            f"following entries: {average_densities[nan_mask]}"
        )

    negative_mask = (average_densities["measurement"] < 0.0) | (
        average_densities["standard_deviation"] < 0.0
    )

    if np.any(negative_mask):
        raise AtlasDensitiesError(
            f"`average_densities` has a negative measurement or a negative standard deviation "
            f"for the following entries: {average_densities[negative_mask]}"
        )


def _get_group_names(region_map: RegionMap, group_ids_config: dict) -> dict[str, set[str]]:
    """
    Get AIBS names for regions in several region groups of interest.

    Args:
        region_map: object to navigate the mouse brain regions hierarchy
            (instantiated from AIBS 1.json).
        group_ids_config: mapping of regions to their constituent ids

    Returns:
        A dictionary whose keys are region group names and whose values are
        sets of brain region names.
    """

    group_ids = utils.get_group_ids(region_map, group_ids_config)

    # Make the Rest group stable undertaking descendants
    # The name of any ascendant region of the Cerebellum and
    #    Isocortex groups are removed from the names of the Rest group. This makes sure that
    #    the set of names of the Rest group is closed under taking descendants.
    isocortex_id = region_map.find("Isocortex", attr="name").pop()
    cerebellum_id = region_map.find("Cerebellum", attr="name").pop()
    ascendant_ids = set(region_map.get(isocortex_id, attr="id", with_ascendants=True)) | set(
        region_map.get(cerebellum_id, attr="id", with_ascendants=True)
    )
    group_ids["Rest"] -= ascendant_ids

    return {
        group_name: {region_map.get(id_, attr="name") for id_ in group_ids[group_name]}
        for group_name in ["Cerebellum group", "Isocortex group", "Rest"]
    }


def _get_group_region_names(groups):
    """
    Create a set of the union of AIBS region names that belong to the given groups.

    Args:
        groups: dictionary from group name to list of region name.

    Returns:
        A list of region names.
    """

    return list(set(it.chain.from_iterable(groups.values())))


def linear_fitting(  # pylint: disable=too-many-arguments
    region_map: RegionMap,
    annotation: AnnotationT,
    neuron_density: FloatArray,
    gene_marker_volumes: MarkerVolumes,
    average_densities: pd.DataFrame,
    homogenous_regions: pd.DataFrame,
    cell_density_stddevs: Optional[dict[str, float]] = None,
    region_name: str = "root",
    group_ids_config: dict | None = None,
    min_data_points: int = 5,
) -> pd.DataFrame:
    """
    Estimate the average densities of every region in `region_map` using a linear fitting
    of the 2D points (average marker intensity, average cell density) for the markers in
    `gene_marker_volumes` and the average cell densities of `average_densities`.

    A cell type is named after a marker in `gene_marker_volumes` to which the corresponding cell
    population reacts, e.g., gad67+ (=inhibitory neurons), pv+, sst+ or vip+.

    Note: Due to insufficient coverage of regions by gene maker slices, the average
    densities of several regions (for several cell types) cannot be computed and are
    set to NaN.

    Args:
        region_map: RegionMap object to navigate the brain region hierarchy. Used to
            define the regions whose cell densities will be computed.
        annotation: int array of shape (W, H, D) holding the annotation of the whole
            brain model. (The integers W, H and D are the dimensions of the array).
        neuron_density: non-negative float array of shape (W, H, D) where W, H and D are integer
            dimensions. This array holds the volumetric neuron density of the brain model expressed
            in number of neurons per mm^3.
        gene_marker_volumes: dict of the form {
                  "gad67": {"intensity": <array>, "slices": <list>},
                  "pv": {"intensity": <array>, "slices": <list>},
                  ...
                }
            where each intensity array is of shape (W, H, D) and where the items of each slice
            list range in [0, W - 1]. The keys "gad67", "pv", ... are gene marker names and the
            corresponding reacting cell populations are denoted "gad67+", "pv+", ... in the output
            data frame.
        average_densities: a data frame whose columns are described in
            :func:`atlas_densities.app.densities.compile_measurements` and where all
            measurements are of type "cell density" in number of cells per mm^3.
        homogenous_regions: data frame with two columns, brain_region and cell_type.
            The brain_region column holds region names and the values of the cell_type column are
            "inhibitory" or "excitatory" depending on whether every neuron of the region is
            inhibitory or excitatory. Note that the "inhibitory" cell type will supersede by
            its synonym "gad67+" in the output data frame.
        cell_density_stddevs: dict whose keys are brain regions names and whose values are
            standard deviations of average cell densities of the corresponding regions.
        region_name: (str) name of the root region of interest
        group_ids_config: mapping of regions to their constituent ids
        min_data_points: minimum number of datapoints required for running
            the linear regression. If the number of datapoints is less than
            min_data_points then no fitting is done, and np.nan values are
            returned.

    Returns:
        tuple (densities, fitting_coefficients)
            densities:
                data frame holding an average density estimate for every region in `region_map` and
                every cell type marked by the gene markers of `gene_marker_volumes`.
                The output data frame has a column `standard_deviation` for every gene marker.
            fitting_coefficients: dict returned by
                :fun:`atlas_densities.densities.fitting.compute_fitting_coefficients`.
    """
    # pylint: disable=too-many-locals
    assert group_ids_config is not None
    L.info("Checking input data frames sanity ...")
    _check_average_densities_sanity(average_densities)
    _check_homogenous_regions_sanity(homogenous_regions)

    hierarchy_info = utils.get_hierarchy_info(region_map, root=region_name)
    remove_unknown_regions(average_densities, region_map, annotation, hierarchy_info)
    remove_unknown_regions(homogenous_regions, region_map, annotation, hierarchy_info)

    L.info("Creating a data frame from known densities ...")
    densities = create_dataframe_from_known_densities(
        hierarchy_info["brain_region"].to_list(), average_densities
    )

    L.info("Filling homogenous regions data ...")
    fill_in_homogenous_regions(
        homogenous_regions,
        annotation,
        neuron_density,
        densities,
        hierarchy_info,
        cell_density_stddevs,
    )

    # From now on, cell populations are named using gene markers that uniquely
    # identifies them.
    densities.rename(
        columns={
            "inhibitory_neuron": "gad67+",
            "inhibitory_neuron_standard_deviation": "gad67+_standard_deviation",
        },
        inplace=True,
    )

    L.info("Computing average intensities ...")
    groups = _get_group_names(region_map, group_ids_config)
    # Limit the fitting to the regions groups.
    indexes = np.isin(
        hierarchy_info["brain_region"].to_numpy(),
        _get_group_region_names(groups),
        invert=True,
    )
    average_intensities = compute_average_intensities(
        annotation,
        gene_marker_volumes,
        hierarchy_info.drop(hierarchy_info.index[indexes]),
        region_map,
    )

    L.info("Computing fitting coefficients ...")
    fitting_coefficients = compute_fitting_coefficients(
        groups,
        average_intensities,
        densities.drop(densities.index[indexes]),
        min_data_points=min_data_points,
    )
    L.info("Fitting unknown average densities ...")
    fit_unknown_densities(groups, average_intensities, densities, fitting_coefficients)

    return densities, fitting_coefficients
