"""
Functions to compute the average cell densities of AIBS mouse brain regions for several types
including PV+, SST+, VIP+ and the inhibitory neurons (GAD67+) based on experimental measurements.

Densities are either obtained from a density measurement reported in a scientific paper or are
inferred from one or more measurements (e.g., cell counts, cell proportions, volumes, ...) and the
the volume of the annotated AIBS mouse atlas.

Measurements are taken from a compilation made by Dimitri Rodarie (BBP) which collects data acorss
more than 40 scientific articles.

Densities are expressed in number of cells per mm^3.
"""
from typing import Set, Tuple, Union

import numpy as np
import pandas as pd
from atlas_commons.typing import AnnotationT, FloatArray
from voxcell import RegionMap  # type: ignore

from atlas_densities.densities.utils import compute_region_volumes, get_hierarchy_info


def get_parent_region(region_name: str, region_map: RegionMap) -> Union[str, None]:
    """
    Return the name of the parent region of `region_name` in the hierarchy of `region_map`

    Args:
        region_name: Name of the region whose parent (immediate superset) is queried.
        region_map: RegionMap object to navigate the brain regions hierarchy.

    Returns:
        Name of the parent region of `region_name` in the hierarchy of `region_map`.
        Example: get_parent_region("Isocortex", region_map) returns "Cortical plate" if
        `region_map` has been instantiated from AIBS 1.json.
        If `region_name` is the region at the root of the hierarchy tree, the function returns None.
    """

    id_ = region_map.find(region_name, attr="name").pop()
    parent_id = region_map.get(id_, attr="parent_structure_id")

    return region_map.get(parent_id, attr="name") if parent_id is not None else None


def compute_region_densities(
    annotation: AnnotationT,
    cell_density: FloatArray,
    hierarchy_info: "pd.DataFrame",
) -> "pd.DataFrame":
    """
    Compute the average cell density of every annotated brain region of `annotation` whose
    id can be found by `region_map` based on the volumetric `cell_density`.

    Args:
        annotation: int array of shape (W, H, D) holding the annotation of the whole AIBS
            mouse brain. (The integers W, H and D are the dimensions of the array).
        cell_density: float array of shape (W, H, D) holding the overall volumetric cell density
            of the AIBS mouse brain. A voxel value represents the average cell density in that
            voxel expressed in number of cells per mm^3.
        hierarchy_info: data frame returned by
            :func:`atlas_densities.densities.utils.get_hierarchy_info`.

    Returns:
        DataFrame of the following form (values are fake):
             brain region                    cell density
        5    Basic cell groups and regions   0.005
        123  Cerebrum                        0.001
        ...  ...                             ...
        The index is the sorted list of all region identifiers.
    """
    densities = []
    for set_ in hierarchy_info["descendant_id_set"]:
        mask = np.isin(annotation, list(set_))
        densities.append(np.sum(cell_density[mask]) / np.count_nonzero(mask))

    return pd.DataFrame(
        {"brain_region": hierarchy_info["brain_region"], "cell density": densities},
        index=hierarchy_info.index,
    )


def cell_count_to_density(
    measurements: "pd.DataFrame", volumes: "pd.DataFrame", region_map: RegionMap
) -> None:
    """
    Turn the measurements of type "cell count" into measurements of type "cell density".

    The data frame `measurements` is modified in place.

    Args:
        measurements: dataframe whose columns are described in
            :func:`atlas_densities.app.densities.compile_measurements`.
        volumes: data frame returned by
            :func:`atlas_densities.densities.cell_densities_from_measurements.
            compute_region_volumes`.
        region_map: RegionMap object to navigate the brain regions hierarchy.
    """

    def find_parent_volume(
        row: "pd.Series", measurements: "pd.DataFrame", region_map: RegionMap
    ) -> Union["pd.Series", None]:
        """
        Find the among the measurements of type volume with source `row["source"]` if there is a
        record for the parent region of `row["brain_region"]`.

        Return None if no such record exists.
        """
        mask = (measurements["source_title"] == row["source_title"]) & (
            measurements["measurement_type"] == "volume"
        )
        volumes_from_same_source = measurements[mask]
        parent_region = get_parent_region(row["brain_region"], region_map)
        if parent_region is None:
            return None
        for _, measurement in volumes_from_same_source.iterrows():
            if measurement["brain_region"] == parent_region:
                return measurement

        return None

    count_mask = measurements["measurement_type"] == "cell count"
    cell_counts = measurements[count_mask]
    volumes = volumes.set_index("brain_region")
    for index, row in cell_counts.iterrows():
        # Some reported cell counts are provided with the volume of the parent region
        # but not the volume of the region itself. The computation below tries to make
        # sense of such a situation.
        # Example: cell counts and volumes of Figure 2d in "Ethanol exposure during development
        # reduces GABAergic/glycinergic neuron numbers and lobule volumes in the mouse cerebellar
        # vermis" DOI: 10.1016/j.neulet.2016.08.039
        name = row["brain_region"]
        parent_volume = find_parent_volume(row, measurements, region_map)
        if parent_volume is not None:
            parent_name = parent_volume["brain_region"]
            ratio = volumes.loc[name, "volume"] / volumes.loc[parent_name, "volume"]
            cell_counts.at[index, "measurement"] /= parent_volume["measurement"] * ratio
            cell_counts.at[index, "standard_deviation"] /= parent_volume["measurement"] * ratio
        else:
            cell_counts.at[index, "measurement"] /= volumes.loc[name, "volume"]
            cell_counts.at[index, "standard_deviation"] /= volumes.loc[name, "volume"]
        cell_counts.at[index, "measurement_type"] = "cell density"
        cell_counts.at[index, "measurement_unit"] = "number of cells per mm^3"

    measurements[count_mask] = cell_counts


def cell_proportion_to_density(
    measurements: "pd.DataFrame",
    cell_densities: "pd.DataFrame",
    measurement_type: str,
) -> None:
    """
    Turn the measurements of type "cell proportion" or "neuron proportion" into measurements of
    type "cell density".

    The data frame `measurements` is modified in place.

    Args:
        measurements: dataframe whose columns are described in
            :func:`atlas_densities.app.densities.compile_measurements`.
        cell_densities: data frame returned by
            :func:`atlas_densities.densities.cell_densities_from_measurements.
            compute_region_densities`.
        measurement_type: Either "cell proportion" or "neuron proportion". The type of measurement
            to turn into a cell density.
    """

    proportion_mask = measurements["measurement_type"] == measurement_type
    cell_proportions = measurements[proportion_mask]
    cell_densities = cell_densities.set_index("brain_region")
    for index, row in cell_proportions.iterrows():
        density = cell_densities.loc[row["brain_region"], "cell density"]
        cell_proportions.at[index, "measurement"] *= density
        cell_proportions.at[index, "standard_deviation"] *= density
        cell_proportions.at[index, "measurement_type"] = "cell density"
        cell_proportions.at[index, "measurement_unit"] = "number of cells per mm^3"

    measurements[proportion_mask] = cell_proportions


def get_average_voxel_count_per_slice(
    id_set: Set[int], annotation: AnnotationT, thickness: int
) -> float:
    """
    Returns the average number of voxels with an id in `id_set` over the x-slices of `annotation`.

    Slices are cross-sections orthogonal the x-axis (axis = 0), often referred to as coronal slices
    due to the orientation of the AIBS annotated brain volumes.

    Args:
        id_set: set of integer identifiers defining a region in `annotation`.
        annotation: int array of shape (W, H, D) holding the annotation of the whole AIBS
            mouse brain. (The integers W, H and D are the dimensions of the array).
        thickness: thickness of an x-slice in number of voxels.

    Returns:
        average number of voxels with ids in `id_set` and lying inside a slice of `annotation` with
        thickness `thickness` voxels.

    """
    region_mask = np.isin(annotation, list(id_set))
    x_indices = np.where(region_mask)[0]
    _, counts = np.unique(x_indices, return_counts=True)
    slice_counts = [np.sum(counts[i : i + thickness]) for i in range(len(counts) - thickness + 1)]

    return np.mean(slice_counts)


def cell_count_per_slice_to_density(
    measurements: "pd.DataFrame",
    annotation: AnnotationT,
    voxel_dimensions: Tuple[float, float, float],
    voxel_volume: float,
    hierarchy_info: "pd.DataFrame",
) -> None:
    """
    Turn the measurements of type "cell count per slice" into measurements of
    type "cell density".

    The data frame `measurements` is modified in place.

    Args:
        measurements: dataframe whose columns are described in
            :func:`atlas_densities.app.densities.compile_measurements`.
        annotation: int array of shape (W, H, D) holding the annotation of the whole AIBS
            mouse brain. (The integers W, H and D are the dimensions of the array).
        voxel_dimensions:
        voxel_volume: volume in mm^3 of a voxel in any of the volumetric input arrays.
        hierarchy_info: data frame returned by
            :func:`atlas_densities.densities.utils.get_hierarchy_info`.
    """

    # Thickness of a coronal slice in number of voxels
    # assuming that voxel dimensions are expressed in micrometers (um)
    thickness = round(
        50.0 / voxel_dimensions[0]
    )  # Expected: 2 resp. 5 for the usual 25um and 10um AIBS resolutions.
    mask_50um = (measurements["measurement_type"] == "cell count per slice") & (
        measurements["measurement_unit"] == "number of cells per 50-micrometer-thick slice"
    )
    cell_counts_per_slice = measurements[mask_50um]
    hierarchy_info = hierarchy_info.set_index("brain_region")
    for index, row in cell_counts_per_slice.iterrows():
        id_set = hierarchy_info.loc[row["brain_region"], "descendant_id_set"]
        average_slice_volume = voxel_volume * get_average_voxel_count_per_slice(
            id_set, annotation, thickness
        )
        cell_counts_per_slice.at[index, "measurement"] /= average_slice_volume
        cell_counts_per_slice.at[index, "standard_deviation"] /= average_slice_volume
        cell_counts_per_slice.at[index, "measurement_type"] = "cell density"
        cell_counts_per_slice.at[index, "measurement_unit"] = "number of cells per mm^3"

    measurements[mask_50um] = cell_counts_per_slice


def measurement_to_average_density(
    region_map: RegionMap,
    annotation: AnnotationT,
    voxel_dimensions: Tuple[float, float, float],
    voxel_volume: float,
    cell_density: FloatArray,
    neuron_density: FloatArray,
    measurements: "pd.DataFrame",
) -> "pd.DataFrame":
    """
    Compute average cell densities in AIBS brain regions based on experimental `measurements`.

    The input measurements have the types listed in MEASUREMENT_TYPES and MEASUREMENT_UNITS,
    see :mod:`densities.utils`.

    If for a given brain region, several cell density measurements are available in `measurements`
    (or if several cell density computations are possible from measurements of different
    articles), the output cell density of the region is the average of the possible cell densities.

    The region names in `measurements` which are not compliant with the AIBS nomenclature (1.json)
    are ignored.

    Args:
        region_map: RegionMap object to navigate the brain regions hierarchy.
        annotation: int array of shape (W, H, D) holding the annotation of the whole AIBS
            mouse brain. (The integers W, H and D are the dimensions of the array).
        cell_density: non-negative float array of shape (W, H, D) holding the overall volumetric
            cell density of the AIBS mouse brain. A voxel value represents the average cell density
            in that voxel expressed in number of cells per mm^3.
        voxel_volume: volume in mm^3 of a voxel in any of the volumetric input arrays.
            This is (25 * 1e-6) ** 3 for an AIBS atlas nrrd file with 25um resolution.
        neuron_density: float array of shape (W, H, D) holding the overall volumetric neuron
            density of the AIBS mouse brain. A voxel value represents the average neuron density
            in that voxel expressed in number of neurons per mm^3.
        measurements: dataframe whose columns are described in
            :func:`atlas_densities.app.densities.compile_measurements`.

    Returns:
        dataframe of the same format as `measurements` but where all measurements of type
        "cell count", "cell proportion" or "neuron proportion" have been turned in measurements of
        type "cell density". Densities are expressed in number of cells per mm^3.
    """

    # Filter out non-AIBS compliant region names
    hierarchy_info = get_hierarchy_info(region_map)
    indices = measurements.index[~measurements["brain_region"].isin(hierarchy_info["brain_region"])]
    measurements = measurements.drop(indices)

    # Replace NaN standard deviations by measurement values
    nan_mask = measurements["standard_deviation"].isna()
    measurements.loc[nan_mask, "standard_deviation"] = measurements["measurement"][nan_mask]

    # Compute the volumes of every AIBS brain region
    volumes = compute_region_volumes(annotation, voxel_volume, hierarchy_info)
    cell_count_to_density(measurements, volumes, region_map)

    # Compute the average cell density of every AIBS brain region according to the content of
    # cell_density
    cell_densities = compute_region_densities(annotation, cell_density, hierarchy_info)

    cell_proportion_to_density(measurements, cell_densities, measurement_type="cell proportion")

    # Compute the average neuron densities of every AIBS brain region according to the content of
    # neuron_density
    neuron_densities = compute_region_densities(annotation, neuron_density, hierarchy_info)

    cell_proportion_to_density(measurements, neuron_densities, measurement_type="neuron proportion")

    cell_count_per_slice_to_density(
        measurements, annotation, voxel_dimensions, voxel_volume, hierarchy_info
    )

    return measurements


def remove_non_density_measurements(measurements: "pd.DataFrame") -> None:
    """
    Remove every measurement which is not of type "cell density".

    The `measurements` data frame is mutated in-place.

    Args:
        measurements: dataframe whose columns are described in
            :func:`atlas_densities.app.densities.compile_measurements`.

    """
    indices = measurements.index[measurements["measurement_type"] != "cell density"]
    measurements.drop(indices, inplace=True)
    measurements.reset_index(drop=True, inplace=True)
