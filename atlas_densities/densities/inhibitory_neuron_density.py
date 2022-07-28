"""Functions to compute inhibitory neuron density."""

from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import numpy as np
from atlas_commons.typing import AnnotationT, BoolArray, FloatArray

from atlas_densities.densities.utils import (
    compensate_cell_overlap,
    constrain_cell_counts_per_voxel,
    get_group_ids,
    get_region_masks,
)
from atlas_densities.exceptions import AtlasDensitiesError

if TYPE_CHECKING:
    from voxcell import RegionMap  # type: ignore


def compute_inhibitory_neuron_intensity(
    gad1: FloatArray,
    nrn1: FloatArray,
    inhibitory_data: Dict[str, Union[int, Dict[str, float], Dict[str, BoolArray]]],
) -> Tuple[FloatArray, int]:
    """
    Compute a first approximation of the inhibitory neuron density based on gene markers.

    The input genetic marker datasets GAD1 and NRN1 are used to shape the spatial density
    distribution of the inhibitory and excitatory neurons respectively. Gene marker stained
    intensities are assumed to depend linearly on neuron density (number of neurons per mm^3).

    Note regarding these markers:
        Every Gabaergic neuron expresses GAD1 and every GAD1 reacting cell is a gabaergic neuron.
        This genetic marker is indeed responsible for over 90% of the synthesis of GABA. GAD1 is
        only expressed in neurons.

        From "A Cell Atlas for the Mouse Brain" by C. Eroe et al., 2018, in the section
        Neuron Type Differentiation:
        "GAD67 is mainly expressed in inhibitory neurons, and can thus be used to estimate their
        density, while NRN1 is mainly expressed in excitatory neurons (Figure 4A). We normalized
        the GAD67 marker with a sum of both markers, with an overall ratio of 7.94% between
        inhibitory neurons (Kim et al., 2017). We then used the resulting volumetric inhibitory
        marker density."

    Args:
        gad1: float array of shape (W, H, D) with non-negative entries. The GAD1 (a.k.a GAD67)
            marker dataset.
        nrn1: float array of shape (W, H, D) with non-negative entries. The Nrn1 marker dataset.
        inhibitory_data: a dictionary with 3 keys:
            'proportions': a dictionary of type Dict[str, float] assigning the proportion of
                inhibitory neurons in each group named by a key string.
            'neuron_count': the inhibitory neuron count (float).
            'region_masks': dictionary whose keys are region group names and whose values are
                the boolean masks of these groups. Each boolean array is of shape (W, H, D) and
                encodes which voxels belong to the corresponding group.

    Returns:
        `inhibitory_neuron_intensity`, a float array of shape (W, H, D) with non-negative entries.
    """

    inhibitory_neuron_intensity = gad1 / np.mean(gad1)
    excitatory_neuron_counts_per_voxel = nrn1 / np.mean(nrn1)

    marker_sum = np.zeros_like(inhibitory_neuron_intensity)
    for group, mask in inhibitory_data["region_masks"].items():  # type: ignore
        inhibitory_proportion = inhibitory_data["proportions"][group]  # type: ignore
        inhibitory_neuron_intensity[mask] = (
            inhibitory_neuron_intensity[mask] * inhibitory_proportion
        )
        marker_sum[mask] = inhibitory_neuron_intensity[mask] + excitatory_neuron_counts_per_voxel[
            mask
        ] * (1.0 - inhibitory_proportion)

    inhibitory_neuron_intensity[marker_sum > 0.0] /= marker_sum[marker_sum > 0.0]
    inhibitory_neuron_intensity /= np.max(inhibitory_neuron_intensity)

    return inhibitory_neuron_intensity


InhibitoryData = Dict[str, Union[int, Dict[str, float]]]


def compute_inhibitory_neuron_density(  # pylint: disable=too-many-arguments
    region_map: "RegionMap",
    annotation: AnnotationT,
    voxel_volume: float,
    gad1: FloatArray,
    nrn1: FloatArray,
    neuron_density: FloatArray,
    inhibitory_proportion: Optional[float] = None,
    inhibitory_data: Optional[InhibitoryData] = None,
) -> FloatArray:
    """
    Compute the inhibitory neuron density using a prescribed neuron count and the overall neuron
    density as an upper bound.

    Further constraints are imposed:
        * voxels of Purkinje layer are assigned the largest possible cell density
        * voxels sitting both in cerebellar cortex and the molecular layer are also assigned
        the largest possible neuron density.

    Args:
        region_map: object to navigate the brain regions hierarchy.
        annotation: an integer array of shape (W, H, D) enclosing the AIBS annotation of the whole
            mouse brain.
        voxel_volume: the common volume in mm^3 of a voxel associated to any of the input arrays.
        gad1: float array of shape (W, H, D) with non-negative entries. The GAD marker dataset.
        nrn1: float array of shape (W, H, D) with non-negative entries. The Nrn1 marker dataset.
        neuron_density: float array of shape (W, H, D) with non-negative entries. The input
            overall neuron density in number of neurons par mm^3.
        inhibitory_proportion: (Optional) proportion of inhibitory neurons among all neurons.
            If not provided, then `inhibitory_data` must be specified.
        inhibitory_data: (Optional) a dictionary with two keys:
            'proportions': the corresponding value is a dictionary of type Dict[str, float]
                assigning the proportion of ihnibitory neurons in each group named by a key string.
            'neuron_count': the total number of inhibitory neurons (float).
            Used only if `inhibitory_proportion` is None.

    Returns:
        float64 array of shape (W, H, D) with non-negative entries.
        The overall inhibitory neuron density is bounded by `neuron_density`, sums up to its
        prescribed neuron count after multiplication by the voxel volume, and respects some region
        "hints" where maximal density values are prescribed (Purkinje layer and molecular layer).

    Raises:
        AtlasDensitiesError if both `inhibitory_proportion` and `inhibitory_data`
        are None.
    """

    # The algorithm constraining cell counts per voxel needs double precision
    gad1 = np.asarray(gad1, dtype=float)
    nrn1 = np.asarray(nrn1, dtype=float)
    neuron_density = np.asarray(neuron_density, dtype=float)

    if inhibitory_proportion is None:
        if inhibitory_data is None:
            raise AtlasDensitiesError(
                "Either inhibitory_proportion or inhibitory_data should be provided"
                ". Both are None."
            )
        group_ids = get_group_ids(region_map)
        inhibitory_data["region_masks"] = get_region_masks(group_ids, annotation)
    else:
        inhibitory_data = {
            "proportions": {"whole brain": inhibitory_proportion},
            "neuron_count": round(np.sum(neuron_density) * voxel_volume * inhibitory_proportion),
        }
        inhibitory_data["region_masks"] = {
            "whole brain": np.ones(annotation.shape, dtype=bool)  # type: ignore
        }

    inhibitory_neuron_intensity = compute_inhibitory_neuron_intensity(
        compensate_cell_overlap(gad1, annotation, gaussian_filter_stdv=1.0),
        compensate_cell_overlap(nrn1, annotation, gaussian_filter_stdv=1.0),
        inhibitory_data,
    )

    inhibitory_neurons_mask = np.isin(
        annotation,
        list(group_ids["Purkinje layer"]),
    )
    inhibitory_neurons_mask = np.logical_or(
        inhibitory_neurons_mask,
        np.isin(
            annotation,
            list(group_ids["Cerebellar cortex"] & group_ids["Molecular layer"]),
        ),
    )
    inhibitory_neurons_mask = np.logical_or(
        inhibitory_neurons_mask,
        np.isin(
            annotation,
            region_map.find("Striatum", attr="name", with_descendants=True),
        ),
    )
    inhibitory_neurons_mask = np.logical_or(
        inhibitory_neurons_mask,
        np.isin(
            annotation,
            region_map.find(
                "Reticular nucleus of the thalamus", attr="name", with_descendants=True
            ),
        ),
    )
    # Cortical L1 regions
    inhibitory_neurons_mask = np.logical_or(
        inhibitory_neurons_mask,
        np.isin(
            annotation,
            (
                region_map.find("Isocortex", attr="name", with_descendants=True)
                & region_map.find("@.*1[ab]?$", attr="acronym", with_descendants=True)
            ),
        ),
    )

    assert isinstance(inhibitory_data["neuron_count"], int)

    return (
        constrain_cell_counts_per_voxel(
            inhibitory_data["neuron_count"],
            inhibitory_neuron_intensity,
            neuron_density * voxel_volume,
            max_cell_counts_mask=inhibitory_neurons_mask,
            copy=False,
        )
        / voxel_volume
    )
