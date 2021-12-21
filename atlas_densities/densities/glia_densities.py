"""Functions to compute glia cell densities."""
import logging
from typing import TYPE_CHECKING, Dict, Set

import numpy as np
from nptyping import NDArray  # type: ignore

from atlas_densities.densities.utils import (
    compensate_cell_overlap,
    constrain_cell_counts_per_voxel,
    get_group_ids,
)

if TYPE_CHECKING:  # pragma: no cover
    from voxcell import RegionMap, VoxelData  # type: ignore

L = logging.getLogger(__name__)


def compute_glia_cell_counts_per_voxel(  # pylint: disable=too-many-arguments
    glia_cell_count: int,
    group_ids: Dict[str, Set[int]],
    annotation: NDArray[int],
    glia_intensity: NDArray[float],
    cell_counts_per_voxel: NDArray[float],
    copy: bool = True,
) -> NDArray[float]:
    """
    Compute the overall glia cell counts per voxel using a prescribed total glia cell count and
    the overall cell counts per voxel as an upper bound.

    Further constraints are imposed:
        * voxels of fiber tracts are assigned the largest possible cell counts per voxel
        * voxels lying both in the Cerebellum and in the Purkinje layer are assigned zero
            cell counts.
    Args:
        glia_cell_count: overall glia cell count found in the scientific literature.
        group_ids: dictionary whose keys are group names and whose values are
            sets of AIBS structure ids. These groups are used to identify the voxels
            with maximum density (fiber tracts) and those of zero density (Purkinje layer).
        annotation: integer array of shape (W, H, D) enclosing the AIBS annotation of the whole
            mouse brain.
        glia_intensity: float array of shape (W, H, D) with non-negative entries. The input
            glia density obtained by averaging different marker datasets.
        cell_counts_per_voxel: float array of shape (W, H, D) with non-negative entries.
            The overall cell density of the mouse brain expressed in number of cells per voxel.
        copy: If True, the input `glia_intensity` array is copied. Otherwise it is modified in-place
             and returned by the function.

    Returns:
        float array of shape (W, H, D) with non-negative entries. The overall glia cell counts per
        voxel, respecting the constraints imposed by the `cell_counts_per_voxel` upper
        bound, the `glia_cell_count`, as well as region specific hints (fiber tracts and Purkinje
        layer).
    """

    fiber_tracts_mask = np.isin(annotation, list(group_ids["Fiber tracts group"]))
    fiber_tracts_free_mask = np.isin(
        annotation,
        list(group_ids["Purkinje layer"]),
    )

    return constrain_cell_counts_per_voxel(
        glia_cell_count,
        glia_intensity,
        cell_counts_per_voxel,
        max_cell_counts_mask=fiber_tracts_mask,
        zero_cell_counts_mask=fiber_tracts_free_mask,
        copy=copy,
    )


def compute_glia_densities(  # pylint: disable=too-many-arguments
    region_map: "RegionMap",
    annotation: NDArray[int],
    voxel_volume: float,
    glia_cell_count: int,
    glia_intensities: Dict[str, NDArray[float]],
    cell_density: NDArray[float],
    glia_proportions: Dict[str, str],
    copy: bool = False,
) -> Dict[str, NDArray[float]]:
    """
    Compute the overall glia cell density as well as astrocyte, olgidendrocyte and microglia
    densities.

    Each of the output glia cell densities should satisfy the following properties:
        * It is bounded voxel-wise by the specified overall cell density.
        * It sums up, after multiplication by the voxel volume, to a cell count matching the total
            cell count times the prescribed glia cell type proportion.

    Args:
        region_map: object to navigate the mouse brain regions hierarchy.
        annotation: an integer array of shape (W, H, D) which encloses the AIBS annotation of the
            whole mouse brain.
        voxel_volume: the common volume of a voxel associated to any of the input arrays.
        glia_cell_count: overall glia cell count (taken for instance from the scientific
            literature).
        cell_density: float array of shape (W, H, D) with non-negative entries. The overall
            cell density of the mouse brain expressed in number of cells per mm^3.
        glia_intensities: dict whose keys are glia cell types (astrocytes, oligodendrocytes,
            microglia) and whose values are the unconstrained glia cell densities corresponding to
            these types. Each density array is a float array of shape (W, H, D) with non-negative
            entries. It holds the input (unconstrained) glia density obtained by averaging different
            marker datasets.
        glia_proportions: a dict whose keys are glia cell types and whose values are strings
            encoding glia cell type proportions. These are float values in the range [0.0, 1.0]
            which sums up to 1.0.
        copy: If True, the input `glia_intensities` arrays are copied. Otherwise they are
            modified in-place

    Returns:
        A dict whose keys are glia cell types and whose values are float64 arrays of shape (W, H, D)
        with non-negative entries.
        Example: {
            "glia": <NDArray[float]>,
            "astrocyte": <NDArray[float]>,
            "microglia": <NDArray[float]>,
            "oligodendrocyte": <NDArray[float]>,
        }
        The overall glia density field is bounded by the `cell_density` field.
        It sums up to `glia_cell_count` after multiplication by the voxel volume. In
        addition, it respects some region-specific hints (fiber tracts and Purkinje layer).
        For every glia cell type, the correspondging output density field is bounded by the overall
        glia density field and sums up to its prescribed cell count when multiplied with by the
        voxel volume.

    """
    glia_densities = glia_intensities.copy()
    # The algorithm constraining cell counts per voxel requires double precision
    for glia_type in glia_densities:
        glia_densities[glia_type] = np.asarray(glia_densities[glia_type], dtype=float)
    cell_density = np.asarray(cell_density, dtype=float)

    glia_densities["glia"] = compensate_cell_overlap(
        glia_densities["glia"], annotation, gaussian_filter_stdv=1.0, copy=False
    )
    L.info(
        "Computing overall glia density field with a target cell count of %d ...",
        glia_cell_count,
    )

    glia_densities["glia"] = compute_glia_cell_counts_per_voxel(
        glia_cell_count,
        get_group_ids(region_map),
        annotation,
        glia_densities["glia"],
        cell_density * voxel_volume,
    )
    for glia_type in ["astrocyte", "oligodendrocyte"]:
        glia_densities[glia_type] = compensate_cell_overlap(
            glia_densities[glia_type],
            annotation,
            gaussian_filter_stdv=1.0,
            copy=copy,
        )
        cell_count = glia_cell_count * float(glia_proportions[glia_type])
        L.info(
            "Computing %s density field with a target cell count of %d ...",
            glia_type,
            cell_count,
        )
        glia_densities[glia_type] = constrain_cell_counts_per_voxel(
            cell_count,
            glia_densities[glia_type],
            glia_densities["glia"],
            copy=copy,
        )

    # pylint: disable=fixme
    # FIXME(Luc): The microglia density can be negative.
    glia_densities["microglia"] = (
        glia_densities["glia"] - glia_densities["astrocyte"] - glia_densities["oligodendrocyte"]
    )
    for glia_type, cell_counts_per_voxel in glia_densities.items():
        glia_densities[glia_type] = cell_counts_per_voxel / voxel_volume

    return glia_densities
