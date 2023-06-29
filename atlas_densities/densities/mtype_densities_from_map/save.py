"""
Save the density field.
"""
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from voxcell import VoxelData  # type: ignore

logger = logging.getLogger(__name__)


def save_densities(
    mtype: str,
    annotation: "VoxelData",
    region_acronyms: List[str],
    region_masks: Dict[str, "np.ndarray"],
    mtype_density: "np.ndarray",
    output_dirpath: str,
) -> List[str]:
    """
    Save the density field into files.

    Args:
        mtype: Morphology type to save.
        annotation: VoxelData holding an int array of shape (W, H, D) where W, H and D are integer
            dimensions; this array is the annotated volume of the brain region of interest.
        region_acronyms: Region acronyms from the hierarchy that are to be saved.
        region_masks: Dictionary of binary masks for each region acronym.
        mtype_density: Calculated density for the mtype.
        output_dirpath: path of the directory where to save the volumetric density nrrd files.
            It will be created if it doesn't exist already. It will contain two subdirectories,
            namely `no_regions` and `with_regions`. They will be created if they don't exist.
            The subdirectory `no_regions` contains a volumetric density file of each mtype
            appearing as column label of `probability_map`.
            The subdirectory `with_regions` contains the volumetric density of each mtype for each
            layer.
    """
    zero_density_mtypes = []
    mtype_filename = f"{mtype.replace('_', '-')}_densities.nrrd"

    (Path(output_dirpath) / "no_regions").mkdir(exist_ok=True, parents=True)
    (Path(output_dirpath) / "with_regions").mkdir(exist_ok=True, parents=True)

    if not np.isclose(np.sum(mtype_density), 0.0):
        filepath = str(Path(output_dirpath) / "no_regions" / mtype_filename)
        logger.info("Saving %s ...", filepath)
        annotation.with_data(mtype_density).save_nrrd(filepath)

    for region_acronym in region_acronyms:
        region_density = np.zeros(region_masks[region_acronym].shape, dtype=float)
        region_density[region_masks[region_acronym]] = mtype_density[region_masks[region_acronym]]
        filename = f"{region_acronym.replace('/', '')}_{mtype_filename}"
        if not np.isclose(np.sum(region_density), 0.0):
            filepath = str(Path(output_dirpath) / "with_regions" / filename)
            logger.info("Saving %s ...", filepath)
            annotation.with_data(region_density).save_nrrd(filepath)
        else:
            zero_density_mtypes.append(filename.replace("_densities.nrrd", ""))

    return zero_density_mtypes
