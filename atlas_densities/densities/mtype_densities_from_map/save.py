"""
Save the density field.
"""
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from voxcell import VoxelData  # type: ignore

logger = logging.getLogger(__name__)


def save_densities(
    mtype: str,
    annotation: "VoxelData",
    mtype_density: "np.ndarray",
    output_dirpath: str,
) -> None:
    """
    Save the density field into files.

    Args:
        mtype: Morphology type to save.
        annotation: VoxelData holding an int array of shape (W, H, D) where W, H and D are integer
            dimensions; this array is the annotated volume of the brain region of interest.
        mtype_density: Calculated density for the mtype.
        output_dirpath: path of the directory where to save the volumetric density nrrd files.
            It will be created if it doesn't exist already. It will contain a volumetric density
            file of each mtype appearing as column label of `probability_map`.
    """
    mtype_filename = f"{mtype.replace('_', '-')}_densities.nrrd"

    Path(output_dirpath).mkdir(exist_ok=True, parents=True)

    if not np.isclose(np.sum(mtype_density), 0.0):
        filepath = str(Path(output_dirpath) / mtype_filename)
        logger.info("Saving %s ...", filepath)
        annotation.with_data(mtype_density).save_nrrd(filepath)
