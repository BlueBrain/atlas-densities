"""Generate and save combined annotations or combined markers

Combination operates on two or more volumetric files with nrrd format.
"""
import json
import logging
from pathlib import Path

import click
import pandas as pd
import voxcell  # type: ignore
import yaml  # type: ignore
from atlas_commons.app_utils import EXISTING_FILE_PATH, common_atlas_options, log_args, set_verbose

from atlas_densities.combination import annotations_combinator, markers_combinator

L = logging.getLogger(__name__)


@click.group()
@click.option("-v", "--verbose", count=True)
def app(verbose):
    """Run the combination CLI"""
    set_verbose(L, verbose)


# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
@app.command()
@click.option(
    "--hierarchy-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help="Path to AIBS 1.json file or an equivalent BBP hierarchy.json file.",
)
@click.option(
    "--brain-annotation-ccfv2",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("This brain annotation file contains the most complete annotation."),
)
@click.option(
    "--fiber-annotation-ccfv2",
    type=EXISTING_FILE_PATH,
    required=True,
    help="Fiber annotation is not included in the CCF-v2 2011 annotation files.",
)
@click.option(
    "--brain-annotation-ccfv3",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("More recent brain annotation file with missing leaf regions."),
)
@click.option("--output-path", required=True, help="Path of the nrrd file to write")
@log_args(L)
def combine_v2_v3_annotations(
    hierarchy_path,
    brain_annotation_ccfv2,
    fiber_annotation_ccfv2,
    brain_annotation_ccfv3,
    output_path,
):
    # pylint: disable=line-too-long
    """Generate and save the v2-v3 combined annotation file.

    The annotation file `brain_annotation_ccfv3` is the annotation file containing
    the least complete annotation. There are two use cases: a resolution of 10 or 25 um.

    For a resolution of 10 um, the path arguments should be the following:

    \b
    - `hierarchy_path` is the path to a copy of
    http://api.brain-map.org/api/v2/structure_graph_download/1.json

    \b
    - `brain_annotation_ccfv2` is the path to a copy of
    ``AIBS_ANNOTATION_URL``/mouse_2011/annotation_10.nrrd

    \b
    - `fiber_annotation` is the path to a copy of
    ``AIBS_ANNOTATION_URL``/mouse_2011/annotationFiber_10_2011.nrrd

    \b
    - `brain_annotation_ccfv3` s the path to a copy of
    ``AIBS_ANNOTATION_URL``/ccf_2017/annotation_10.nrrd

    where ``AIBS_ANNOTATION_URL`` is
    http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation.
    """

    region_map = voxcell.RegionMap.load_json(hierarchy_path)

    brain_annotation_ccfv2 = voxcell.VoxelData.load_nrrd(brain_annotation_ccfv2)
    fiber_annotation_ccfv2 = voxcell.VoxelData.load_nrrd(fiber_annotation_ccfv2)
    brain_annotation_ccfv3 = voxcell.VoxelData.load_nrrd(brain_annotation_ccfv3)

    combined_annotation = annotations_combinator.combine_annotations(
        region_map,
        brain_annotation_ccfv2,
        fiber_annotation_ccfv2,
        brain_annotation_ccfv3,
    )
    combined_annotation.save_nrrd(output_path)


@app.command()
@click.option(
    "--brain-annotation-ccfv2",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("This brain annotation file contains the most complete annotation."),
)
@click.option(
    "--fiber-annotation-ccfv2",
    type=EXISTING_FILE_PATH,
    required=True,
    help="Fiber annotation is not included in the CCF-v2 2011 annotation files.",
)
@click.option("--output-path", required=True, help="Path of the nrrd file to write")
@log_args(L)
def combine_ccfv2_annotations(
    brain_annotation_ccfv2,
    fiber_annotation_ccfv2,
    output_path,
):
    # pylint: disable=line-too-long
    """Generate and save the ccfv2 combined annotation file.

    The ccfv2 annotations are split into two volumes. `fiber_annotation_ccfv2` describes solely the
    fibers and ventricular related regions while `brain_annotation_ccfv2` contains all other brain
    regions. There are two use cases: a resolution of 10 or 25 um.

    For a resolution of 10 um, the path arguments should be the following:

    \b
    - `brain_annotation_ccfv2` is the path to a copy of
    ``AIBS_ANNOTATION_URL``/mouse_2011/annotation_10.nrrd

    \b
    - `fiber_annotation` is the path to a copy of
    ``AIBS_ANNOTATION_URL``/mouse_2011/annotationFiber_10_2011.nrrd

    where ``AIBS_ANNOTATION_URL`` is
    http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation.
    """

    brain_annotation_ccfv2 = voxcell.VoxelData.load_nrrd(brain_annotation_ccfv2)
    fiber_annotation_ccfv2 = voxcell.VoxelData.load_nrrd(fiber_annotation_ccfv2)

    combined_annotation = annotations_combinator.combine_ccfv2_annotations(
        brain_annotation_ccfv2, fiber_annotation_ccfv2
    )
    combined_annotation.save_nrrd(output_path)


@app.command()
@common_atlas_options
@click.option(
    "--config",
    type=EXISTING_FILE_PATH,
    required=True,
    help=(
        "Path to the gene markers configuration file. This yaml file indicates which markers are"
        " used to identify the different glia cell types. It contains the path to the gene marker"
        " volumes as well as their average expression intensities and the glia intensity output "
        "paths. See tests/markers_config.yaml for an example."
    ),
)
@log_args(L)
def combine_markers(annotation_path, hierarchy_path, config):
    """Generate and save the combined glia files and the global celltype scaling factors.

    This function performs the operations indicated by the formula of the
    'Glia differentiation' section in 'A Cell Atlas for the Mouse Brain' by
    C. Eroe et al., 2018.
    See https://www.frontiersin.org/articles/10.3389/fninf.2018.00084/full.

    The output consists in:

    \b
    - A 3D volumetric nrrd file for each cell type (oligodendrocyte, astrocyte, microglia)
    representing the average density of each cell type, up to a uniform constant factor.

    \b
    - A 3D volumetric nrrd file representing the overall average density of the glia in the
    whole mouse brain up to the same uniform constant factor.

    \b
    - The global cell type scaling factors S_celltype of the 'Glia differentiation' section in
    'A Cell Atlas for the Mouse Brain' by C. Eroe et al. 2018,
    i.e., the proportions of each glia cell type in the whole mouse brain.
    """
    annotation = voxcell.VoxelData.load_nrrd(annotation_path)
    hierarchy = voxcell.RegionMap.load_json(hierarchy_path)
    with open(config, "r", encoding="utf-8") as file_:
        config = yaml.load(file_, Loader=yaml.FullLoader)
    glia_celltype_densities = pd.DataFrame(config["cellDensity"])
    combination_data = pd.DataFrame(config["combination"])
    volumes = pd.DataFrame(
        [
            [gene, voxcell.VoxelData.load_nrrd(path).raw]
            for (gene, path) in config["inputGeneVolumePath"].items()
        ],
        columns=["gene", "volume"],
    )
    combination_data = combination_data.merge(volumes, how="inner", on="gene")

    glia_intensities = markers_combinator.combine(
        hierarchy, annotation.raw, glia_celltype_densities, combination_data
    )

    for type_, output_path in config["outputCellTypeVolumePath"].items():
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        annotation.with_data(glia_intensities.intensity[type_]).save_nrrd(output_path)

    annotation.with_data(glia_intensities.intensity["glia"]).save_nrrd(
        config["outputOverallGliaVolumePath"]
    )

    proportions = dict(glia_intensities.proportion.astype(str))
    with open(config["outputCellTypeProportionsPath"], "w", encoding="utf-8") as out:
        json.dump(proportions, out, indent=1, separators=(",", ": "))
