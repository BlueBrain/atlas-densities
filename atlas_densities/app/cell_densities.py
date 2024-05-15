"""Generate and save cell densities

A density value is a non-negative float number corresponding to the number of cells in mm^3.
A density field is a 3D volumetric array assigning to each voxel a density value, that is
the mean cell density within this voxel.

In BBP terminology, a density field is often referred to as a volumetric density array.

This script computes and saves the following cell densities under the form of density fields.

* overall cell density
* overall glia cell density and overall neuron density
* among glial cells:
    - astrocyte density
    - oligodendrocyte density
    - microglia density
* among neuron cells:
    - inhibitory neuron (a.k.a GAD67+) density:
        - PV+ density
        - SST+ density
        - VIP+ density
    - excitatory neuron density

Note: if M is a gene marker (e.g., GAD67), then M+ denotes the cells reacting to M (e.g., GAD67+).

Density estimates are based on datasets produced by in-situ hybridization experiments of the
Allen Institute for Brain Science (AIBS). We used in particular AIBS genetic marker datasets and
the Nissl volume of the Allen Mouse Brain Annotation Atlas.
Genetic marker stained intensity and Nissl stained intensity are assumed to be a good indicator
of the soma density in a population of interest.

It is assumed throughout that such intensities depend "almost" linearly on the cell density when
restricted to a brain region, but we shall not give a precise meaning to the word "almost".

The implementation of this module is based on the methods of
- 'A Cell Atlas for the Mouse Brain' by C. Eroe et al., 2018,
- 'Atlas of inhibitory neurons in the mouse brain' by D. Rodarie et al., 2021,
and the code provided by the authors.
"""
# pylint: disable=too-many-lines
import json
import logging
import os
from pathlib import Path
from typing import Callable, Dict

import click
import numpy as np
import pandas as pd
import yaml  # type: ignore
from atlas_commons.app_utils import (
    EXISTING_FILE_PATH,
    assert_properties,
    common_atlas_options,
    log_args,
    set_verbose,
    verbose_option,
)
from atlas_commons.typing import FloatArray
from voxcell import RegionMap, VoxelData  # type: ignore

from atlas_densities.app.utils import AD_PATH, DATA_PATH
from atlas_densities.densities import (
    excitatory_inhibitory_splitting,
    inhibitory_neuron_densities_optimization,
    refined_inhibitory_neuron_densities,
    utils,
)
from atlas_densities.densities.cell_counts import (
    extract_inhibitory_neurons_dataframe,
    glia_cell_counts,
    inhibitory_data,
)
from atlas_densities.densities.cell_density import compute_cell_density
from atlas_densities.densities.excel_reader import (
    read_homogenous_neuron_type_regions,
    read_measurements,
)
from atlas_densities.densities.fitting import linear_fitting
from atlas_densities.densities.glia_densities import compute_glia_densities
from atlas_densities.densities.inhibitory_neuron_density import compute_inhibitory_neuron_density
from atlas_densities.densities.measurement_to_density import (
    measurement_to_average_density,
    remove_non_density_measurements,
)
from atlas_densities.exceptions import AtlasDensitiesError

EXCITATORY_SPLIT_CORTEX_ALL_TO_EXC_MTYPES = (
    DATA_PATH / "mtypes" / "mapping_cortex_all_to_exc_mtypes.csv"
)
EXCITATORY_SPLIT_METADATA = DATA_PATH / "metadata" / "excitatory-inhibitory-splitting.json"
HOMOGENOUS_REGIONS_PATH = DATA_PATH / "measurements" / "homogenous_regions.csv"
HOMOGENOUS_REGIONS_REL_PATH = HOMOGENOUS_REGIONS_PATH.relative_to(AD_PATH)
MARKERS_README_REL_PATH = (DATA_PATH / "markers" / "README.rst").relative_to(AD_PATH)
LINPROG_PATH = "doc/source/bbpp82_628_linprog.pdf"

ALGORITHMS: Dict[str, Callable] = {
    "keep-proportions": refined_inhibitory_neuron_densities.create_inhibitory_neuron_densities,
    "linprog": inhibitory_neuron_densities_optimization.create_inhibitory_neuron_densities,
}

L = logging.getLogger(__name__)


def _zero_negative_values(array: FloatArray) -> None:
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


def _get_voxel_volume_in_mm3(voxel_data: "VoxelData") -> float:
    """
    Returns the voxel volume of `voxel_data` in mm^3.

    Note: the voxel_dimensions of `voxel_data` are assumed to be
    expressed in um (micron = 1e-6 m).

    Args:
        voxel_data: VoxelData object whose voxel volume will be computed.

    Returns:
        The volume in mm^3 of a `voxel_data` voxel.
    """
    return voxel_data.voxel_volume / 1e9


@click.group()
@verbose_option
def app(verbose):
    """Run the cell densities CLI"""
    set_verbose(L, verbose)


@app.command()
@common_atlas_options
@click.option(
    "--nissl-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("The path to the AIBS Nissl stains nrrd file."),
)
@click.option(
    "--output-path",
    type=str,
    required=True,
    help="Path where to write the output cell density nrrd file."
    "A voxel value is a number of cells per mm^3",
)
@click.option(
    "--group-ids-config-path",
    type=EXISTING_FILE_PATH,
    default=utils.GROUP_IDS_PATH,
    help="Path to density groups ids config",
    show_default=True,
)
@log_args(L)
def cell_density(annotation_path, hierarchy_path, nissl_path, output_path, group_ids_config_path):
    """Compute and save the overall mouse brain cell density.

    The input Nissl stain volume of AIBS is turned into an actual density field complying with
    the cell counts of several regions.

    Density is expressed as a number of cells per mm^3.
    The output density field array is a float64 array of shape (W, H, D) where (W, H, D)
    is the shape of the input annotated volume.

    The computation of the overall cell density is based on:

    \b
    - the Nissl stain intensity, which is supposed to represent the overall cell density, up to
    region-dependent constant scaling factors.
    - cell counts from the scientific literature, which are used to determine a local
    linear dependency factor for each region where a cell count is available.
    - the optional soma radii, used to operate a correction.
    """

    annotation = VoxelData.load_nrrd(annotation_path)
    nissl = VoxelData.load_nrrd(nissl_path)

    # Check nrrd metadata consistency
    assert_properties([annotation, nissl])

    region_map = RegionMap.load_json(hierarchy_path)
    group_ids_config = utils.load_json(group_ids_config_path)

    overall_cell_density = compute_cell_density(
        region_map,
        annotation.raw,
        _get_voxel_volume_in_mm3(annotation),
        nissl.raw,
        group_ids_config=group_ids_config,
    )
    nissl.with_data(overall_cell_density).save_nrrd(output_path)


@app.command()
@common_atlas_options
@click.option(
    "--cell-density-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("The path to the overall cell density nrrd file."),
)
@click.option(
    "--glia-density-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("The path to the unconstrained overall glia cell density nrrd file."),
)
@click.option(
    "--astrocyte-density-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("The path to the unconstrained astrocyte density nrrd file."),
)
@click.option(
    "--oligodendrocyte-density-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("The path to the unconstrained oligodendrocyte density nrrd file."),
)
@click.option(
    "--microglia-density-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("The path to the unconstrained microglia density nrrd file."),
)
@click.option(
    "--glia-proportions-path",
    type=EXISTING_FILE_PATH,
    help="Path to the json file containing the different proportions of each glia type."
    "This file must hold a dictionary of the following form: "
    '{"astrocyte": <proportion>, "microglia": <proportion>, "oligodendrocyte": <proportion>,'
    ' "glia": 1.0}',
)
@click.option(
    "--output-dir",
    type=str,
    required=True,
    help="Path to the directory where to write the output cell density nrrd files."
    " It will be created if it doesn't exist already.",
)
@click.option(
    "--group-ids-config-path",
    type=EXISTING_FILE_PATH,
    default=utils.GROUP_IDS_PATH,
    help="Path to density groups ids config",
    show_default=True,
)
@log_args(L)
def glia_cell_densities(
    annotation_path,
    hierarchy_path,
    cell_density_path,
    glia_density_path,
    astrocyte_density_path,
    oligodendrocyte_density_path,
    microglia_density_path,
    glia_proportions_path,
    output_dir,
    group_ids_config_path,
):  # pylint: disable=too-many-arguments, too-many-locals
    """Compute and save the glia cell densities.

    Density is expressed as a number of cells per mm^3.
    The output density field arrays are float64 arrays of shape (W, H, D) where (W, H, D)
    is the shape of the input annotated volume.

    The computation is based on:

    \b
    - an estimate of the overall cell density
    - estimates of unconstrained densities for the different glia cell types
    - glia cell counts from the scientific literature

    The cell counts and the overall cell density are used to constrain the glia cell densities
    so that:

    \b
    - they do not exceed voxel-wise the overall cell density
    - the density sums multiplied by the voxel volume match the provided cell counts

    An optimization process is responsible for enforcing these constraints while keeping
    the output densities as close as possible to the unconstrained input densities.

    Note: optimization is not fully implemented and the current process only returns a
    feasible point.

    The ouput glia densities are saved in the specified output directory under the following
    names:

    \b
    - glia_density.nrrd (overall glia density)
    - astrocyte_density.nrrd
    - oligodendrocyte_density.nrrd
    - microglia_density.nrrd

    In addition, the overall neuron cell density is inferred from the overall cell density and
    the glia cell density and saved in the same directory under the name:

    \b
    - neuron_density.nrrd
    """

    L.info("Loading annotation ...")
    annotation = VoxelData.load_nrrd(annotation_path)
    L.info("Loading overall cell density ...")
    overall_cell_density = VoxelData.load_nrrd(cell_density_path)

    if np.any(overall_cell_density.raw < 0.0):
        raise AtlasDensitiesError(f"Negative density value found in {cell_density_path}.")

    L.info("Loading unconstrained glia cell densities ...")
    glia_densities = {
        "glia": VoxelData.load_nrrd(glia_density_path),
        "astrocyte": VoxelData.load_nrrd(astrocyte_density_path),
        "oligodendrocyte": VoxelData.load_nrrd(oligodendrocyte_density_path),
        "microglia": VoxelData.load_nrrd(microglia_density_path),
    }

    atlases = list(glia_densities.values()) + [annotation, overall_cell_density]
    L.info("Checking input files consistency ...")
    assert_properties(atlases)

    L.info("Loading hierarchy ...")
    region_map = RegionMap.load_json(hierarchy_path)
    glia_proportions = utils.load_json(glia_proportions_path)

    glia_densities = {
        glia_cell_type: voxel_data.raw for glia_cell_type, voxel_data in glia_densities.items()
    }

    group_ids_config = utils.load_json(group_ids_config_path)

    L.info("Compute volumetric glia densities: started")
    glia_densities = compute_glia_densities(
        region_map,
        annotation.raw,
        _get_voxel_volume_in_mm3(annotation),
        sum(glia_cell_counts().values()),
        glia_densities,
        overall_cell_density.raw,
        glia_proportions,
        copy=False,
        group_ids_config=group_ids_config,
    )

    if not Path(output_dir).exists():
        os.makedirs(output_dir)

    L.info("Saving overall neuron density to file %s", str(Path(output_dir, "neuron_density.nrrd")))
    neuron_density = overall_cell_density.raw - glia_densities["glia"]
    _zero_negative_values(neuron_density)
    annotation.with_data(np.asarray(neuron_density, dtype=float)).save_nrrd(
        str(Path(output_dir, "neuron_density.nrrd"))
    )
    L.info("Saving glia densities to %s", str(Path(output_dir)))
    for glia_type, density in glia_densities.items():
        annotation.with_data(np.asarray(density, dtype=float)).save_nrrd(
            str(Path(output_dir, f"{glia_type}_density.nrrd"))
        )


@app.command()
@common_atlas_options
@click.option(
    "--gad1-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("The path to the GAD marker nrrd file."),
)
@click.option(
    "--nrn1-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("The path to Nrn1 marker nrrd file."),
)
@click.option(
    "--neuron-density-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=(
        "The path to the overall neuron density nrrd file obtained as output of the command "
        "`glia_cell_densities`."
    ),
)
@click.option(
    "--inhibitory-neuron-counts-path",
    type=EXISTING_FILE_PATH,
    required=False,
    default=Path(Path(__file__).parent, "data", "measurements", "mmc1.xlsx"),
    help=(
        "The path to the excel document mmc1.xlsx of the suplementary materials of "
        '"Brain-wide Maps Reveal Stereotyped Cell-Type- Based Cortical Architecture '
        'and Subcortical Sexual Dimorphism" by Kim et al., 2017. '
        "https://ars.els-cdn.com/content/image/1-s2.0-S0092867417310693-mmc1.xlsx. "
        "Defaults to `atlas_densities/app/data/measurements/mmc1.xlsx`."
    ),
)
@click.option(
    "--output-dir",
    type=str,
    required=True,
    help="Path to the directory where to write the output cell density nrrd files."
    " It will be created if it doesn't exist already.",
)
@click.option(
    "--group-ids-config-path",
    type=EXISTING_FILE_PATH,
    default=utils.GROUP_IDS_PATH,
    help="Path to density groups ids config",
    show_default=True,
)
@log_args(L)
def inhibitory_and_excitatory_neuron_densities(
    annotation_path,
    hierarchy_path,
    gad1_path,
    nrn1_path,
    neuron_density_path,
    inhibitory_neuron_counts_path,
    output_dir,
    group_ids_config_path,
):  # pylint: disable=too-many-arguments
    """Compute and save the inhibitory and excitatory neuron densities.

    Density is expressed as a number of cells per mm^3.
    The output density field arrays are float64 arrays of shape (W, H, D) where (W, H, D)
    is the shape of the input annotated volume.

    The computation is based on:

    \b
    - an estimate of the overall neuron density
    - estimates of unconstrained inhibitory and excitatory neuron densities provided by
    the GAD1 and Nrn1 markers intensities respectively.

    The overall neuron density and region-specific neuron counts from the scientific literature are
    used to constrain the inhibitory and excitatory neuron densities so that:

    \b
    - they do not exceed voxel-wise the overall neuron cell density
    - the ratio (inhibitory neuron count / excitatory neuron count) matches a prescribed value
    wherever it is constrained.

    An optimization process is responsible for enforcing these constraints while keeping
    the output densities as close as possible to the unconstrained input densities.

    Note: optimization is not fully implemented and the current process only returns a
    feasible point.

    The output densities are saved in the specified output directory under the following
    names:

    \b
    - inhibitory_neuron_density.nrrd
    - excitatory_neuron_density.nrrd
    """

    annotation = VoxelData.load_nrrd(annotation_path)
    neuron_density = VoxelData.load_nrrd(neuron_density_path)

    assert_properties([annotation, neuron_density])
    if np.any(neuron_density.raw < 0.0):
        raise AtlasDensitiesError(f"Negative density value found in {neuron_density_path}.")

    region_map = RegionMap.load_json(hierarchy_path)
    inhibitory_df = extract_inhibitory_neurons_dataframe(inhibitory_neuron_counts_path)
    group_ids_config = utils.load_json(group_ids_config_path)
    inhibitory_neuron_density = compute_inhibitory_neuron_density(
        region_map,
        annotation.raw,
        _get_voxel_volume_in_mm3(annotation),
        VoxelData.load_nrrd(gad1_path).raw,
        VoxelData.load_nrrd(nrn1_path).raw,
        neuron_density.raw,
        inhibitory_data=inhibitory_data(inhibitory_df),
        group_ids_config=group_ids_config,
    )

    if not Path(output_dir).exists():
        os.makedirs(output_dir)

    annotation.with_data(np.asarray(inhibitory_neuron_density, dtype=float)).save_nrrd(
        str(Path(output_dir, "inhibitory_neuron_density.nrrd"))
    )
    excitatory_neuron_density = neuron_density.raw - inhibitory_neuron_density
    _zero_negative_values(excitatory_neuron_density)
    annotation.with_data(np.asarray(excitatory_neuron_density, dtype=float)).save_nrrd(
        str(Path(output_dir, "excitatory_neuron_density.nrrd"))
    )


@app.command()
@click.option(
    "--measurements-output-path",
    required=True,
    help="Path where the density-related measurement series will be written. CSV file whose columns"
    " are described in the main help section.",
)
@click.option(
    "--homogenous-regions-output-path",
    required=True,
    help="Path where the list of AIBS brain regions with homogenous neuron type (e.g., inhibitory"
    ' or excitatory) will be saved. CSV file with 2 columns: "brain_region" and "cell_type".',
)
@log_args(L)
def compile_measurements(
    measurements_output_path,
    homogenous_regions_output_path,
):
    """
    Compile the cell density related measurements of mmc3.xlsx and `gaba_papers.xsls` into a CSV
    file.

    In addition to various measurements found in the scientific literature, a list of AIBS mouse
    brain regions with homogenous neuron type is saved to `homogenous_regions_output_path`.

    Two input excel files containing measurements are handled:

    \b
    - `mm3c.xls` from the supplementary materials of
    'Brain-wide Maps Reveal Stereotyped Cell-Type-Based Cortical Architecture and Subcortical
    Sexual Dimorphism' by Kim et al., 2017.
    https://ars.els-cdn.com/content/image/1-s2.0-S0092867417310693-mmc3.xlsx
    - `atlas_densities/app/data/measurements/gaba_papers.xlsx`, a compilation of measurements
    from the scientific literature made by Rodarie Dimitri (BBP).

    This command extracts measurements from the above two files and gathers them into a unique
    CSV file with the following columns:

    \b
    - brain_region (str), a mouse brain region name, not necessarily compliant
    with AIBS 1.json file. Thus some filtering must be done when working with AIBS
    annotated files.
    - ``cell type`` (str, e.g, 'PV+' for cells reacting to parvalbumin, 'inhibitory neuron'
    for non-specific inhibitory neuron)
    - ``measurement`` (float)
    - ``standard_deviation`` (non-negative float)
    - ``measurement_type`` (str), see measurement types below
    - ``measurement_unit`` (str), see measurement units below
    - ``comment`` (str), a comment on how the measurement has been obtained
    - ``source_title`` (str), the title of the article where the measurement can be exracted
    - ``specimen_age`` (str, e.g., '8 week old', 'P56', '3 month old'), age of the mice used to
    obtain the measurement

    The different measurement types are, for a given brain region R and a given cell type T:

    \b
    - ``cell density``, number of cells of type T per mm^3 in R
    - ``cell count``, number of cells of type T in R
    - ``neuron proportion``, number of cells of type T / number of neurons in R
    (a cell of type T is assumed to be a neuron, e.g., T = GAD67+)
    - ``cell proportion``, number of cells of type T / number of cells in R
    - ``cell count per slice``, number of cells of type T per slice of R

    Measurement units:

    \b
    - ``cell density``: 'number of cells per mm^3'
    - ``neuron proportion``: None (empty)
    - ``cell proportion``: None (empty)
    - ``cell count per slice``: e.g, number of cells per 50-micrometer-thick slice

    See `atlas_densities/densities/excel_reader.py` for more information.

    Note: This function should be deprecated once its output has been stored permanently as the
    unique source of density-related measurements for the AIBS mouse brain. New measurements
    should be added to the stored file (Nexus).
    """

    L.info("Loading hierarchy ...")
    region_map = RegionMap.load_json(Path(DATA_PATH, "1.json"))  # Unmodified AIBS 1.json
    L.info("Loading excel files ...")
    measurements = read_measurements(
        region_map,
        Path(DATA_PATH, "measurements", "mmc3.xlsx"),
        Path(DATA_PATH, "measurements", "gaba_papers.xlsx"),
        # The next measurement file has been obtained after manual extraction
        # of non-density measurements from the worksheets PV-SST-VIP and 'GAD67 densities'
        # of gaba_papers.xlsx.
        Path(DATA_PATH, "measurements", "non_density_measurements.csv"),
    )
    homogenous_regions = read_homogenous_neuron_type_regions(
        Path(DATA_PATH, "measurements", "gaba_papers.xlsx")
    )
    L.info("Saving to CSV files ...")
    measurements.to_csv(measurements_output_path, index=False)
    homogenous_regions.to_csv(homogenous_regions_output_path, index=False)


@app.command()
@common_atlas_options
@click.option(
    "--region-name",
    type=str,
    default="root",
    help="Name of the root region in the hierarchy",
)
@click.option(
    "--cell-density-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("The path to the to the overall cell density nrrd file."),
)
@click.option(
    "--neuron-density-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("The path to the overall neuron density nrrd file."),
)
@click.option(
    "--measurements-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=(
        "The path to measurements.csv, the compilation of cell density "
        "related measurements of Dimitri Rodarie (BBP)."
    ),
)
@click.option(
    "--output-path",
    type=str,
    required=True,
    help="Path where to write the output average cell densities (.csv file), that is, a data frame"
    " of the same format as the input measurements file (see --measurements-path) but comprising "
    "only measurements of type ``cell density``.",
)
@log_args(L)
def measurements_to_average_densities(
    annotation_path,
    hierarchy_path,
    region_name,
    cell_density_path,
    neuron_density_path,
    measurements_path,
    output_path,
):  # pylint: disable=too-many-arguments
    """Compute and save average cell densities based on measurements and AIBS region volumes.

    Measurements from Dimitri Rodarie's compilation, together with volumes from the AIBS mouse brain
    (`annotation`) and precomputed volumetric cell densities (`cell_density_path` and
    `neuron_density_path`) are used to compute average cell densities in every AIBS region where
    sufficient information is available.

    Measurements from regions which are not in the provided brain region hierarchy or not in the
    provided annotation volume will be ignored. A warning with all ignored lines from the
    measurements file will be displayed.

    The different cell types (e.g., PV+, SST+, VIP+ or overall inhibitory neurons) and
    brain regions under consideration are prescribed by the input measurements.

    Measurements can be cell densities in number of cells per mm^3 for instance.
    If several cell density measurements are available for the same region, the output dataframe
    records the average of these measurements.

    Measurements can also be cell counts, in which case the AIBS brain model volumes
    (`annotation_path`) are used in addition to compute average cell densities.

    For measurements such as cell proportions, neuron proportions or cell counts per slice, the
    brain-wide volumetric cell densities (`cell_density_path` or `neuron_density_path`) are used to
    compute average cell densities.

    If several combinations of measurements yield several average cell densities for the same
    region, then the output data frame records the average of these measurements.

    The output average cell densities are saved in under the CSV format as a dataframe with the same
    columns as the input data frame specified via `--measurements-path`.
    See :mod:`atlas_densities.app.densities.compile_measurements`.

    All output measurements are average cell densities of various cell types over AIBS brain
    regions expressed in number of cells per mm^3.
    """

    L.info("Loading annotation ...")
    annotation = VoxelData.load_nrrd(annotation_path)
    L.info("Loading overall cell density ...")
    overall_cell_density = VoxelData.load_nrrd(cell_density_path)
    if np.any(overall_cell_density.raw < 0.0):
        raise AtlasDensitiesError(f"Negative density value found in {cell_density_path}.")
    L.info("Loading overall neuron density ...")
    neuron_density = VoxelData.load_nrrd(neuron_density_path)
    if np.any(neuron_density.raw < 0.0):
        raise AtlasDensitiesError(f"Negative density value found in {neuron_density_path}.")

    L.info("Checking input consistency ...")
    assert_properties([annotation, overall_cell_density, neuron_density])

    L.info("Loading hierarchy ...")
    region_map = RegionMap.load_json(hierarchy_path)
    L.info("Loading measurements ...")
    measurements_df = pd.read_csv(measurements_path)

    L.info("Measurement to average density: started")
    average_cell_densities_df = measurement_to_average_density(
        region_map,
        annotation.raw,
        annotation.voxel_dimensions,
        _get_voxel_volume_in_mm3(annotation),
        overall_cell_density.raw,
        neuron_density.raw,
        measurements_df,
        region_name,
    )

    remove_non_density_measurements(average_cell_densities_df)

    L.info("Saving average cell densities to file %s", output_path)
    average_cell_densities_df.to_csv(
        output_path,
        index=False,
    )


@app.command()
@common_atlas_options
@click.option(
    "--region-name",
    type=str,
    default="root",
    help="Name of the root region in the hierarchy",
)
@click.option(
    "--neuron-density-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=(
        "Path to the overall neuron volumetric density nrrd file obtained as output of the command "
        "`glia-cell-densities`."
    ),
)
@click.option(
    "--gene-config-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=(
        "Path to the gene markers configuration file. This yaml file contains the paths to the "
        "gene marker volumes (nrrd files from AIBS) that will be used to estimate average cell "
        "densities accross all AIBS brain regions: PV, SST, VIP and GAD67. See "
        f"`{MARKERS_README_REL_PATH}`."
    ),
)
@click.option(
    "--average-densities-path",
    required=True,
    help="Path to the average densities data frame, i.e., the output of measurement-to-density."
    "The format of this CSV file is described in the main help section of compile-measurements. It"
    " contains only measurements of type ``cell density``.",
)
@click.option(
    "--homogenous-regions-path",
    type=EXISTING_FILE_PATH,
    required=False,
    help=f"Optional path to the CSV file containing names of regions whose neurons are "
    f"either all inhibitory or all excitatory. Defaults to `{HOMOGENOUS_REGIONS_REL_PATH}`.",
    default=HOMOGENOUS_REGIONS_PATH,
)
@click.option(
    "--fitted-densities-output-path",
    required=True,
    help="Path where to write the data frame containing the average cell density of every region"
    "found in the brain hierarchy (see --hierarchy-path option) for the cell types marked by the "
    "gene markers listed in the gene configuration file (see --gene-config-path option). "
    "The output file is a CSV file whose first column is a list of region names. The other columns"
    " come in pairs for each cell type: ``<cell_type>`` and ``<cell_type>_standard_deviation``."
    " Cell types are derived from marker names: ``<cell_type> = <marker>+``.",
)
@click.option(
    "--fitting-maps-output-path",
    required=False,
    help="Path to the json file containing the fitting coefficients and standard deviations"
    "for each region group and each cell type.",
)
@click.option(
    "--group-ids-config-path",
    type=EXISTING_FILE_PATH,
    default=utils.GROUP_IDS_PATH,
    help="Path to density groups ids config",
    show_default=True,
)
@click.option(
    "--min-data-points",
    type=int,
    default=1,
    help="minimum number of datapoints required for running the linear regression.",
    show_default=True,
)
@log_args(L)
def fit_average_densities(
    hierarchy_path,
    annotation_path,
    region_name,
    neuron_density_path,
    gene_config_path,
    average_densities_path,
    homogenous_regions_path,
    fitted_densities_output_path,
    fitting_maps_output_path,
    group_ids_config_path,
    min_data_points,
):  # pylint: disable=too-many-arguments, too-many-locals
    """
    Estimate average cell densities of brain regions in `hierarchy_path` for the cell types
    marked by the markers listed in `gene_config_path`.

    We perform a linear fitting based on average cell densities inferred from the scientific
    literature (`average_densities_path`) to estimate average cell densities in regions where
    no record is available.

    In addition to the records from the scientific literature, we consider as input of our fitting
    the average densities of the brain regions where neurons are either all inhibitory or all
    excitatory. These regions are listed in `homogenous_regions_path`. The volumetric density
    `neuron_density_path` is used to compute the average density of inhibitory neurons (a.k.a
    gad67+) in every homogenous region of type "inhibitory".

    Regions from the literature values and homogenous regions which are not in the provided brain
    region hierarchy or not in the provided annotation volume will be ignored. A warning with all
    ignored lines from the measurements file will be displayed.

    Our linear fitting of density values relies on the assumption that the average cell density
    (number of cells per mm^3) of a cell type T in a brain region R depends linearly on the
    average intensity of a gene marker of T. The conversion factor is a constant which depends only
    on T and on which of the following three groups R belongs to:

    \b
    - isocortex
    - cerebellum
    - the rest

    (For each cell type T, there are hence three linear fittings.)

    The cerebellum was singled out because of its very high cell densities wrt the rest of the
    mouse brain. The isocortex was singled out because its layer densities and cell type
    compositions are quite similar across its subregions.

    Each fitted density value of `fitted_densities_output_path` comes along with standard
    deviation.

    The standard deviations are computed differently depending on whether the output value is a
    record from the scientific literature or it has been produced by applying the fitted linear
    map to an average gene marker intensity - we call it a fitted value.
    In the first case, the deviation is the standard deviation attached to the initial and unchanged
    density in `average_densities_path`. In the second case the deviation equals the fitted value
    times the standard deviation of the linear fitting.

    Notes:

    \b
    - 2D points for which the average marker intensity or the average cell density or is 0.0 are
    filtered out before fitting.
    - some regions can have NaN density values for one or more cell types because they are not
    covered by the selected slices of the volumetric gene marker intensities.
    """
    Path(fitted_densities_output_path).parent.mkdir(parents=True, exist_ok=True)

    L.info("Loading annotation ...")
    annotation = VoxelData.load_nrrd(annotation_path)
    L.info("Loading neuron density ...")
    neuron_density = VoxelData.load_nrrd(neuron_density_path)
    if np.any(neuron_density.raw < 0.0):
        raise AtlasDensitiesError(f"Negative density value found in {neuron_density_path}.")

    L.info("Loading hierarchy ...")
    region_map = RegionMap.load_json(hierarchy_path)
    L.info("Loading gene config ...")
    with open(gene_config_path, "r", encoding="utf-8") as input_file:
        config = yaml.load(input_file, Loader=yaml.FullLoader)

    gene_voxeldata = {
        gene: VoxelData.load_nrrd(path) for (gene, path) in config["inputGeneVolumePath"].items()
    }
    # Consistency check
    voxel_data = [annotation, neuron_density]
    voxel_data += list(gene_voxeldata.values())
    assert_properties(voxel_data)

    slices = utils.load_json(config["realignedSlicesPath"])
    cell_density_stddev = utils.load_json(config["cellDensityStandardDeviationsPath"])
    cell_density_stddev = {
        # Use the AIBS name attribute as key (this is a unique identifier in 1.json)
        # (Ex: former key "|root|Basic cell groups and regions|Cerebrum" -> new key: "Cerebrum")
        name.split("|")[-1]: stddev
        for (name, stddev) in cell_density_stddev.items()
    }

    gene_marker_volumes = {}
    for gene, gene_data in gene_voxeldata.items():
        loc_slices = slices[config["sectionDataSetID"][gene]]
        gene_marker_volumes[gene] = {
            "intensity": gene_data.raw,
            # list of integer slice indices
            "slices": (
                loc_slices - np.asarray(gene_data.offset / gene_data.voxel_dimensions, dtype=int)[0]
                if loc_slices is not None
                else None
            ),
        }

    group_ids_config = utils.load_json(group_ids_config_path)

    L.info("Loading average densities dataframe ...")
    average_densities_df = pd.read_csv(average_densities_path)
    homogenous_regions_df = pd.read_csv(homogenous_regions_path)

    L.info("Fitting of average densities: started")
    fitted_densities_df, fitting_maps = linear_fitting(
        region_map,
        annotation.raw,
        neuron_density.raw,
        gene_marker_volumes,
        average_densities_df,
        homogenous_regions_df,
        cell_density_stddev,
        region_name=region_name,
        group_ids_config=group_ids_config,
        min_data_points=min_data_points,
    )

    # Turn index into column to ease off the save and load operations on csv files
    fitted_densities_df["brain_region"] = fitted_densities_df.index

    L.info("Saving fitted densities to file %s ...", fitted_densities_output_path)
    fitted_densities_df.to_csv(fitted_densities_output_path, index=False)
    if fitting_maps_output_path is not None:
        L.info("Saving fitting maps to file %s ...", fitting_maps_output_path)
        with open(fitting_maps_output_path, mode="w+", encoding="utf-8") as file_:
            json.dump(fitting_maps, file_, indent=1, separators=(",", ": "))


@app.command()
@common_atlas_options
@click.option(
    "--region-name",
    type=str,
    default="root",
    help="Name of the root region in the hierarchy",
)
@click.option(
    "--neuron-density-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=(
        "Path to the overall neuron volumetric density nrrd file obtained as output of the command "
        "`glia-cell-densities`."
    ),
)
@click.option(
    "--average-densities-path",
    required=True,
    help="Path to the average densities data frame, e.g., the output of fit-average-densities."
    " The format of this CSV file is described in the main help section of fit-average-densities.",
)
@click.option(
    "--algorithm",
    type=click.Choice(list(ALGORITHMS.keys())),
    required=False,
    default="linprog",
    help=f"Algorithm to be used. Defaults to 'linprog'. "
    f"See `{str(LINPROG_PATH)}` for a description of the linear program.",
)
@click.option(
    "--output-dir",
    required=True,
    help="Path to the directory where the volumetric inhibitory neuron density files (nrrd) will"
    " be saved. If it doesn't exist already, the directory will be created.",
)
@log_args(L)
def inhibitory_neuron_densities(
    hierarchy_path,
    annotation_path,
    region_name,
    neuron_density_path,
    average_densities_path,
    algorithm,
    output_dir,
):
    """
    Create volumetric cell densities of brain regions in `hierarchy_path` for the cell types
    labelling the columns of the data frame stored in `average_densities_path`.

    This function support the use case of "Atlas of inhibitory neurons in the mouse brain" by
    D. Rodarie et al., 2021. The densities to be computed in this case are those of GAD67+
    (inhibitory neurons) and those of the inhibitory neuron subtypes PV+, SST+ and VIP+.

    The function modifies the estimates in `average_densities` (the "first estimates" of the paper)
    to make them consistent across cell types: the average density of GAD67+ cells in a leaf
    region L should be at most the sum of the average densities of its subtypes under scrutiny
    (e.g. PV+, SST+ and VIP+) and should not exceed the neuron density of L.

    Two algorithms can be used:

    - keep-proportions:

        Whenever possible, modified densities are kept in the range [initial value - std deviation,
        initial value + std deviation] for the standard deviations specified in `average_densities`
        The proportions of the initial subtype densities are also preserved whenever possible.

    - linprog:

        A linear program minimizes the distances between the output densities and the initial
        estimates from `average_densities`  while enforcing the consistency of average densities
        across cell types. See :download:`pdf file <bbpp82_628_linear_program.pdf>` in the
        `doc/source` folder.
    """
    L.info("Loading annotation ...")
    annotation = VoxelData.load_nrrd(annotation_path)
    L.info("Loading neuron density ...")
    neuron_density = VoxelData.load_nrrd(neuron_density_path)
    if np.any(neuron_density.raw < 0.0):
        raise AtlasDensitiesError(f"Negative density value found in {neuron_density_path}.")

    L.info("Loading hierarchy ...")

    hierarchy = utils.load_json(hierarchy_path)
    if "msg" in hierarchy:
        L.warning("Top-most object contains 'msg'; assuming AIBS JSON layout")
        if len(hierarchy["msg"]) > 1:
            raise AtlasDensitiesError("Unexpected JSON layout (more than one 'msg' child)")
        hierarchy = hierarchy["msg"][0]

    # Consistency check
    L.info("Checking consistency ...")
    assert_properties([annotation, neuron_density])

    L.info("Loading average densities ...")
    average_densities_df = pd.read_csv(average_densities_path)
    average_densities_df.set_index("brain_region", inplace=True)
    if np.any(average_densities_df < 0.0):
        raise AtlasDensitiesError(f"Negative entry found in {average_densities_path}.")

    L.info("Create inhibitory neuron densities: started")
    volumetric_densities = ALGORITHMS[algorithm](
        hierarchy,
        annotation.raw,
        _get_voxel_volume_in_mm3(annotation),
        neuron_density.raw,
        average_densities_df,
        region_name=region_name,
    )

    L.info("Create inhibitory neuron densities: finished")
    if not Path(output_dir).exists():
        os.makedirs(output_dir)

    L.info("Saving density nrrd files to %s ...", output_dir)
    for cell_type, volumetric_density in volumetric_densities.items():
        annotation.with_data(volumetric_density).save_nrrd(
            str(Path(output_dir, f"{cell_type}_density.nrrd"))
        )


@app.command()
@click.option(
    "--annotation-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help="The path to the whole mouse brain annotation file (nrrd).",
)
@click.option(
    "--hierarchy-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help="The path to the hierarchy file, i.e., AIBS 1.json or BBP hierarchy.json.",
)
@click.option(
    "--neuron-density",
    type=EXISTING_FILE_PATH,
    required=True,
    help="Complete neuron density for full brain",
)
@click.option(
    "--inhibitory-density",
    type=EXISTING_FILE_PATH,
    required=True,
    help="Complete inhibitory density for full brain",
)
@click.option(
    "--cortex-all-to-exc-mtypes",
    type=EXISTING_FILE_PATH,
    required=True,
    default=EXCITATORY_SPLIT_CORTEX_ALL_TO_EXC_MTYPES,
    help="CSV file with mappings for isocortex mtypes",
    show_default=True,
)
@click.option(
    "--metadata-path",
    type=EXISTING_FILE_PATH,
    required=True,
    default=EXCITATORY_SPLIT_METADATA,
    help="CSV file with mappings for isocortex mtypes",
    show_default=True,
)
@click.option("--output-dir", required=True, help="Output path")
@log_args(L)
def excitatory_split(
    annotation_path,
    hierarchy_path,
    neuron_density,
    inhibitory_density,
    cortex_all_to_exc_mtypes,
    metadata_path,
    output_dir,
):
    """
    This program makes exc and inh densities with isocortex cut out
    It also remaps exc cells to morphological fractions in those regions
    using the m-type fractions in the csv file. All etype exc types are the same
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    region_map = RegionMap.load_json(hierarchy_path)
    brain_regions = VoxelData.load_nrrd(annotation_path)

    inhibitory_density = VoxelData.load_nrrd(inhibitory_density)
    excitatory_density = excitatory_inhibitory_splitting.make_excitatory_density(
        VoxelData.load_nrrd(neuron_density), inhibitory_density
    )

    layer_ids = excitatory_inhibitory_splitting.gather_isocortex_ids_from_metadata(
        region_map, metadata_path
    )

    excitatory_mapping = pd.read_csv(cortex_all_to_exc_mtypes).set_index("layer")

    excitatory_inhibitory_splitting.scale_excitatory_densities(
        output_dir, brain_regions, excitatory_mapping, layer_ids, excitatory_density
    )

    remove_ids = sum(layer_ids.values(), [])

    excitatory_inhibitory_splitting.set_ids_to_zero_and_save(
        str(output_dir / "Generic_Excitatory_Neuron_MType_Generic_Excitatory_Neuron_EType.nrrd"),
        brain_regions,
        excitatory_density,
        remove_ids,
    )
    excitatory_inhibitory_splitting.set_ids_to_zero_and_save(
        str(output_dir / "Generic_Inhibitory_Neuron_MType_Generic_Inhibitory_Neuron_EType.nrrd"),
        brain_regions,
        inhibitory_density,
        remove_ids,
    )
