"""Generate and save the volumetric cell densities of the BBP mtypes

A BBP mtype is a morphological type, i.e., a BBP type string such as "NGC-SA", "CHC" or "DLAC"
for instance.
A density value is a non-negative float number corresponding to the number of cells in mm^3.
A density field, a.k.a volumetric density, is a 3D volumetric array assigning to each voxel
a density value, that is the mean cell density within this voxel.

The commands of this module create a density field for each mtype listed either in

- `app/data/mtypes/density_profiles/mapping.tsv`, or
- `app/data/mtypes/probability_map/probability_map.csv`

Volumetric density nrrd files are created for each mtype listed in either `mapping.tsv` or
`probability_map.csv`.

This module re-use the overall excitatory and inhibitory neuron densities
computed in mod:`app/cell_densities` in the first case.

In the second case, it re-uses the
computation of the densities of the neurons reacting to PV, SST, VIP and GAD67,
see also mod:`app/cell_densities`.

Note that excitatory mtypes are handled in the first but not in the second case.
"""
import json
import logging
import re
from pathlib import Path
from typing import Dict, List

import click
import numpy as np
import pandas as pd
import yaml  # type: ignore
from atlas_commons.app_utils import (
    EXISTING_FILE_PATH,
    assert_meta_properties,
    assert_properties,
    common_atlas_options,
    log_args,
    set_verbose,
)
from atlas_commons.utils import assert_metadata_content
from voxcell import RegionMap, VoxelData  # type: ignore

from atlas_densities.app.utils import AD_PATH, DATA_PATH
from atlas_densities.densities.mtype_densities_from_composition import (
    create_from_composition as _create_from_composition,
)
from atlas_densities.densities.mtype_densities_from_map import check_probability_map_sanity
from atlas_densities.densities.mtype_densities_from_map import (
    create_from_probability_map as create_from_map,
)
from atlas_densities.densities.mtype_densities_from_profiles import DensityProfileCollection
from atlas_densities.exceptions import AtlasDensitiesError

MTYPES_PROFILES_REL_PATH = (DATA_PATH / "mtypes" / "density_profiles").relative_to(AD_PATH)
MTYPES_PROBABILITY_MAP_REL_PATH = (DATA_PATH / "mtypes" / "probability_map").relative_to(AD_PATH)
MTYPES_COMPOSITION_REL_PATH = (DATA_PATH / "mtypes" / "composition").relative_to(AD_PATH)
METADATA_PATH = DATA_PATH / "metadata"
METADATA_REL_PATH = METADATA_PATH.relative_to(AD_PATH)

L = logging.getLogger(__name__)


@click.group()
@click.option("-v", "--verbose", count=True)
def app(verbose):
    """Run the mtype densities CLI"""
    set_verbose(L, verbose)


@app.command()
@common_atlas_options
@click.option(
    "--metadata-path",
    type=EXISTING_FILE_PATH,
    required=False,
    help=(
        "(Optional) Path to the metadata json file. Defaults to "
        f"`{str(METADATA_REL_PATH / 'isocortex_metadata.json')}`"
    ),
    default=str(METADATA_PATH / "isocortex_metadata.json"),
)
@click.option(
    "--direction-vectors-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("Path to the mouse isocortex direction vectors file, e.g., `direction_vectors.nrrd`."),
)
@click.option(
    "--mtypes-config-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help="Path to the yaml configuration file. "
    f"See `{str(MTYPES_PROFILES_REL_PATH / 'README.rst')}` for an example.",
)
@click.option(
    "--output-dir",
    required=True,
    help="Path to output directory. It will be created if it doesn't exist already.",
)
@log_args(L)
def create_from_profile(
    annotation_path,
    hierarchy_path,
    metadata_path,
    direction_vectors_path,
    mtypes_config_path,
    output_dir,
):  # pylint: disable=too-many-locals
    """
    Create neuron density nrrd files for the mtypes listed in the mapping tsv file.

    Somatosensory cortex layers are subdivided into slices (a.k.a bins). Each mtype in
    the mapping tsv file (see configuration file description) is assigned a density profile,
    that is, the list of the numbers of neurons with this mtype in each slice. From this, a
    relative density profile is derived, i.e. the list of the neuron proportions in each slice.
    Using the overall inhibitory neuron and excitatory neuron densities together with the relative
    density profiles, we obtain a volumetric neuron density for each mtype under the form of nrrd
    files.

    The streamlines of the direction vectors filed are used to divide layers into slices, i.e.,
    sublayers of equal thickness along the cortical axis. The number of slices per layer is
    specified by the field layerSlicesPath of the configuration file (defaults to `layers.tsv`).

    Neuron densities are expressed in number of neurons per voxel.

    The density profile datasets were obtained in
    "A Derived Positional Mapping of Inhibitory Subtypes in the Somatosensory Cortex"
    <https://www.frontiersin.org/articles/10.3389/fnana.2019.00078/full>.
    These datasets and associated metadata files can be found in
    :mod:`atlas_densities/app/data/mtypes/density_profiles`.
    """

    L.info("Collecting density profiles ...")
    with open(mtypes_config_path, "r", encoding="utf-8") as file_:
        config = yaml.load(file_, Loader=yaml.FullLoader)

    density_profile_collection = DensityProfileCollection.load(
        config["mtypeToProfileMapPath"],
        config["layerSlicesPath"],
        config["densityProfilesDirPath"],
    )

    L.info("Density profile collection successfully instantiated.")
    with open(metadata_path, "r", encoding="utf-8") as file_:
        metadata = json.load(file_)
    region_map = RegionMap.load_json(hierarchy_path)

    annotation = VoxelData.load_nrrd(annotation_path)
    direction_vectors = VoxelData.load_nrrd(direction_vectors_path)
    voxeldata = [annotation, direction_vectors]

    inhibitory_neuron_density = None
    excitatory_neuron_density = None

    if "inhibitoryNeuronDensityPath" in config:
        inhibitory_neuron_density = VoxelData.load_nrrd(config["inhibitoryNeuronDensityPath"])
        voxeldata.append(inhibitory_neuron_density)
    if "excitatoryNeuronDensityPath" in config:
        excitatory_neuron_density = VoxelData.load_nrrd(config["excitatoryNeuronDensityPath"])
        voxeldata.append(excitatory_neuron_density)

    if inhibitory_neuron_density is None and inhibitory_neuron_density is None:
        raise AtlasDensitiesError(
            "No neuron density files were provided. Expected: excitatory neuron density, or"
            "inhibitory neuron density or both."
        )
    # Check metadata consistency
    assert_meta_properties(voxeldata)

    density_profile_collection.create_mtype_densities(
        annotation,
        region_map,
        metadata,
        np.asarray(direction_vectors.raw, dtype=np.float32),
        output_dir,
        excitatory_neuron_density,
        inhibitory_neuron_density,
    )


def _check_config_sanity(config: dict) -> None:
    """
    Check if `config` has the expected keys.

    Raises otherwise.

    Args:
        config: the dict to be checked.
    Raises: AtlasBuildingTools error on failure.
    """
    diff = {"probabilityMapPath", "molecularTypeDensityPaths"} - set(config.keys())
    if diff:
        raise AtlasDensitiesError(
            f"The following keys are missing from the configuration file: {list(diff)}"
        )


def standardize_probability_map(probability_map: "pd.DataFrame") -> "pd.DataFrame":
    """
    Standardize the labels of the rows and the columns of `probability_map` and
    remove unused rows.

    Output labels are all lower case.
    The underscore is the only delimiter used in an output label.
    The layer names refered to by output labels are:
        "layer_1," "layer_23", "layer_4", "layer_5" and "layer_6".
    Rows whose labels contain "VIP" or "6b" are removed.

    Row example: "L2/3 Pvalb-IRES-Cre" -> "layer_23_pv"
    Column example: "NGC-SA" -> "ngc_sa"

    Args:
        probability_map: probability_map:
            data frame whose rows are labeled by molecular types and layers (e.g.,
            "L6a Htr3a-Cre_NO152", "L2/3 Pvalb-IRES-Cre", "L4 Htr3a-Cre_NO152") and whose columns
            are labeled by mtypes (mtypes = morphological types, e.g., "NGC-SA", "ChC", "DLAC").

    Returns: a data frame complying with all the above constraints.
    """

    def standardize_row_label(row_label: str):
        """
        Lowercase labels and use explicit layer names.
        """
        splitting = re.split("_", row_label)  # remove unused Creline information
        splitting[0] = splitting[0].replace("L", "layer_")
        splitting[1] = splitting[1].replace("Pvalb", "pv")
        # Although Gad2 = Gad65, see e.g. https://www.genecards.org/cgi-bin/carddisp.pl?gene=GAD2
        # Gad1 = Gad67 is taken as an acceptable substitute for density estimates.
        splitting[1] = splitting[1].replace("Gad2", "gad67")

        return "_".join(splitting).lower()

    def standardize_column_label(col_label: str):
        """
        Lowercase labels and use underscore as delimiter for composed
        molecular types such as NGC-DA or NGC-SA.

        Example: "NGC-SA" -> "ngc_sa"
        """
        col_label = col_label.replace("-", "_")

        return col_label.lower()

    bbp_mtypes_map = {"DLAC": "LAC", "SLAC": "SAC"}
    probability_map.rename(bbp_mtypes_map, axis="columns", inplace=True)
    probability_map.rename(standardize_column_label, axis="columns", inplace=True)
    probability_map.rename(standardize_row_label, axis="rows", inplace=True)

    return probability_map


@app.command()
@common_atlas_options
@click.option(
    "--metadata-path",
    type=EXISTING_FILE_PATH,
    required=False,
    help=(
        "(Optional) Path to the metadata json file. Defaults to "
        f"`{str(METADATA_REL_PATH / 'isocortex_metadata.json')}`"
    ),
    default=str(METADATA_PATH / "isocortex_metadata.json"),
)
@click.option(
    "--mtypes-config-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help="Path to the yaml configuration file. "
    f"See `{str(MTYPES_PROBABILITY_MAP_REL_PATH / 'README.rst')}` for an example.",
)
@click.option(
    "--output-dir",
    required=True,
    help="Path to output directory. It will be created if it doesn't exist already.",
)
@log_args(L)
def create_from_probability_map(
    annotation_path,
    hierarchy_path,
    metadata_path,
    mtypes_config_path,
    output_dir,
):  # pylint: disable=too-many-locals
    """
    Create neuron density nrrd files for the mtypes listed in the probability mapping csv file.

    Neuron densities are expressed in number of neurons per voxel.

    The probability mapping was obtained in PUBLICATION by Y. Roussel et al.
    It is a mapping between BBP mtypes and the molecular types PV, SST, VIP, HTR3A and GAD67,
    (these are molecular markers of inhibitory neurons).
    It can be found in
    :mod:`atlas_densities/app/data/mtypes/probability_map`.

    Note: this command does not generate volumetric density files for excitatory neurons.
    """

    L.info("Loading configuration file ...")
    with open(mtypes_config_path, "r", encoding="utf-8") as file_:
        config = yaml.load(file_, Loader=yaml.FullLoader)
    _check_config_sanity(config)

    L.info("Loading probability mapping ...")
    probability_map = pd.DataFrame(pd.read_csv(config["probabilityMapPath"]))
    if "molecular_type" in probability_map.columns:
        probability_map.set_index("molecular_type", inplace=True)
    else:
        probability_map.set_index(probability_map.columns[0], inplace=True)

    # Remove useless lines, use lower case and "standardized" explicit label names
    probability_map = standardize_probability_map(probability_map)
    check_probability_map_sanity(probability_map)

    L.info("Loading brain region metadata ...")
    with open(metadata_path, "r", encoding="utf-8") as file_:
        metadata = json.load(file_)
        assert_metadata_content(metadata)

    L.info("Loading hierarchy json file ...")
    region_map = RegionMap.load_json(hierarchy_path)

    L.info("Loading annotation nrrd file ...")
    annotation = VoxelData.load_nrrd(annotation_path)

    L.info("Loading volumetric densities of molecular types ...")
    molecular_type_densities = {
        molecular_type: VoxelData.load_nrrd(density_path)
        for (molecular_type, density_path) in config["molecularTypeDensityPaths"].items()
    }

    # Check metadata consistency
    voxeldata = [annotation] + list(molecular_type_densities.values())
    assert_meta_properties(voxeldata)

    L.info("Creating volumetric densities of mtypes specified in probability map ...")
    create_from_map(
        annotation,
        region_map,
        metadata,
        {
            molecular_type: density.raw
            for (molecular_type, density) in molecular_type_densities.items()
        },
        probability_map,
        output_dir,
    )


@app.command()
@common_atlas_options
@click.option(
    "--metadata-path",
    type=EXISTING_FILE_PATH,
    required=False,
    help=(
        "(Optional) Path to the metadata json file. Defaults to "
        f"`{str(METADATA_REL_PATH / 'isocortex_metadata.json')}`"
    ),
    default=str(METADATA_PATH / "isocortex_metadata.json"),
)
@click.option(
    "--excitatory-neuron-density-path",
    required=True,
    help="Path to excitatory neuron density file (nrrd).",
)
@click.option(
    "--taxonomy-path",
    required=True,
    help=(
        "Path to mtype taxonomy file (tsv). "
        f"See `{str(MTYPES_COMPOSITION_REL_PATH / 'composition.yaml')}` for an example."
    ),
)
@click.option(
    "--composition-path",
    required=True,
    help=(
        "Path to mtype composition file (yaml). "
        f"See `{str(MTYPES_COMPOSITION_REL_PATH / 'neurons-mtype-taxonomy.tsv')}` for an example."
    ),
)
@click.option(
    "--output-dir",
    required=True,
    help="Path to output directory. It will be created if it doesn't exist already.",
)
@log_args(L)
def create_from_composition(
    annotation_path,
    hierarchy_path,
    metadata_path,
    excitatory_neuron_density_path,
    taxonomy_path,
    composition_path,
    output_dir,
):  # pylint: disable=too-many-locals
    """Create neuron density nrrd files for the excitatory mtypes listed in the taxonomy and compo-
    sition files.

    Neuron densities are expressed in number of neurons per mm^3.

    The algorithm extracts the excitatory mtypes found in the taxonomy file and obtains their
    average density and layer from the composition file. Then it calculates the ratio of each mtype
    average density over the total average excitatory density in the layer it is located in. The
    new excitatory volumetric densities are created per mtype by multiplying the aforementioned
    ratio with the total excitatory volumetric density.

    It generates in the specified `output_dir` nrrd densities, one per mtype with the following
    naming convention: {mtype}_densities.nrrd

    Note:
        - Does not generate volumetric density files for inhibitory neurons
        - Only works for brain regions with well defined layers, numbered from 1 to some upper bound
    """
    L.info("Loading annotation nrrd file ...")
    annotation = VoxelData.load_nrrd(annotation_path)

    L.info("Loading hierarchy json file ...")
    region_map = RegionMap.load_json(hierarchy_path)

    L.info("Loading metadata json file ...")
    with open(metadata_path, "r", encoding="utf-8") as jsonfile:
        metadata = json.load(jsonfile)

    L.info("Loading excitatory neuron densities ...")
    excitatory_neuron_density = VoxelData.load_nrrd(excitatory_neuron_density_path)
    _validate_density(excitatory_neuron_density)

    L.info("Loading neuronal mtype taxonomy file ...")
    neuronal_mtype_taxonomy = _load_neuronal_mtype_taxonomy(taxonomy_path)
    _validate_mtype_taxonomy(neuronal_mtype_taxonomy)

    L.info("Loading neuronal mtype composition file ...")
    neuronal_mtype_composition = _load_neuronal_mtype_composition(composition_path)
    _validate_neuronal_mtype_composition(neuronal_mtype_composition)

    # check conforming shape, voxel dimensions, offset
    assert_properties([annotation, excitatory_neuron_density])
    _check_taxonomy_composition_congruency(neuronal_mtype_taxonomy, neuronal_mtype_composition)

    L.info("Creating volumetric densities of mtypes specified in composition ...")
    per_mtype_volumetric_density_generator = _create_from_composition(
        annotation,
        region_map,
        metadata,
        excitatory_neuron_density.raw,
        neuronal_mtype_taxonomy,
        neuronal_mtype_composition,
    )

    L.info("Writing mtype density files ...")
    for mtype, density in per_mtype_volumetric_density_generator:
        path = Path(output_dir, f"{mtype}_densities.nrrd")
        annotation.with_data(density).save_nrrd(str(path))


def _load_neuronal_mtype_taxonomy(filename: str) -> pd.DataFrame:
    """Loads the taxonomy tsv file into a dataframe"""
    return pd.read_csv(filename, header=0, delim_whitespace=True)


def _validate_mtype_taxonomy(taxonomy: pd.DataFrame) -> None:
    """Checks if the taxonomy file consists of three columns [mtype, mClass, sClass] and if the
    sClass column consists only of EXC and INH entries.

    Raises:
        AtlasBuildingError in case of failure.
    """
    expected_columns = {"mtype", "mClass", "sClass"}
    if expected_columns != set(taxonomy.columns):
        raise AtlasDensitiesError(
            f"Column name missmatch. Expected {expected_columns}, Found: {taxonomy.columns}"
        )

    sclasses = set(taxonomy["sClass"])
    expected_sclasses = {"EXC", "INH"}
    if expected_sclasses != sclasses:
        raise AtlasDensitiesError(
            f"sClass column values are different than expected.\n"
            f"Expected {expected_sclasses}. Found {sclasses}"
        )


def _load_neuronal_mtype_composition(filename: str) -> pd.DataFrame:
    """
    Returns:
        dict whose keys are the cell groups (e.g., neurons, glia, etc.) and whose values are
        dataframes with the following columns:
            [density, region, layer, mtype]

    Notes:
        The densities are expressed in number of cells per mm^3
    """
    with open(filename, "r", encoding="utf-8") as stream:
        composition = yaml.safe_load(stream)["neurons"]

    composition_dict: Dict[str, List] = {"density": [], "layer": [], "mtype": []}

    for entry in composition:

        traits = entry["traits"]
        composition_dict["density"].append(entry["density"])
        composition_dict["layer"].append(f"layer_{traits['layer']}")
        composition_dict["mtype"].append(traits["mtype"])

    return pd.DataFrame(composition_dict, columns=["density", "layer", "mtype"])


def _validate_density(density: VoxelData) -> None:
    """Checks that density does not have zero everywhere or negative values"""
    if np.allclose(density.raw, 0.0):
        raise AtlasDensitiesError("Density with zeros everywhere encountered.")

    if np.any(density.raw < 0.0):
        raise AtlasDensitiesError("Density with negative values encountered.")


def _validate_neuronal_mtype_composition(composition: pd.DataFrame) -> None:
    """Checks that composition does not have negative values"""
    if np.any(composition["density"] < 0.0):
        raise AtlasDensitiesError("Negative density values encountered in composition.")


def _check_taxonomy_composition_congruency(
    taxonomy: pd.DataFrame, composition: pd.DataFrame
) -> None:
    """Checks if the taxonomy and composition have the same set of mtypes"""
    taxonomy_mtypes = set(taxonomy["mtype"])
    composition_mtypes = set(composition["mtype"])

    if not taxonomy_mtypes == composition_mtypes:
        raise AtlasDensitiesError(
            "Taxonomy and composition mtypes are inconsistent:\n"
            f"In taxonomy but not in composition:{' '.join(taxonomy_mtypes - composition_mtypes)}\n"
            f"In composition but not in taxonomy:{' '.join(composition_mtypes - taxonomy_mtypes)}\n"
            f"Taxonomy   :{' '.join(sorted(taxonomy_mtypes))}\n"
            f"Composition:{' '.join(sorted(composition_mtypes))}"
        )
