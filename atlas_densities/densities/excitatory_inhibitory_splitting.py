""" This program makes exc and inh densities with SSCX cut out

It also remaps exc cells to morphological fractions in those regions
using the m-type fractions in the cxv file. All etype exc types are the same.
"""
import json
import logging

import numpy as np

L = logging.getLogger(__name__)


def scale_excitatory_densities(output, brain_regions, mapping, layer_ids, density):
    """Scale density by `mapping`

    Args:
        output (Path): path to output directory
        brain_regions (VoxelData): annotations of brain regions
        mapping (DataFrame): mapping of layers and mtypes to scaling factor ex:
                       L2_IPC  L2_TPC:A ...
            layer
            layer_2  0.082613  0.129525 ...
            layer_3  0.000000  0.000000 ...
            layer_4  0.000000  0.000000 ...
            layer_5  0.000000  0.000000 ...
            layer_6  0.000000  0.000000 ...

        layer_ids (dict layer_name -> list of ids): ids to scale
        density (VoxelData): initial density to scale
    """
    assert (mapping.to_numpy() >= 0).all(), "Cannot have negative scaling factors"

    for layer, df in mapping.iterrows():
        L.info("Performing layer: %s", layer)
        idx = np.nonzero(np.isin(brain_regions.raw, layer_ids[layer]))

        for mtype, scale in df.items():
            if scale == 0:
                continue

            L.info("  Performing %s", mtype)

            raw = np.zeros_like(density.raw)
            raw[idx] = scale * density.raw[idx]

            output_name = output / f"{mtype}_cADpyr.nrrd"
            density.with_data(raw).save_nrrd(str(output_name))


def set_ids_to_zero_and_save(output_name, brain_regions, nrrd, remove_ids):
    """Take an VoxelData, set some of it's values to 0, and save it

    Args:
        output_name (str): output filename
        brain_regions (VoxelData): annotations of brain regions
        nrrd (VoxelData): data where values should be set to zero
        remove_ids (list of ids): ids where the value should be set to zero
    """
    L.info("Setting region ids to zero, outputting %s", output_name)

    idx = np.nonzero(np.isin(brain_regions.raw, remove_ids))
    raw = nrrd.raw.copy()
    raw[idx] = 0.0
    nrrd.with_data(raw).save_nrrd(str(output_name))


def make_excitatory_density(neuron_density, inhibitory_density):
    """Given neuron and inhibitory cell density, calculate the excitatory density

    Args:
        neuron_density (VoxelData): total neuron density
        inhibitory_density (VoxelData): total inhibitory density

    Returns:
        VoxelData: excitatory density
    """
    exc_density = neuron_density.with_data(
        np.clip(neuron_density.raw - inhibitory_density.raw, a_min=0, a_max=None)
    )
    return exc_density


def gather_isocortex_ids_from_metadata(region_map, metadata_path):
    """given metadata, return ids of region interest

    Args:
        region_map (voxcell.RegionMap): RegionMap object to use
        metadata_path (str): path to json with metadata

    Returns:
        dict with layer -> list of ids
    """
    with open(metadata_path, encoding="utf-8") as fd:
        metadata = json.load(fd)

    # set of ids of Isocortex
    region_ids = region_map.find(
        metadata["region"]["query"],
        attr=metadata["region"]["attribute"],
        with_descendants=metadata["region"].get("with_descendants", False),
    )

    # set of ids of one layer
    metadata_layers = metadata["layers"]
    layer_ids = {
        f"layer_{i}": list(
            region_ids
            & region_map.find(
                query,
                attr=metadata_layers["attribute"],
                with_descendants=metadata_layers.get("with_descendants", False),
            )
        )
        for i, query in enumerate(metadata_layers["queries"], 1)
    }

    return layer_ids
