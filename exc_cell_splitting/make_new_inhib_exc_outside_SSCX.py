###############################################################################
# Copyright 2022 Blue Brain
# Author: Daniel Keller
# October 21, 2022
# This program makes exc and inh densities with SSCX cut out
# It also remaps exc cells to morphological fractions in those regions
# using the m-type fractions in the cxv file. All etype exc types are the same.

# inputs: annoation atlas and hirearchy
#         total neuron and inhibitory nrrd files
#         mapping csv file
# outputs: (in output directory)
#         exc and inhib densities without SSCX
#         ME type excitatory nrrd files
###############################################################################

import logging
import os

from pathlib import Path

from voxcell import VoxelData
from voxcell.nexus.voxelbrain import Atlas

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO)

L = logging.getLogger(__name__)


#make excitatory density
neuron_density = VoxelData.load_nrrd("neuron_density_v3.nrrd")
inh_density = VoxelData.load_nrrd("gad67+_density_v3.nrrd")
exc_density = neuron_density.with_data(np.clip(neuron_density.raw - inh_density.raw, a_min=0, a_max=None))

excitatory_mapping = pd.read_csv('mapping_cortex_all_to_exc_mtypes.csv').set_index('layer')
atlas = Atlas.open('.')
region_map = atlas.load_region_map()
brain_regions = atlas.load_data('brain_regions')

#layer names, starting with layer 1 and going to layer 6
layer_names = {
 'L1': ['SS1', 'SSp1','SSp-n1', 'SSp-bfd1', 'SSp-ll1', 'SSp-m1', 'SSp-tr1', 'SSp-ul1', 'SSp-un1', 'SSs1'],
 'L2': ['SS2', 'SSp2','SSp-n2', 'SSp-bfd2', 'SSp-ll2', 'SSp-m2', 'SSp-tr2', 'SSp-ul2', 'SSp-un2', 'SSs2'],
 'L3': ['SS3', 'SSp3','SSp-n3', 'SSp-bfd3', 'SSp-ll3', 'SSp-m3', 'SSp-tr3', 'SSp-ul3', 'SSp-un3', 'SSs3'],
 'L4': ['SS4', 'SSp4','SSp-n4', 'SSp-bfd4', 'SSp-ll4', 'SSp-m4', 'SSp-tr4', 'SSp-ul4', 'SSp-un4', 'SSs4'],
 'L5': ['SS5', 'SSp5','SSp-n5', 'SSp-bfd5', 'SSp-ll5', 'SSp-m5', 'SSp-tr5', 'SSp-ul5', 'SSp-un5', 'SSs5'],
 'L6': ['SS6a', 'SS6b','SSp6a', 'SSp6b', 'SSp-n6a', 'SSp-n6b','SSp-bfd6a', 'SSp-bfd6b', 'SSp-ll6a', 'SSp-ll6b', 'SSp-m6a', 'SSp-m6b',  'SSp-tr6a', 'SSp-tr6b', 'SSp-ul6a', 'SSp-ul6b', 'SSp-un6a', 'SSp-un6b', 'SSs6a', 'SSs6b'],
}


def get_ids(region_map, acronyms):
    return [next(iter(region_map.find(acronym, 'acronym'))) for acronym in acronyms]


def scale_excitatory_densities(output, region_map, brain_regions, mapping, layer_names, exc_density):
    output = Path(output)

    for layer, df in mapping.iterrows():
        ids = get_ids(region_map, layer_names[layer])
        idx = np.nonzero(np.isin(brain_regions.raw, ids))

        for mtype, scale in df.iteritems():
            if scale == 0:
                continue
            L.info('Performing %s', mtype)
            raw = np.zeros_like(exc_density.raw)
            raw[idx] = scale * exc_density.raw[idx]
            output_name = output / f'{mtype}_cADpyr.nrrd'
            exc_density.with_data(raw).save_nrrd(output_name)

scale_excitatory_densities('output', region_map, brain_regions, excitatory_mapping, layer_names, exc_density)

def set_acronyms_to_zero(output_name, region_map, brain_regions, nrrd, acronyms):
    L.info('Setting acronyms to zero, outputting %s', output_name)
    ids = get_ids(region_map, acronyms)
    idx = np.nonzero(np.isin(brain_regions.raw, list(ids)))
    raw = nrrd.raw.copy()
    raw[idx] = 0.
    nrrd.with_data(raw).save_nrrd(output_name)


acronyms = sum(layer_names.values(), [])
set_acronyms_to_zero('./output/Generic_Excitatory_Neuron_MType_Generic_Excitatory_Neuron_EType.nrrd', region_map, brain_regions, exc_density, acronyms)
set_acronyms_to_zero('./output/Generic_Inhibitory_Neuron_MType_Generic_Inhibitory_Neuron_EType.nrrd', region_map, brain_regions, inh_density, acronyms)
