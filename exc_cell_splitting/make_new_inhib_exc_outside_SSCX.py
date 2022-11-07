###############################################################################
# Copyright 2022 Blue Brain
# Author: Daniel Keller
# October 21, 2022
# This program makes exc and inh densities with SSCX cut out
# It also remaps exc cells to morphological fractions in those regions
# using the m-type fractions in the cxv file. All etype exc types are the same.

# inputs: annoation atlas and hierarchy
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



#make dictionary of layers in cortex
layer_ids={}
iso_ids = region_map.find('Isocortex', 'acronym', with_descendants=True)
for count1 in range(0,6):
    temp=atlas.get_layer(count1)
    ids=[]
    for id in list(temp[1]):
        if id in iso_ids:
            ids.append(id)
    layer_ids[temp[0]]=ids

def scale_excitatory_densities(output, region_map, brain_regions, mapping, layer_ids, exc_density):
    output = Path(output)

    for layer, df in mapping.iterrows():
        print(layer)
        ids = list(layer_ids[layer])
        
#        for id in ids:
#            print(region_map.get(id, "name"))
 
        idx = np.nonzero(np.isin(brain_regions.raw, ids))

        for mtype, scale in df.iteritems():
            if scale == 0:
                continue
            L.info('Performing %s', mtype)
            raw = np.zeros_like(exc_density.raw)

            raw[idx] = scale * exc_density.raw[idx]

            output_name = './output/'+ mtype+'_cADpyr.nrrd'
            print(output_name)
       
            exc_density.with_data(raw).save_nrrd(output_name)

scale_excitatory_densities('output', region_map, brain_regions, excitatory_mapping, layer_ids, exc_density)

def set_ids_to_zero(output_name, region_map, brain_regions, nrrd, remove_ids):
    L.info('Setting region ids to zero, outputting %s', output_name)

    idx = np.nonzero(np.isin(brain_regions.raw, remove_ids))

    raw = nrrd.raw.copy()
    raw[idx] = 0.
    nrrd.with_data(raw).save_nrrd(output_name)


remove_ids =  [i for entry in layer_ids.values() for i in entry]
set_ids_to_zero('./output/Generic_Excitatory_Neuron_MType_Generic_Excitatory_Neuron_EType.nrrd', region_map, brain_regions, exc_density, remove_ids)
set_ids_to_zero('./output/Generic_Inhibitory_Neuron_MType_Generic_Inhibitory_Neuron_EType.nrrd', region_map, brain_regions, inh_density, remove_ids)
