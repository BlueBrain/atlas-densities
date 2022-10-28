
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

#packages
import os
import re
from voxcell import VoxelData
from voxcell.nexus.voxelbrain import Atlas
import numpy as np

#load an nrrd
def find_regions(fn_cand, root, format="path"):
    #for fn_cand in fn_regions:
    if 1:
        expected_fn_regions = os.path.join(root, fn_cand)
        if os.path.isfile(expected_fn_regions):
            if format == "path":
                return expected_fn_regions
            elif format == "voxcell":
                return VoxelData.load_nrrd(expected_fn_regions)
            raise ValueError("Unknown format spec: {0}".format(format))
    return None


#make excitatory density
neuron_nrrd= find_regions("neuron_density_v3.nrrd",'./', "voxcell")
inh_nrrd= find_regions("gad67+_density_v3.nrrd",'./', "voxcell")
exc_nrrd=neuron_nrrd
exc_nrrd.raw=neuron_nrrd.raw-inh_nrrd.raw
exc_nrrd.raw=exc_nrrd.raw*(exc_nrrd.raw>0)

#Do remapping

#read in mapping file
import csv
allrows=[]
with open('mapping_cortex_all_to_exc_mtypes.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in reader:
        allrows.append(row)
        #print(', '.join(row))


#load atlas
atlas = Atlas.open('.')
brain_regions = atlas.load_data('brain_regions')


#make output directory
if (os.path.exists('./output')):
    print('output directory already exists')
else:
    os.mkdir('output')


output_brain_regions=atlas.load_data('brain_regions')
PC_morphology_names=allrows[0]

#layer IDs, starting with layer 2 and going to layer 6
allids=[{614454286,614454294,614454296,614454298,614454300,614454302,614454304,614454306},{614454287,614454289,614454291,614454293,614454295,614454297,614454299,614454301,614454303,614454305,614454307},{12995,865,654,1047,1094,950,577,1086,182305701,1035}, {12996,921,702,1070,1128,974,625,1111,182305705,1090}, {12997,12998,686,719,889,929,1038,1062,478,510,1102,2,945,1026,9,461,182305709,182305713,862,893}]

#loop through the layers and write out ME fractions of PCs in those layers
for count1 in range(1,len(allrows)):
    print(count1)
    ids=allids[count1-1]
    
    row=allrows[count1] 
    region_inds=np.transpose(np.nonzero(np.isin(brain_regions.raw, list(ids))))
    for count2 in range(1,len(PC_morphology_names)):
        if row[count2]=='0':
            continue
        scale=float(row[count2])
        output_name='./output/'+(PC_morphology_names[count2])+'_cADpyr.nrrd'
        output_array=0*exc_nrrd.raw
        allval=0
        for ind in region_inds:
            val=exc_nrrd.raw[ind[0],ind[1],ind[2]]
            output_array[ind[0],ind[1],ind[2]]=scale*val
            allval+=val

        output_brain_regions.raw=output_array
        output_brain_regions.save_nrrd(output_name)

#now remove regions from excitatory
#append layer 1
allids.append({12993,793,558,981,480149206,1030,878,450,1006,182305693,873  })
for ids in allids:
    region_inds=np.transpose(np.nonzero(np.isin(brain_regions.raw, list(ids))))
    for ind in region_inds:
        exc_nrrd.raw[ind[0],ind[1],ind[2]]=0
        inh_nrrd.raw[ind[0],ind[1],ind[2]]=0

#save the output
exc_nrrd.save_nrrd('./output/Generic_Excitatory_Neuron_MType_Generic_Excitatory_Neuron_EType.nrrd')
inh_nrrd.save_nrrd('./output/Generic_Inhibitory_Neuron_MType_Generic_Inhibitory_Neuron_EType.nrrd')
