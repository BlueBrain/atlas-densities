"""
Create a density field for each mtype listed in `app/data/mtypes/density_profiles/mapping.tsv`.

This input file can be replaced by user's custom file of the same format.

Volumetric density nrrd files are created for each mtype listed in either `mapping.tsv`.
This module re-uses the overall excitatory and inhibitory neuron densities computed in
mod:`app/cell_densities`.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import voxcell
from atlas_commons.typing import BoolArray, FloatArray, NDArray
from cgal_pybind import slice_volume
from tqdm import tqdm

from atlas_densities.exceptions import AtlasDensitiesError

if TYPE_CHECKING:
    VoxelIndices = Tuple[NDArray[np.signedinteger], ...]

logging.basicConfig(level=logging.INFO)
logging.captureWarnings(True)
L = logging.getLogger(__name__)


class DensityProfileCollection:
    """
    Class to manage neuron density profiles.

    - load profiles from files
    - assemble a full profile for each specified mtype
    - create a neuron density file (nrrd) for each mtype

    Each mtype is assigned a neuron density profile, that is, a list of cells counts corresponding
    to the layer slices (a. k. a. bins) defined in app/data/meta/layers.tsv.
    The delination of the layer slices, or sub-layers, within the annotated 3D volume of the
    AIBS mouse isocortex is based on the placement hints computed in app/placement_hints.
    Placement hints's distance information allows us to split each layer into slices of
    approximately the same thickness along the cortical axis, as presribed by
    `app/data/meta/layers.tsv`.

    The input neuron density profiles have been obtained in
    "A Derived Positional Mapping of Inhibitory Subtypes in the Somatosensory Cortex", 2019
    by D. Keller et al.

    Lexicon:
        - mtype: morphological type, e.g., L23_DBC, L5_TPC:C or L6_UPC (iscortex mtypes are
            listed in `mapping.tsv`).
        - synapse class: class of the synapses of a pre-synaptic neuron. Either 'inhibitory' or
            'excitatory'.
        - layer slice: layers are sliced along the "cortical depth axis", resulting in sublayers
            of equal thickness.
            In "A Derived Positional Mapping of Inhibitory Subtypes in the Somatosensory Cortex",
            our layer slices are called "bins" so as to avoid confusion with actual rat brain
            slices cut orthogonally to the sagital axis. A layer slice in our sense is laminar
            refinement of a layer, orthogonal to the "cortical depth axis".
    """

    def __init__(
        self,
        mtype_to_profile_map: pd.DataFrame,
        layer_slices_map: Dict[str, range],
        density_profiles: Dict[str, List[float]],
    ) -> None:
        """
        Initialization of DensityProfileCollection

        Args:
            mtype_to_profile_map: dataframe of the following form
                        mtype      synapse_class  layer  profile_name
                    0   L1_DAC     inhibitory     1      L1_DAC
                    1   L1_HAC     inhibitory     1      L1_HAC
                    2   L1_LAC     inhibitory     1      L1_LAC
                    ...

            layer_slices_map: dict of the following form
                {
                    'layer_1': range(93, 100),
                    'layer_2': range(86, 93),
                    'layer_3': range(69, 86),
                    'layer_4': range(60, 69),
                    'layer_5': range(35, 60),
                    'layer_6': range(0, 35),
                }
                This data structure indicates how each layer splits into slices of equal thickness.
                In the above example, there is a total of 100 slices for the whole mouse isocortex.
                Layer 6, for instance, is divided into 35 slices of equal thickness whose indices
                range from 0 to 34.

            stop_layer: name (str) of the layer with the highest slice indices.
                In app/data/meta/layers.tsv, this is 'layer_1' whose slice indices range from 93 to
                99.

            density_profiles: dict of the following form
                {
                    'BP.dat': [0.0, 99.556, 269.83, ...],
                    'BTC.dat': [128.3, 276.48, 439.17, ...],
                    'CHC.dat': [1.82, 33.975, 57.9, ...],
                    'L1_DAC.dat': [0.0, 0.0, 0.0, ...],
                    ...
                }
                The dict keys are names of density profile files and the dict values are lists of
                non-negative float numbers with equal length. The lenght of each list is the total
                number of layer slices defined by `layer_slices_map`. Each list holds the average
                numbers of cells in each slice described in `layer_slices_map`.
        """
        self.mtype_to_profile_map = mtype_to_profile_map
        self.layer_slices_map = layer_slices_map

        # Find the layer with the highest slice indices
        self.stop_layer = max(layer_slices_map, key=lambda key: layer_slices_map[key].stop)

        self.density_profiles = density_profiles
        self.excitatory_mtypes: List[str] = []
        self.inhibitory_mtypes: List[str] = []
        self._collect_density_profiles()

    def _collect_density_profiles(self) -> None:
        """Collect input density profiles and assembles a full profile for each specified mtypes

        This function populates the attribute `self.profile_data` holding the information
        needed to create density nrrd files for each mtype specified in
        `self.mtype_to_profile_map`.

        This function also creates the lists of the excitatory and inhibitory mtypes:
            * self.excitatory_mtypes (e.g., ['L1_DAC', 'L1_HAC', ...])
            * self.inhibitory_mtypes (e.g., ['L2_IPC', 'L2_IPC:A', ...])

        `self.profile_data` is a dict of the form
        {
            'layer_1': {
                'excitatory': <pd.DataFrame>,
                'inhibitory': <pd.DataFrame>,
            },
            'layer_2': {
                'excitatory': <pd.DataFrame>,
                'inhibitory': <pd.DataFrame>,
            },
            ...
        }
        Each dataframe value is either empty or of the form
        (example for layer 4):

                L4_SSC    L4_TPC    L4_UPC
            60  0.090906  0.642046  0.267049
            61  0.090912  0.642041  0.267047
            62  0.090910  0.642044  0.267045
            63  0.090912  0.642043  0.267045
            64  0.090906  0.642048  0.267046
            65  0.090909  0.642044  0.267047
            66  0.090909  0.642048  0.267043
            67  0.090909  0.642047  0.267044
            68  0.090909  0.642043  0.267048

        The row indices are the indices of the layer slices as defined by
        `self.layer_slices_map`. Each column holds the proportion of cells
        in each slice for the corresponding mtype. For instance, around
        9% of the cells in the slice 60 (layer 4) are cells of mtype L4_SSC.
        The sum over each line should be 1.0.
        """
        L.info("Collecting density profiles ...")
        self.profile_data = {
            layer: {"excitatory": pd.DataFrame(), "inhibitory": pd.DataFrame()}
            for layer in self.layer_slices_map
        }
        for _, row in self.mtype_to_profile_map.iterrows():
            # To handle the special case of layer 2/3, we need to split "2,3"
            for layer_index in str(row.layer).split(","):
                layer = "layer_" + str(layer_index)
                range_ = self.layer_slices_map[layer]
                if row.synapse_class == "excitatory":
                    self.excitatory_mtypes.append(row.mtype)
                elif row.synapse_class == "inhibitory":
                    self.inhibitory_mtypes.append(row.mtype)
                self.profile_data[layer][row.synapse_class][row.mtype] = self.density_profiles[
                    row.profile_name
                ][slice(range_.start, range_.stop)]

        # Set DataDrame index with layer slice indices and normalize rows to
        # get mtype proportions for each layer slice
        for layer, range_ in self.layer_slices_map.items():
            for synapse_class in ["excitatory", "inhibitory"]:
                data_frame = self.profile_data[layer][synapse_class]
                if data_frame.empty:
                    continue
                data_frame.index = range_
                self.profile_data[layer][synapse_class] = data_frame.div(
                    data_frame.sum(axis=1), axis=0
                ).fillna(0.0)
                # Check for each slice if there are cells from at least one mtype
                for row_index, row in self.profile_data[layer][synapse_class].iterrows():
                    if np.sum(row) == 0.0:
                        warnings.warn(
                            f"No {synapse_class} cells assigned to slice {row_index} of {layer}"
                        )

    @classmethod
    def load(
        cls,
        mtype_to_profile_map_path: Union[str, Path],
        layer_slices_path: Union[str, Path],
        density_profiles_dirpath: Union[str, Path],
    ) -> "DensityProfileCollection":
        # fmt: off
        '''
        Load data files, build and return a DensityProfileCollection

        Args:
            mtype_to_profile_map_path: path to the .tsv file describing which neuron density
                profiles should be associated to which mtype.
                The content of such file looks like this (excerpt):
                    mtype	   sclass	layer	file
                    L1_DAC	   INH	    1	    L1_DAC
                    L1_HAC	   INH	    1	    L1_HAC
                    L1_LAC	   INH	    1	    L1_LAC
                    L1_NGC-DA  INH	    1	    L1_NGC-DA
                    ...

            layer_slices_path: path to the .tsv file defining the layer slices
                Each layer is split into several slices of equal thickness.
                Slices are identified by a unique index. The content of such a file
                looks like this:
                    layer	from  upto
                    6 	    0	  35
                    5	    35	  60
                    4	    60	  69
                    3	    69	  86
                    2	    86	  93
                    1	    93	  100
                Here layers and slices are ordered according to cortical depth.

            density_profiles_dirpath: path to the directory containing the neuron density profiles
                under the form of .dat files (e.g., BP.dat, BTC.dat, etc.). Each file contains
                a single column of non-negative float numbers, one cell number for each slice
                defined in `layer_slices_path`.

        Returns:
            DensityProfileCollection object.
        '''
        # fmt: on
        mtype_to_profile_map = pd.read_csv(str(mtype_to_profile_map_path), sep=r"\s+")

        def _get_synapse_class_longname(short_name: str) -> str:
            if short_name == "EXC":
                return "excitatory"
            if short_name == "INH":
                return "inhibitory"
            raise AssertionError(f"Unrecognized synapse class {short_name}")

        L.info("Loading density profiles from files ...")
        mtype_to_profile_map = mtype_to_profile_map.rename(
            columns={"sclass": "synapse_class", "file": "profile_name"}
        )
        mtype_to_profile_map["synapse_class"] = list(
            map(_get_synapse_class_longname, mtype_to_profile_map["synapse_class"])
        )
        # Get a list of profile names without duplicates
        density_profile_filenames = list(dict.fromkeys(mtype_to_profile_map["profile_name"]))
        density_profiles = {
            filename: list(np.loadtxt(Path(density_profiles_dirpath, filename + ".dat")))
            for filename in density_profile_filenames
        }

        L.info("Loading layer slice ranges from file %s ...", layer_slices_path)
        layer_slices_df = pd.read_csv(str(layer_slices_path), sep=r"\s+")
        layer_slices_map = {
            f"layer_{str(row['layer'])}": range(row["from"], row["upto"])
            for _, row in layer_slices_df.iterrows()
        }

        return cls(mtype_to_profile_map, layer_slices_map, density_profiles)

    @staticmethod
    def slice_layer(
        layer_mask: BoolArray,
        annotation: voxcell.VoxelData,
        vector_field: FloatArray,
        slice_count: int,
    ) -> NDArray[np.integer]:
        """
        Split `layer_mask` into `slice_count` slices of approximately equal thickness.

        The numbering of slices follows the stream of `vector_field`: vectors flow through slices
        with increasing indices.

        The splitting is based on the positions of voxels on the streamlines of `vector_field` and
        the streamlines lengths.

        Args:
            layer_mask: 3D boolean mask of the layer to split with shape (W, H, D) where W, H and
                D are integer dimensions.
            annotation: annotated volume of the brain region of interest. It holds an int array of
                shape (W, H, D).
            vector_field: 3D unit vector field defined on the layer domain. It holds a float32 array
                of shape (W, H, D).

        Returns:
            An int array of shape (W, H, D) where every voxel in `layer_mask` is labeled by a
            slice index in [1, `slice_count`]. Voxels labeled by i define the i-th slice of
            `layer mask`.
        """

        return slice_volume(
            layer_mask,
            # default offset can be of type float if voxcell<=3.0.1
            np.asarray(annotation.offset, dtype=np.float32),
            annotation.voxel_dimensions,
            vector_field,
            thicknesses=[1.0] * slice_count,
            resolution=0.5,
        )

    def compute_layer_slice_voxel_indices(
        self,
        annotation: voxcell.VoxelData,
        region_map: voxcell.RegionMap,
        metadata: Dict,
        direction_vectors: NDArray[np.float32],
    ) -> Dict[int, VoxelIndices]:
        """
        Compute the voxel indices of each layer slice defined in `self.layer_slices_map`.

        Placement hints (see atlas_densities.placement_hints), i.e., distances along direction
        vectors from each voxel to each layer boundary, are used to split each layer in slices of
        roughly equal thickness. The number of slices per layer is determined by
        `self.layer_slices_map`.

        Args:
            annotation: VoxelData object holding an int array of shape (W, H, D) where W, H and D
                are integer dimensions; this array is the annotated volume of the brain region of
                interest.
            region_map: RegionMap object to navigate the brain regions hierarchy.
            metadata: dict describing the region of interest and its layers.
            direction_vectors: 3D vector field defined over the `annotation` volume. This is a
                float32 array of shape (W, H, D, 3).

        Returns:
            dict whose keys are layer slice indices and whose values are the voxel indices for
            each slice. To each slice index corresponds a tuple (X, Y, Z) whose components are
            arrays of shape (N, ) where N is the number of voxels in the corresponding slice.
        """
        # pylint: disable=too-many-locals

        L.info("Computing the mask of each layer ...")
        isocortex_ids = region_map.find(
            metadata["region"]["query"],
            attr=metadata["region"]["attribute"],
            with_descendants=metadata["region"]["with_descendants"],
        )
        layers_info = metadata["layers"]
        layer_ids = {
            layers_info["names"][i]: region_map.find(
                layers_info["queries"][i],
                attr=layers_info["attribute"],
                with_descendants=layers_info["with_descendants"],
            )
            for i in range(len(layers_info["names"]))
        }
        layer_masks = {
            layer_name: np.isin(annotation.raw, list(layer_ids[layer_name] & isocortex_ids))
            for layer_name in layers_info["names"]
        }

        L.info("Computing the voxel indices of each layer slice ...")

        layer_slice_voxel_indices: Dict[int, VoxelIndices] = {}
        vector_field = np.asarray(direction_vectors, dtype=np.float32)
        for layer_name, range_ in tqdm(self.layer_slices_map.items()):
            slices = self.slice_layer(
                layer_masks[layer_name],
                annotation,
                vector_field,
                len(range_),
            )
            for i, slice_index in enumerate(range_, 1):
                layer_slice_voxel_indices[slice_index] = np.where(slices == i)
                if len(next(iter(layer_slice_voxel_indices[slice_index]))) == 0:
                    raise AtlasDensitiesError(
                        f"Slice with index {slice_index} is empty. Cannot compute mtype density"
                        f" reliably"
                    )

        return layer_slice_voxel_indices

    def create_density(
        self,
        mtype: str,
        synapse_class: str,
        synapse_class_density: voxcell.VoxelData,
        layer_slice_voxel_indices: Dict[int, VoxelIndices],
        output_dirpath: Union[str, Path],
    ) -> None:
        """
        Create and save to file a density field for the specified mtype.

        The density nrrd file is saved in `output_dirpath` under the name `mtype`_density.nrrd.

        Args:
            mtype: the morphological cell type for which the creation of a density nrrd
                file is requested (e.g., L23_DBC, L5_TPC:C or L6_UPC).
            synapse_class: class of the synapses of a pre-synaptic neuron.
                Either 'inhibitory' or 'excitatory'.
            synapse_class_density: volumetric density of the neurons with class
                `synapse_class`: This is a float array of shape (W, H, D) where
                (W, H, D) is the shape of the underlying annotated volume.
            layer_slice_voxel_indices: the list of voxel indices in each layer slice, see
                :meth:`mtype_densities.DensityProfileCollection.compute_layer_slice_voxel_indices`
            output_dirpath: directory path where to save the created density file.
        """
        density = np.zeros_like(synapse_class_density.raw)
        for layer, range_ in self.layer_slices_map.items():
            if mtype in self.profile_data[layer][synapse_class].columns:
                layer_density_profile = self.profile_data[layer][synapse_class][mtype]
                for index in range_:
                    slice_voxel_indices = layer_slice_voxel_indices[index]
                    density[slice_voxel_indices] = (
                        layer_density_profile[index]
                        * synapse_class_density.raw[slice_voxel_indices]
                    )

        synapse_class_density.with_data(density).save_nrrd(
            str(Path(output_dirpath, mtype + "_density.nrrd"))
        )

    def create_mtype_densities(  # pylint: disable=too-many-arguments
        self,
        annotation: voxcell.VoxelData,
        region_map: voxcell.RegionMap,
        metadata: Dict,
        direction_vectors: NDArray[np.float32],
        output_dirpath: Union[str, Path],
        excitatory_neuron_density: Optional[voxcell.VoxelData] = None,
        inhibitory_neuron_density: Optional[voxcell.VoxelData] = None,
    ) -> None:
        """
        Create and save to file a density field for each specified mtype.

        Density nrrd files are saved in `output_dirpath` under the name `mtype`_density.nrrd.

        Args:
            annotation: VoxelData object holding an int array of shape (W, H, D) where W, H and D
                are integer dimensions; this array is the annotated volume of the brain region of
                interest.
            region_map: RegionMap object to navigate the brain regions hierarchy.
            metadata: dict describing the region of interest and its layers.
            direction_vectors: 3D vector field defined over the `annotation` volume. This is a
                float32 array of shape (W, H, D, 3).
            excitatory_neuron_density: VoxelData holding the density field of excitatory neurons.
                This array is a float array of shape (W, H, D)
            inhibitory_neuron_density_path: VoxelData holding the density field of inhibitory
                neurons. This array is a float array of shape (W, H, D).
            output_dirpath: directory path where to save the created density files.
        """

        layer_slice_voxel_indices = self.compute_layer_slice_voxel_indices(
            annotation, region_map, metadata, direction_vectors
        )
        Path(output_dirpath).mkdir(exist_ok=True)

        if excitatory_neuron_density is not None:
            L.info(
                "Creating density files for the %d excitatory mtypes ...",
                len(self.excitatory_mtypes),
            )
            for mtype in tqdm(self.excitatory_mtypes):
                self.create_density(
                    mtype,
                    "excitatory",
                    excitatory_neuron_density,
                    layer_slice_voxel_indices,
                    output_dirpath,
                )
        if inhibitory_neuron_density is not None:
            L.info(
                "Creating density files for the %d inhibitory mtype ...",
                len(self.inhibitory_mtypes),
            )
            for mtype in tqdm(self.inhibitory_mtypes):
                self.create_density(
                    mtype,
                    "inhibitory",
                    inhibitory_neuron_density,
                    layer_slice_voxel_indices,
                    output_dirpath,
                )
