import numpy as np
from voxcell import VoxelData


class MockedLoader:
    def __init__(self, args_to_values_dict):
        self.dict = args_to_values_dict


class Mocked_get_region_mask(MockedLoader):
    def __call__(self, acronym, annotation, hierarchy):
        if acronym.startswith("@"):
            import re

            acronym_regexp = re.compile(acronym[1:])
            matches = [acro for acro in self.dict if acronym_regexp.match(acro) is not None]
            return np.any([self(m, annotation, hierarchy) for m in matches], axis=0)
        else:
            return self.dict[acronym]


class Mocked_VoxelData_load_nrrd(MockedLoader):
    def __call__(self, filename):
        return VoxelData(self.dict[filename], voxel_dimensions=(1.0, 1.0, 1.0))


class MockxelData(VoxelData):
    def __init__(self, saved_files_dict, *args, **kwargs):
        self._saved = saved_files_dict
        super().__init__(*args, **kwargs)

    def with_data(self, newdata):
        return self.__class__(self._saved, newdata, self.voxel_dimensions)

    def save_nrrd(self, fname):
        self._saved[fname] = self.raw
