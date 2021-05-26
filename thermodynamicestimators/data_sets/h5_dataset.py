import h5py
import torch
import numpy as np
import tables



class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, filepath, N_i):
        self.filepath = filepath
        # self.f = h5py.File(filepath, 'r')

        # with tables.open_file(filepath, 'r', driver="H5FD_CORE", driver_core_backing_store=0) as h5_file:
        #     self.data = torch.Tensor(np.asarray(h5_file.root['data']))
        # self.file = h5py.
        self._N_i = N_i
        # assert(torch.sum(N_i) == self.len())
        self._normalized_N_i = (self._N_i / torch.sum(N_i)).double()


    def __len__(self):
        with h5py.File(self.filepath, 'r') as f:
            length = f['data'].shape[0]
        return length


    def __getitem__(self, item):
        with h5py.File(self.filepath, 'r') as f:
            data_item = f['data'][item]
        return data_item


    @property
    def normalized_N_i(self):
        """The relative number of samples taken per state (`torch.Tensor`)

        This is the normalized N_i, so that the number of samples per state sum
        up to 1.
        For use with batch-wise iteration, to not have to re-calculating this on
        every batch.
        """
        return self._normalized_N_i


    @property
    def N_i(self):
        """The number of samples taken per thermodynamic state"""
        return self._N_i