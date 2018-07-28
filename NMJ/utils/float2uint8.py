from __future__ import print_function, division
import numpy as np
import h5py

filename = ['volume_0.h5', 'volume_1.h5', 'volume_2.h5']
for idx in range(len(filename)):
    data = np.array(h5py.File(filename[idx], 'r')['main'])
    print("volume shape: ", data.shape)
    data = (data*255).astype(np.uint8)
    hf = h5py.File('volume_uint8_'+str(idx)+'.h5','w')
    hf.create_dataset('main', data=data)
    hf.close()
