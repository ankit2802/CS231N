import matplotlib.image as mpimg
import os
import numpy as np
import time
import h5py
t=time.time()
data = np.array([mpimg.imread(name) for name in os.listdir('../train')], dtype=np.float64)
data = data.reshape(35116,196608)
h5f = h5py.File('img_data.h5', 'w')
h5f.create_dataset('dataset_1', data=data)
h5f.close() 
print time.time()-t

