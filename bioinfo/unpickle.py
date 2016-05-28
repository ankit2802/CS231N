import numpy as np
import h5py
import time
t=time.time()
h5f = h5py.File('img_data.h5','r')
b = h5f['dataset_1'][:]
h5f.close()
print time.time()-t,b.shape
