from os import listdir
from os.path import isfile, join
import numpy as np

x = list()
y = list()

for f in  [f for f in listdir('.') if '.npz' in f]:
  npzfile = np.load(f)
  x.extend(npzfile['x'])
  y.extend(npzfile['y'])
