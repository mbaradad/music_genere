import numpy as np

NUMBER_OF_TAGS = 256

lines = np.genfromtxt('../styles.csv', delimiter=",", dtype=None)
styles = list()
for i in range(NUMBER_OF_TAGS):
  styles.append(lines[i][0])

a = np.load('../prediction.npz')
for j in range(0,15):
  example = a['arr_'+str(j)]
  if(len(example) != 256): continue
  values = [i[1] for i in sorted(enumerate(example), key=lambda x:x[1],reverse=True)]
  indices = [i[0] for i in sorted(enumerate(example), key=lambda x:x[1],reverse=True)]
  print(list(styles[i] for i in indices[0:10] ))
  print(list(values[0:]))