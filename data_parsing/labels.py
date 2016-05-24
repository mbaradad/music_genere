import numpy as np

NUMBER_OF_TAGS = 256

example = np.zeros(256);
example[1] = 0.9
example[3] = 0.5
example[10] = 0.3

lines = np.genfromtxt('../styles.csv', delimiter=",", dtype=None)
styles = list()
for i in range(NUMBER_OF_TAGS):
  styles.append(lines[i][0])

indices = [i[0] for i in sorted(enumerate(example), key=lambda x:x[1],
    reverse=True)]
print(list( styles[i] for i in indices[1:5] ))

