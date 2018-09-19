import numpy as np


n = np.array([-1, 0, 1], [-2, 0, 2], [-1, 0, 1])
m = np.array([202, 119, 254],[106, 119, 11], [186, 48, 250])

r = m*n

print(r[1,1])
