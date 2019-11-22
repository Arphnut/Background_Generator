import numpy as np
import matplotlib.pyplot as plt

A = np.zeros((1080, 1920, 3))
for i in range(1080):
    pi = (i/1080)**2
    for j in range(1920):
        if np.random.random() > pi:
            A[i, j] = np.array([0, 0, 0])
        else:
            A[i, j] = np.array([1, 0, 0])

plt.imshow(A)
plt.show()
