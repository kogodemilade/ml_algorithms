import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import *

fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
plt.grid()
features = np.array([[0, 1],
                     [5, 6],
                     [11, 8],
                     [1, 2],
                     [6, 6],
                     [12, 14],
                     [10, 11],
                     [12, 8],
                     [2, 7],
                     [12, 15],
                     [15, 15],
                     [8, 3],
                     [7, 7],
                     [9, 2],
                     [13, 4],
                     [20, 16]])
# Feel free to tinker with features


def train(features_matrix=features[0], learning_rate=0.001, epochs=1000000, tolerance=1e-5):
    weight = randint(-1000, 1000)
    bias = randint(-1000, 1000)
    alpha = learning_rate
    j1 = float('inf')
    X = features_matrix[:, 0]
    i = 0
    for epoch in range(epochs):
        i += 1
        yhat_arr = weight * X + bias
        y = np.array(features_matrix[:, 1])
        j2_array = ((yhat_arr - y) ** 2)
        j2 = j2_array.mean() / 2
        print(f"{j2}, {weight}, {bias}")
        if abs(j1 - j2) > tolerance:
            weight -= alpha * np.mean([yhat_arr - y] * X)
            bias -= alpha * np.mean([yhat_arr-y])
            j1 = j2
        else:
            break
        if (i % 2000) == 0 and i > 999:
            fig_, ax_ = plt.subplots(figsize=(10, 10), dpi=100)
            plt.grid()
            x = np.linspace(-10, 20, 100)
            v = lambda x: weight * x + bias
            ax_.scatter(features_matrix[:, 0], features_matrix[:, 1])
            ax_.plot(x, v(x), linewidth=2)

    x = np.linspace(-10, 20, 100)
    v = lambda x: weight * x + bias
    ax.scatter(features_matrix[:, 0], features_matrix[:, 1])
    ax.plot(x, v(x), linewidth=2)
    print(f"Number of iterations: {i}")


print(train(features))
plt.show()
#5GdpVBW36KWdgJ6
