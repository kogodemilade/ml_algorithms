import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import *

fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
plt.grid()
# plt.figure(0, figsize=(10, 10), dpi=10)
labels = np.array([1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1])
features = np.array([[0, 1],
                     [5, 6],
                     [-11, -8],
                     [1, 2],
                     [6, 6],
                     [-12, -14],
                    [10, 11],
                    [12, 8],
                    [-2, -7],
                    [-12, 30],
                    [15, 15],
                    [8, -3],
                    [7, 7],
                    [9, 2],
                    [-13, 4],
                    [-20, -16]])
# norm = ax.scatter(features[:, 0], features[:, 1], c=labels, s=120).norm

# random_data = [[randint(-15, 15), randint(-15, 15)] for i in range(10)]

def train(features_matrix=features):
    w1, w2, b = 0, 0, 0
    weight_vector = np.array([0.0, 0.0, 0.0]).reshape(1, 3)
    origin = np.array([0, 0, 0])

    input_vector = []
    done = False
    while not done:
        i = 0
        for feature in range(features_matrix.shape[0]):
            input_vector = np.array([features_matrix[feature][0], features_matrix[feature][1], 1])
            z = np.dot(weight_vector, input_vector)
            if z > 0:
                yhat = 1
            else:
                yhat = -1
            y = labels[feature]
            if y*yhat <= 0:
                weight_vector += 0.01*y*input_vector
                i += 1
        if i == 0:
            done = True
    print(weight_vector)
    x = np.linspace(-10, 10, 100)
    weight_vector = weight_vector[0]
    v = lambda z: (-weight_vector[2] - weight_vector[0]*x)/weight_vector[1]
    ax.plot(x, v(x), linewidth=2)
    ax.scatter(features_matrix[:, 0], features_matrix[:, 1],  c=labels, cmap='cividis')
    # ax.scatter(weight_vector[0][0], weight_vector[0][1], weight_vector[0][2])
    return 0


# def classifier(datapoints: list | tuple, features_matrix=features, labels_vector=labels, k=2, show_graph=True):

print(train(features))
# classifier([[3, 3]])
# classifier([[-2, -4]])
# classifier([[-8, 0]])
# classifier([[8, 9]])
# classifier(random_data)
plt.show()
