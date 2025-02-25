import numpy as np
import matplotlib.pyplot as plt
from random import *

fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
plt.grid()
# plt.figure(0, figsize=(10, 10), dpi=10)
labels = np.array([0, 1, 2, 0, 1, 2])
features = np.array([[0, 1],
                     [5, 6],
                     [-11, -8],
                     [1, 2],
                     [6, 6],
                     [-12, -14]])
norm = ax.scatter(features[:, 0], features[:, 1], c=labels, s=120).norm

random_data = [[randint(-15, 15), randint(-15, 15)] for i in range(10)]


def classifier(datapoints: list | tuple, features_matrix=features, labels_vector=labels, k=2, show_graph=True):
    winner = []
    data_xs = []
    data_ys = []
    for datapoint in datapoints:
        data_x = datapoint[0]
        data_y = datapoint[1]
        distances = np.array(
            [np.sqrt(((data_x - feature[0]) ** 2) + ((data_y - feature[1]) ** 2)) for feature in features_matrix])
        label_dist = np.column_stack((distances, labels_vector))
        sorted_dist = label_dist[label_dist[:, 0].argsort()]
        competitors = sorted_dist[:k, 1]
        winner.append(competitors[competitors[:].argmax()])
        data_xs.append(data_x)
        data_ys.append(data_y)
    print(winner)
    if show_graph:
        ax.scatter(data_xs, data_ys, c=winner, norm=norm, marker='d', s=120)
        plt.draw()


classifier([[3, 3]])
classifier([[-2, -4]])
classifier([[-8, 0]])
classifier([[8, 9]])
classifier(random_data)
plt.show()
