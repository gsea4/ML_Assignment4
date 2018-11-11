import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm as dist

testing_images_file = open('t10k-images.idx3-ubyte', 'rb')
testing_images = testing_images_file.read()
testing_images_file.close()

testing_labels_file = open('t10k-labels.idx1-ubyte', 'rb')
testing_labels = testing_labels_file.read()
testing_labels_file.close()

testing_images = bytearray(testing_images)
X = np.array(testing_images[16:]).reshape(10000, 784)

testing_labels = bytearray(testing_labels)
L = np.array(testing_labels[8:])

def k_means(K, X, M_init):
    # Assignment Step
    # Z_t = [for k in range(K)]
    pass

means_1 = np.random.choice(10000, 10)
M_init_1 = X[means_1]

labels_matrix = np.zeros((10000,10))

best_distances = np.negative(np.ones((10000)))
temp = np.array([dist(M_init_1[k] - X[0]) for k in range(10)])

for i in range(X.shape[0]):
    dist_list = np.array([dist(M_init_1[k] - X[i]) for k in range(10)])
    best_label = np.argmin(dist_list)
    if best_distances[i] == -1:
        best_distances = dist_list[best_label]
        labels_matrix[i,best_label] = 1
    if dist_list[best_label] < best_distances[i]:
        best_distances[i] = dist_list[best_label]
        labels_matrix[i,:] = 0
        labels_matrix[i,best_label] = 1    


