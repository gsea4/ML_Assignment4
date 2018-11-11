import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm as dist
from PIL import Image

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

def calc_J(Z, M, X, K):
    result = 0
    for k in range(K):
        result += np.sum([Z[i,k] * dist(X[i] - M[k]) for i in range(X.shape[0])])
    return result

# means_1 = np.random.choice(5000, 10)
l0 = np.where(L == 0)[0][1]
l1 = np.where(L == 1)[0][1]
l2 = np.where(L == 2)[0][1]
l3 = np.where(L == 3)[0][1]
l4 = np.where(L == 4)[0][1]
l5 = np.where(L == 5)[0][1]
l6 = np.where(L == 6)[0][1]
l7 = np.where(L == 7)[0][1]
l8 = np.where(L == 8)[0][1]
l9 = np.where(L == 9)[0][1]
means_3 = np.array((l0,l1,l2,l3,l4,l5,l6,l7,l8,l9))
M_init_1 = X[means_3]

labels_matrix = np.zeros((10000,10))
best_distances = np.negative(np.ones((10000)))

counter = 0
previous_j = 0
max_count = 100
M = M_init_1
while counter < max_count:
    for i in range(X.shape[0]):
        dist_list = np.array([dist(k - X[i]) for k in M])
        best_label = np.argmin(dist_list)

        if best_distances[i] == -1:
            best_distances[i] = dist_list[best_label]
            labels_matrix[i,best_label] = 1

        if dist_list[best_label] < best_distances[i]:
            best_distances[i] = dist_list[best_label]
            labels_matrix[i,:] = 0
            labels_matrix[i,best_label] = 1

    j1 = calc_J(labels_matrix, M, X, 10)  
    print(j1)
    if j1 == previous_j:
        max_count = counter
    previous_j = j1

    N = labels_matrix.sum(axis=0)
    M_new = np.zeros((10,X.shape[1]))
    for k in range(10):
        M_new[k] = np.array([labels_matrix[i,k] * X[i] for i in range(X.shape[0])]).sum(axis=0)/N[k]
    M = M_new
    counter += 1

img = Image.fromarray(M[0].reshape((28,28)))
img.show()
img = Image.fromarray(M[1].reshape((28,28)))
img.show()
img = Image.fromarray(M[2].reshape((28,28)))
img.show()
img = Image.fromarray(M[3].reshape((28,28)))
img.show()
img = Image.fromarray(M[4].reshape((28,28)))
img.show()
img = Image.fromarray(M[5].reshape((28,28)))
img.show()
img = Image.fromarray(M[6].reshape((28,28)))
img.show()
img = Image.fromarray(M[7].reshape((28,28)))
img.show()
img = Image.fromarray(M[8].reshape((28,28)))
img.show()
img = Image.fromarray(M[9].reshape((28,28)))
img.show()
