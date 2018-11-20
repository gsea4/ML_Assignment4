import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm as dist
from PIL import Image
import time

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
    counter = 0
    previous_j = 0
    max_count = 100
    M = M_init
    labels_matrix = np.zeros((X.shape[0],K))
    best_distances = np.negative(np.ones((X.shape[0])))

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

        J = calc_J(labels_matrix, M, X, K)  
        print(J)
        if J == previous_j:
            max_count = counter
            return labels_matrix, M
        previous_j = J

        N = labels_matrix.sum(axis=0)
        M_new = np.zeros((K,X.shape[1]))
        for k in range(K):
            M_new[k] = np.array([labels_matrix[i,k] * X[i] for i in range(X.shape[0])]).sum(axis=0)/N[k]
        M = M_new
        counter += 1

def calc_J(Z, M, X, K):
    result = 0
    for k in range(K):
        result += np.sum([Z[i,k] * dist(X[i] - M[k]) for i in range(X.shape[0])])
    return result

def k_means_pp_init(X, K):
    ini_mean = X[np.random.choice(10000,1)]
    M =[ini_mean]
    for k in range(1, K):
        D = np.array([min([dist(x-m)**2 for m in M]) for x in X])
        prob_D = D/D.sum()
        prob_cummul_D = prob_D.cumsum()
        r = np.random.rand()
        new_index = np.where(r < prob_cummul_D)[0][0]
        M.append(X[new_index])
    return np.array(M)

means_1 = X[np.random.choice(10000, 10)]
means_2 = k_means_pp_init(X, 10)
means_3 = X[np.array([np.where(L == i)[0][1] for i in range(10)])]

K = 10
counter = 0
for M_init in (means_1, means_2, means_3):
    labels, M_r = k_means(K, X, M_init)
    img = [Image.fromarray(M_r[i].reshape((28,28))) for i in range(K)]
    for i in range(len(img)):
        img[i].convert('RGB').save("mean_" + str(counter) + "__" + str(i) +".png")
    counter += 1

K = 3
M_init = k_means_pp_init(X, K)
labels, M_r = k_means(K, X, M_init)
img = [Image.fromarray(M_r[i].reshape((28,28))) for i in range(K)]
for i in range(len(img)):
    img[i].convert('RGB').save("mean_k++_" + str(i) +".png")

X0 = X[np.where(labels[:,0] == 1)]
X1 = X[np.where(labels[:,1] == 1)]
X2 = X[np.where(labels[:,2] == 1)]

sample_X0 = X0[np.random.choice(X1.shape[0], 3)]
sample_X1 = X1[np.random.choice(X1.shape[0], 3)]
sample_X2 = X2[np.random.choice(X1.shape[0], 3)]

for sample in (sample_X0, sample_X1, sample_X2):
    img = [Image.fromarray(sample[i].reshape((28,28))) for i in range(3)]
    for i in range(len(img)):
        img[i].show()
        time.sleep(1)
