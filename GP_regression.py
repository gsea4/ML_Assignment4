import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

data = np.loadtxt('crash.txt')
X = data[:,0]/np.max(data[:,0])
T = (data[:,1]-np.min(data[:,1]))/(np.max(data[:,1]) - np.min(data[:,1]))

def kernel_functions(X, sigma, sq_expo = True):
    if sq_expo == True:
        return np.array([np.exp(-1*((X - X[i])**2/(2 * sigma**2))) for i in range(X.shape[0])])
    else:
        return np.array([np.exp(-1*(np.abs(X - X[i]))/sigma) for i in range(X.shape[0])])

def predict(x_star, X ,sigma, a, sq_expo = True):
    if sq_expo == True:
        return np.array([np.sum(a * np.exp(-1*(x_star[i] - X)**2/(2 * sigma**2))) for i in range(x_star.shape[0])])
    else:
        return np.array([np.sum(a * np.exp(-1*(np.abs(x_star[i] - X))/sigma)) for i in range(x_star.shape[0])])

# Compute Gram matrix K for Squared Exponential and Exponential kernels
sigma = 0.1
K_se = kernel_functions(X, sigma, True)
K_e = kernel_functions(X, sigma, False)

Beta = 1/((20-np.min(data[:,1]))/(np.max(data[:,1]) - np.min(data[:,1])))**2
C_se = K_se + Beta * np.identity(X.shape[0])
C_e = K_e + Beta * np.identity(X.shape[0])
a_se = inv(C_se).dot(T)
a_e = inv(C_e).dot(T)

x_star = np.linspace(0,1,100)
y_star = predict(x_star, X, sigma, a_e, False)

# plt.plot(x_star, y_star, color = 'green')
# plt.scatter(X,T, color = 'red')
# plt.show()

sigma_list = np.linspace(0.01,1,100)

def cross_validate(X, T, sigma, sq_expo ,k):
    idx = np.random.permutation(X.shape[0])
    X, T = X[idx], T[idx]
    folds = np.array(np.array_split(X, k))
    total_msq = 0
    for i in range(k):
        test_mask = np.isin(X, folds[i])
        train_mask = np.logical_not(test_mask)
        X_train = X[train_mask]
        X_test = X[test_mask]
        T_train = T[train_mask]
        T_test = T[test_mask]

        K_se = kernel_functions(X_train, sigma, sq_expo)
        C_se = K_se + Beta * np.identity(X_train.shape[0])
        a_se = inv(C_se).dot(T_train)

        t_star = predict(X_test, X_train, sigma, a_se, sq_expo)
        mse = np.sum((t_star - T_test)**2)/t_star.shape

        # plt.scatter(X_test, t_star, color = 'green')
        # plt.scatter(X_train,T_train, color = 'red')
        # plt.show(block = False)
        # plt.pause(0.1)
        # plt.clf()
        total_msq += mse
    return total_msq/k

best_s = -999
best_mse = 10000

best_s2 = -999
best_mse2 = 10000
for s in sigma_list:
    e_sq = cross_validate(X, T, s, True, 5)
    if e_sq < best_mse:
        best_mse = e_sq
        best_s = s
    
    e_exp = cross_validate(X, T, s, False, 5)
    if e_exp < best_mse2:
        best_mse2 = e_exp
        best_s2 = s
print("Best sigma: " + str(best_s))
print("Best sigma2: " + str(best_s2))


sigma = (best_s+best_s2)/2
K_se = kernel_functions(X, sigma, True)
K_e = kernel_functions(X, sigma, False)
print(sigma)
Beta = 1/((20-np.min(data[:,1]))/(np.max(data[:,1]) - np.min(data[:,1])))**2
C_se = K_se + Beta * np.identity(X.shape[0])
C_e = K_e + Beta * np.identity(X.shape[0])
a_se = inv(C_se).dot(T)
a_e = inv(C_e).dot(T)

x_star = np.linspace(0,1,100)
y_star_se = predict(x_star, X, sigma, a_se, True)
y_star_e = predict(x_star, X, sigma, a_se, False)

plt.plot(x_star, y_star_se, color = 'green')
plt.plot(x_star, y_star_e, color = 'orange')
plt.scatter(X,T, color = 'red')
plt.show()