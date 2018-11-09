import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

data = np.loadtxt('crash.txt')
X = data[:,0]/np.max(data[:,0])
T = (data[:,1]-np.min(data[:,1]))/(np.max(data[:,1]) - np.min(data[:,1]))

# Compute Gram matrix K for Squared Exponential and Exponential kernels
K_se = np.array([np.exp(-1*((X - X[i])**2/(2 * np.var(X)))) for i in range(X.shape[0])])
K_e = np.array([np.exp(-1*(np.abs(X - X[i]))/np.std(X)) for i in range(X.shape[0])])

Beta = 1/((100000-np.min(data[:,1]))/(np.max(data[:,1]) - np.min(data[:,1])))**2
C_se = K_se + Beta * np.identity(X.shape[0])
C_e = K_e + Beta * np.identity(X.shape[0])
a = inv(C_se).dot(T)

x_star = np.linspace(0,1,100)
y_star = np.array([np.sum(a * np.exp(-1*(x_star[i] - X)**2/(2 * np.var(X))))for i in range(x_star.shape[0])])

plt.plot(x_star, y_star)
plt.scatter(X,T, color = 'red')
plt.show()