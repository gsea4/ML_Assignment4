import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('crash.txt')
X = data[:,0]/np.max(data[:,0])
T = (data[:,1]-np.min(data[:,1]))/(np.max(data[:,1]) - np.min(data[:,1]))

# Compute Gram matrix K for Squared Exponential and Exponential kernels
K_se = np.array([np.exp(-1*((X - X[i])**2/(2 * np.var(X)))) for i in range(X.shape[0])])
K_e = np.array([np.exp(-1*(np.abs(X - X[i]))/np.std(X)) for i in range(X.shape[0])])

Beta = 1/((20-np.min(data[:,1]))/(np.max(data[:,1]) - np.min(data[:,1])))**2
C_se = K_se + Beta * np.identity(X.shape[0])
C_e = K_e + Beta * np.identity(X.shape[0])
