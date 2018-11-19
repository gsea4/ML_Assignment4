import numpy as np
import matplotlib.pyplot as plt

X = np.array([1,2,3,4,5,6])
states = np.array([0,1])

modelTable = np.array([[1/6, 1/6, 1/6, 1/6, 1/6, 1/6],[1/10, 1/10, 1/10, 1/10, 1/10, 5/10]])
A = np.array([[0.95, 0.05],[0.1, 0.9]])

rolls = []
die = []
current_state = 0
for i in range(1000):
    face = np.random.choice(X, p=modelTable[current_state])
    rolls.append(face)
    die.append(current_state)

    r = np.random.random()
    if r > A[current_state][current_state]:
        current_state ^= 1
    

rolls = np.array(rolls)
die = np.array(die)

# k,m = np.unique(rolls, return_counts = True)
# print(np.asarray((k,m)).T)
# z,l = np.unique(die, return_counts=True)

phi_1 = -np.log(0.5) - np.log(modelTable[0,rolls[1]-1])

def phi_t(z,x):
    return -np.log(modelTable[z,x])

def psi_t(z, z_prev):
    return -np.log(A[z_prev, z])

K = 2
T = rolls.shape[0]
T1 = np.empty((K,T))
T2 = np.empty((K,T))

T1[:, 0] = -np.log(modelTable[0, rolls[i]-1])
T2[:, 0] = 0

for i in range(1, T):
    # T1[:,i] = np.min(T1[:,i-1] -np.log(A) - np.log(modelTable[np.newaxis, :, rolls[i]-1]), 1)
    # T2[:,i] = np.argmin(T1[:,i-1] -np.log(A) - np.log(modelTable[np.newaxis, :, rolls[i]-1]), 1)
    T1[:,i] = np.min(T1[:,i-1] -np.log(A) - np.log(modelTable[:, rolls[i]-1]), 1)
    T2[:,i] = np.argmin(T1[:,i-1] -np.log(A) - np.log(modelTable[:, rolls[i]-1]), 1)

z_t = np.array([np.argmin(T1[:,i]) for i in range(T)])

print("Phew")