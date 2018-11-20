import numpy as np
import matplotlib.pyplot as plt

X = np.array([1,2,3,4,5,6])
states = np.array([0,1])

modelTable = np.array([[1/6, 1/6, 1/6, 1/6, 1/6, 1/6], [1/10, 1/10, 1/10, 1/10, 1/10, 5/10]])
A = np.array([[0.95, 0.05],[0.1, 0.9]])

rolls = []
die = []
current_state = 0
for i in range(100):
    face = np.random.choice(X, p=modelTable[current_state])
    rolls.append(face)
    die.append(current_state)

    r = np.random.random()
    if r > A[current_state][current_state]:
        current_state ^= 1
    
rolls = np.array(rolls)
die = np.array(die)

K = 2
T = rolls.shape[0]
Alpha = np.empty((K,T))
T1 = np.empty((K,T))
T2 = np.empty((K,T))

Alpha[:, 0] = np.log(modelTable[0, rolls[0]-1]) + np.log(0.5)
# Alpha[:, 0] = modelTable[:, rolls[0]-1] * 0.5
T2[:, 0] = 0
Z1 = np.zeros(T)

# Forward
for i in range(1, T):
    for k in range(K):
        # T1[k,i] = np.max([T1[q,i-1] + np.log(A[k,q]) + np.log(modelTable[k, rolls[i]-1]) for q in range(K)])
        # T2[k,i] = np.argmax([T1[q,i-1] + np.log(A[k,q]) + np.log(modelTable[k, rolls[i]-1]) for q in range(K)])
        Alpha[k,i] = np.log(modelTable[k, rolls[i]-1]) + np.sum(np.log([Alpha[q,i-1] * A[q,k] for q in range(K)]))
        # Alpha[k,i] = modelTable[k,rolls[i]-1]*np.sum([Alpha[q,i-1] * A[q,k] for q in range(K)])
        pass

# z_t = np.array([np.argmax(Alpha[range(K),i]) for i in range(T)])
# normalized = T1[0,:]/np.sum(T1, axis=0)

# Backward
Beta = np.empty((K,T))
Beta[:, T-1] = 1

for i in range(T-2, 0, -1):
    for k in range(K):
        Beta[k,i] = np.sum(np.log(modelTable[k, rolls[i]-1]) + [Beta[q,i+1] + np.log(A[k,q]) for q in range(K)])

x = range(0,1000)

# plt.plot(x, z_t, color='orange')
plt.plot(x, die)
plt.show()
print("Phew")