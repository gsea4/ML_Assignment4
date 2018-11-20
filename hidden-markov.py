import numpy as np
import matplotlib.pyplot as plt

X = np.array([1,2,3,4,5,6])
states = np.array([0,1])

modelTable = np.array([[1/6, 1/6, 1/6, 1/6, 1/6, 1/6], [1/10, 1/10, 1/10, 1/10, 1/10, 5/10]])
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

phi_1 = np.log(modelTable[0,rolls[1]-1]) #-np.log(0.5) - 

def phi_t(z,x):
    return -np.log(modelTable[z,x])

def psi_t(z, z_prev):
    return -np.log(A[z_prev, z])

K = 2
T = rolls.shape[0]
T1 = np.empty((K,T))
T2 = np.empty((K,T))

T1[:, 0] = np.log(modelTable[0, rolls[i]-1])
T2[:, 0] = 0

for i in range(1, T):
#     T1[:,i] = np.min(T1[:,i-1] -np.log(A.T) - np.log(modelTable[np.newaxis, :, rolls[i]-1].T), 1)
#     T2[:,i] = np.argmin(T1[:,i-1] -np.log(A.T) - np.log(modelTable[np.newaxis, :, rolls[i]-1].T), 1)
#     T1[:,i] = np.min(T1[:,i-1] -np.log(A.T) - np.log(modelTable[:, rolls[i]-1]), 1)
#     T2[:,i] = np.argmin(T1[:,i-1] -np.log(A.T) - np.log(modelTable[:, rolls[i]-1]), 1)
    for k in range(K):
        for q in range(K):
            T1[k,i] = np.max(T1[q,i-1] + np.log(A[q,k]) + np.log(modelTable[k, rolls[i]-1]))
z_t = np.array([(T1[1,i])/T1[0,i] for i in range(T)])


x = range(0,100)

plt.plot(x, z_t[:100])
plt.show()
print("Phew")



# def backtrace(z_pred, back, T, labels):
#     prev_cursor = z_pred[-1]

#     for m in np.arange(T)[::-1]:
#         prev_cursor = back[prev_cursor, m]
#         z_pred[m] = prev_cursor

#     if labels is None:
#         return z_pred

#     return [labels[z] for z in z_pred]


# def viterbi(q_size, A, X, B, q_init, labels=None):
#     Q = np.arange(q_size)
#     T = len(X)
#     p = np.zeros(shape=(len(Q), T))
#     back = np.zeros(shape=(len(Q), T), dtype=np.int)
#     z_pred = np.zeros(shape=(T, ), dtype=np.int)

#     for q in Q:
#         p[q, 0] = A[q_init, q] + np.log(B[q, X[0]])
#         back[q, 0] = q_init

#     for t in np.arange(T)[1:]:
#         for q in Q:
#             p[q, t] = np.max([p[qp, t - 1] + np.log(A[qp, q]) + np.log(B[q, X[t]]) for qp in Q])
#             # back[q, t] = np.argmax([p[qp, t - 1] * A[qp, q] for qp in Q])

#     # z_pred[T - 1] = np.argmax([p[q, T - 1] for q in Q])
#     z_t = np.array([np.max(p[:,i])/np.sum(p[:,i]) for i in range(T)])
#     return q, z_t
#     # return backtrace(z_pred, back, T, labels)
# rolls -= 1
# q_hat, z_hat = viterbi(K, A, rolls, modelTable, 0)