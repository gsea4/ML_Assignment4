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

k,m = np.unique(rolls, return_counts = True)
print(np.asarray((k,m)).T)

z,l = np.unique(die, return_counts=True)