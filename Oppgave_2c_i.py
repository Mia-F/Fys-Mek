from Oppgave_2b_i import*
import numpy as np
import matplotlib.pyplot as plt

K = np.zeros(len(v))
U = np.zeros(len(v))

def kinetisk_energi(v):
    v2 = v**2
    k = np.sum(v2)/2
    return k

for i in range(len(v)):
    K[i] = kinetisk_energi(v[i])
    dr = x[i,0] - x[i,1]
    r = np.linalg.norm(dr)
    U[i] = 4 *((1/r)**(12) - (1/r)**(6))

E = U + K


plt.plot(t, K, label="K" )
plt.plot(t, U, label="U")
plt.plot(t,E, label = "E")
plt.legend()
plt.show()
