from oppgave_2b import *

"""
Potensiell energi
"""
u = np.zeros((len(x), len(x[0])))
def potensiell(x):
    print(r.shape)
    for i in range(len(r)):
        for j in range(len(r[0])):
            r = np.linalg.norm(x,axis = 2)
            for k in range(j + 1, len(r[0])):
                if r[i, j] < 3:
                    u[i, j] = 4 *((1/r[i, k])**(12) - (1/r[i, k])**(6))
                    u[i, k] = u[i, j]

                else:
                    u[i, j] = 0
                    u[i, k] = 0
    return u

U = potensiell(x)


#plt.ylim(-10,10)
plt.plot(t,np.sum(U, axis=1))
plt.show()
"""
plt.plot(x,u)
plt.show()

epsilon = 1
sigma = 1.5

r = abs(x[:,0] - x[:,1])


U = u[:,0]
"""
"""
kinetisk energi
"""
"""
k = 1/2 * v**2

k_t = k[:,0,0] + k[:,1,0]
"""
"""
Total enegi
"""
"""

E = k[:,0,0] + k[:,1,0] + U

plt.plot(t,k_t)
plt.plot(t,U, label="potensiell energi")
plt.plot(t,E)
plt.show()

plt.plot(t,E)
plt.show()
"""
"""
plt.legend()
plt.show()

plt.subplot(1,2,1)
plt.plot(t,k[:,0,0], label="i")
plt.legend()

plt.subplot(1,2,2)
plt.plot(t,k[:,1,0], label="j")
plt.legend()
plt.show()
"""
