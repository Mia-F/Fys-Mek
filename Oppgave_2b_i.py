import numpy as np
import matplotlib.pyplot as plt

dim = 2
n = 10000
T = 5
dt = T/n

v_ri = 1.5
v_rj = 0

t = np.linspace(0,T, n)
v = np.zeros((len(t),2, dim))
x = np.zeros((len(t),2, dim))
a = np.zeros((len(t),2, dim))
t2D = np.zeros((len(t),2, dim))

x[0] = [[v_ri, 0],[v_rj, 0]]
for i in range(1,len(t)):
    t2D[i] = t2D[i-1] + dt

def akselerasjon2D(x):
    ri = np.sqrt((x[0][0])**2 + (x[0][1])**2 )
    rj = np.sqrt((x[1][0])**2 + (x[1][1])**2 )
    r = abs(ri-rj)
    return 24*(2*abs(r)**(-12)- (r)**(-6)) * (x[0] - x[1]) / (r)**2
"""
for i in range(n-1):
    a_ = akselerasjon2D(x[i])
    a[i+1]= [a_,-a_]
    v[i+1] = v[i] + a[i+1] * dt
    x[i+1] = x[i] + v[i+1] * dt
"""
for i in range(n-1):
    a[i] = akselerasjon2D(x[i])
    x[i+1] = x[i] + v[i] * dt + (1/2) * a[i] * dt**2
    a[i+1] = akselerasjon2D(x[i+1])
    v[i+1] = v[i] + 1/2 * (a[i] + a[i+1]) * dt


if __name__ == "__main__":
    plt.plot(x[:,0,0], x[:,0,1])
    plt.plot(x[:,1,0], x[:, 1,1])
    plt.show()

    r = np.linalg.norm(x[:,0], axis=1)
    plt.title("Avstand mellom atomene")
    plt.xlabel("t")
    plt.ylabel("Avstand")
    plt.plot(t2D[:,0], r)
    plt.show()
