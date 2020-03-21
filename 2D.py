import numpy  as np
import matplotlib.pyplot as plt

r_i = 1.5
r_j = 0
m = 39.95

n = 10000
T = 5
dt = T/n

t = np.linspace(0,T, n)

t2D = np.zeros((len(t),2))

v1 = np.zeros((len(t),2))
v2 = np.zeros((len(t),2))

x1 = np.zeros((len(t),2))
x2 = np.zeros((len(t),2))

a1 = np.zeros((len(t),2))
a2 = np.zeros((len(t),2))

x1[0] = [r_i, 0]
x2[0] = [r_j, 0]

for i in range(1,len(t)):
    t2D[i] = t2D[i-1] + dt

def akselerasjon(x1, x2):
    r = np.linalg.norm(x1 - x2)
    R = (24*(2*abs(r)**(-12) - (r)**(-6)) * (x1 - x2) / (r)**2)
    return R

for i in range(n-1):
    a_ = akselerasjon(x1[i], x2[i])
    a2_ =  -a_
    a1[i+1]= a_
    a2[i+1]=  a2_

    v1[i+1] = v1[i] + a1[i+1] * dt
    v2[i+1] = v2[i] + a2[i+1] * dt

    x1[i+1] = x1[i] + v1[i+1] * dt
    x2[i+1] = x2[i] + v2[i+1] * dt

print(x1)

plt.plot(x1[:,0], x1[:,1])
plt.plot(x2[:,0], x2[:,1])
plt.show()
