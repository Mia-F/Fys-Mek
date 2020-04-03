import matplotlib.pyplot as plt
import numpy as np

#tilfelle 1
r_i = [0,0]
r_j = [1.5,0]
"""
#tilfelle 2
r_i = [0,0]
r_j = [0.95,0]
"""
T = 5
n = 100
dt = T/n

t = np.linspace(0,T,n)
x = np.zeros((len(t),2,2))
v = np.zeros((len(t),2,2))
a = np.zeros((len(t),2,2))
p = np.zeros(len(t))
k = np.zeros(len(t))

def akselerasjon(x):
    r = x[0,0] - x[1,0]
    pot = 4 *((1/r)**(12) - (1/r)**(6))
    a_ = 24*(2*abs(r)**(-12) - (r)**(-6)) * (x[0] - x[1]) / (r)**2
    return a_, pot

def kinetisk_energi(v):
    v2 = v**2
    print(v2)
    k = np.sum(v2)/2
    return k

x[0] = [r_j,r_i]

r_1 = r_j[0] - r_i[0]
p[0] = 4 *((1/r_1)**(12) - (1/r_1)**(6))
k[0] = kinetisk_energi(v[0])

#Euler
for i in range(n-1):
    a_, p[i+1] = akselerasjon(x[i])
    a[i+1]= [a_,-a_]
    v[i+1] = v[i] + a[i+1] * dt
    x[i+1] = x[i] + v[i] * dt
    k[i+1] = kinetisk_energi(v[i+1])

plt.plot(t,a[:,0,0])
plt.show()

E = k + p

#plt.plot(t,k)
#plt.plot(t,p)
plt.plot(t,E, label="Euler")
#plt.show()

#Euler chromer
t = np.linspace(0,T,n)
x = np.zeros((len(t),2,2))
v = np.zeros((len(t),2,2))
a = np.zeros((len(t),2,2))
p = np.zeros(len(t))
k = np.zeros(len(t))

x[0] = [r_j,r_i]

r_1 = r_j[0] - r_i[0]
p[0] = 4 *((1/r_1)**(12) - (1/r_1)**(6))
k[0] = kinetisk_energi(v[0])
for i in range(n-1):
    a_, p[i+1] = akselerasjon(x[i])
    a[i+1]= [a_,-a_]
    v[i+1] = v[i] + a[i+1] * dt
    x[i+1] = x[i] + v[i+1] * dt
    k[i+1] = kinetisk_energi(v[i+1])

E = k + p

#plt.plot(t,k)
#plt.plot(t,p)
plt.plot(t,E, label="Euler-Cromer")
#plt.show()

#velcity verlet
t = np.linspace(0,T,n)
x = np.zeros((len(t),2,2))
v = np.zeros((len(t),2,2))
a = np.zeros((len(t),2,2))
p = np.zeros(len(t))
k = np.zeros(len(t))

x[0] = [r_j,r_i]

r_1 = r_j[0] - r_i[0]
p[0] = 4 *((1/r_1)**(12) - (1/r_1)**(6))
k[0] = kinetisk_energi(v[0])

for i in range(n-1):
    a_, p[i] = akselerasjon(x[i])
    a[i]= [a_,-a_]
    x[i+1] = x[i] + v[i] * dt + (1/2) * a[i] * dt**2
    a_new, p[i+1] = akselerasjon(x[i+1])
    a[i+1] = [a_new, -a_new]
    v[i+1] = v[i] + 1/2 * (a[i] + a[i+1]) * dt
    k[i+1] = kinetisk_energi(v[i+1])

E = k + p

#plt.plot(t,k)
#plt.plot(t,p)
plt.title("Total energi for det ulike integrasjons metodene")
plt.xlabel("Tid")
plt.plot(t,E, label="Velocity Verlet")
plt.legend()
plt.show()
