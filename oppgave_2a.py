import numpy as np
import matplotlib.pyplot as plt

#v_ri = np.array((1.5, 0))
#v_rj = np.zeros(len(v_ri))

"""
En dimensjon
"""

v_ri = 0
v_rj = 1.5

n = 500
T = 5
dt = T/n

t = np.linspace(0,T, n)

v = np.zeros((len(t),2))
x = np.zeros((len(t),2))
a = np.zeros((len(t),2))
p = np.zeros(len(t))
k = np.zeros(len(t))

v[0] = [0,0]
x[0] = [v_ri,v_rj]

def akselerasjon(x):
    r = abs(x[0] - x[1])
    p = 4 *((1/r)**(12) - (1/r)**(6))
    a = 24*(2*abs(r)**(-12) - (r)**(-6)) * (x[0] - x[1]) / (r)**2
    return a, p

def kinetisk(v):
    return np.sum(v**2)/2


k[0] = kinetisk(v[0])
#Euler
for i in range(n-1):
    a_, p[i] = akselerasjon(x[i])
    a[i+1]= [a_,-a_]
    v[i+1] = v[i] + a[i+1] * dt
    x[i+1] = x[i] + v[i] * dt
    k[i+1] = kinetisk(v[i+1])

a_, p[-1] = akselerasjon(x[-1])


"""
plt.subplot(3,1,1)
plt.title("Euler")
plt.plot(t, a)
plt.ylabel(f"$m/s^2$")

plt.subplot(3,1,2)
plt.plot(t, v)
plt.ylabel(f"$m/s$")

plt.subplot(3,1,3)
plt.plot(t,x)
plt.ylabel(f"$m$")
plt.xlabel("t")
plt.show()
"""
#Euler chromer

v = np.zeros((len(t),2))
x = np.zeros((len(t),2))
a = np.zeros((len(t),2))
p = np.zeros(len(t))
k = np.zeros(len(t))

v[0] = [0,0]
x[0] = [v_ri,v_rj]


k[0] = kinetisk(v[0])
for i in range(n-1):
    a_ , p[i]= akselerasjon(x[i])
    a[i+1]= [a_,-a_]
    v[i+1] = v[i] + a[i+1] * dt
    x[i+1] = x[i] + v[i+1] * dt
    k[i+1] = kinetisk(v[i+1])

a_, p[-1] = akselerasjon(x[-1])

a_, p[-1] = akselerasjon(x[-1])
plt.plot(t,p, label="Potensiell")
plt.plot(t,k, label="kinetisk")
plt.plot(t, k+p, label ="Total")
plt.title(f"Energi graf \n Euler Chromer \n dt = {T/n} ")
plt.legend()
plt.xlabel("t")
plt.ylabel("J")
plt.show()

"""
plt.subplot(3,1,1)
plt.title("Euler Chromer")
plt.plot(t, a)
plt.ylabel(f"$m/s^2$")

plt.subplot(3,1,2)
plt.plot(t, v)
plt.ylabel(f"$m/s$")

plt.subplot(3,1,3)
plt.plot(t,x)
plt.ylabel(f"$m$")
plt.xlabel("t")
plt.show()
"""

v = np.zeros((len(t),2))
x = np.zeros((len(t),2))
a = np.zeros((len(t),2))
p = np.zeros(len(t))
k = np.zeros(len(t))

v[0] = [0,0]
x[0] = [v_ri,v_rj]

#Velocity-Verlet
for i in range(n-1):
    a_ , p[i]= akselerasjon(x[i])
    a[i]= [a_,-a_]
    x[i+1] = x[i] + v[i] * dt + (1/2) * a[i] * dt**2
    k[i] = kinetisk(v[i])
    a_new, p[i+1] = akselerasjon(x[i+1])
    a[i+1] = [a_new, -a_new]
    v[i+1] = v[i] + 1/2 * (a[i] + a[i+1]) * dt
    k[i+1] = kinetisk(v[i+1])
"""
plt.subplot(3,1,1)
plt.title("Velocity-Verlet")
plt.plot(t, a)
plt.ylabel(f"$m/s^2$")

plt.subplot(3,1,2)
plt.plot(t, v)
plt.ylabel(f"$m/s$")

plt.subplot(3,1,3)
plt.plot(t,x)
plt.ylabel(f"$m$")
plt.xlabel("t")
plt.show()
"""
"""
to dimensjoner
"""

dim = 2
n = 10000
T = 5
dt = T/n

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

    return 24*(2*abs(r)**(-12) - (r)**(-6)) * (x[0] - x[1]) / (r)**2


for i in range(n-1):
    a_ = akselerasjon2D(x[i])
    a[i+1]= [a_,-a_]
    v[i+1] = v[i] + a[i+1] * dt
    x[i+1] = x[i] + v[i+1] * dt
"""
"""
plt.plot(x[:,0,0], x[:,0,1])
plt.plot(x[:,1,0], x[:, 1,1])
plt.title("Posisjonsgraf")
plt.xlabel("x posisjon [$\sigma$]")
plt.ylabel("y posisjon [$\sigma$]]")
plt.show()
"""
#t_x = np.linspace(0,T, n)
#t_y = np.linspace(0,T, n)

#t = np.meshgrid(t_x, t_y)
#print(t)
"""
print(f"t2d  = {t2D[0]} \n x = {x[0]}")

tx = []
ty = []
x_x = []
x_y = []
for i in range(len(t)):
    tx.append(t2D[i][0])
    x_x.append(x[i][0])
    ty.append(t2D[i][1])
    x_y.append(x[i][1])
"""
"""
Avstand Mellom atomene
"""
"""
r = np.linalg.norm(x[:,0], axis=1)
plt.title("Avstand mellom atomene")
plt.plot(t2D[:,0], r)
plt.xlabel("Tid [t]")
plt.ylabel("Avstand [$\sigma$]")
plt.show()
"""
#print(np.shape(r))
"""
PLotter x og y retning ting
"""
