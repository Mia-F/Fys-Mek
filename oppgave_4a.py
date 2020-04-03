import numpy  as np
import matplotlib.pyplot as plt
from oppgave_3c import*
import math

sigma = 1

n = 3
L = 5.1

#n=4
#L = 6.8

#n=5
#L = 8.5

m_atomes = posisjon(n,L)
n_atom = len(m_atomes)
#n_atom = 2

n = 1000
T = 20
dt = T/n
t = np.linspace(0,T, n)

x = np.zeros((len(t),n_atom,3))
v = np.zeros((len(t),n_atom,3))
a = np.zeros((len(t),n_atom,3))
p = np.zeros(len(t))
k = np.zeros(len(t))
Temp_ = np.zeros(len(t))
A = np.zeros(len(t))
ms = np.zeros(len(t))

T_0 = 119.7
Temp_0 = 174

A[0] = 1
x[0] = m_atomes
#x[0] = [[0,0,0],[1.5,0,0]]
"""
x0 = []
infile = open("Ny posisjon.txt", "r")
for line in infile:
    numbers = []
    words = line.split(" ")
    for h in range(3):
        num = float(words[h])
        numbers.append(num)
    x0.append(numbers)
infile.close()

x[0] = x0

v0 = []
infile = open("Ny posisjon.txt", "r")
for line in infile:
    numbers = []
    words = line.split(" ")
    for h in range(3):
        num = float(words[h])
        numbers.append(num)
    v0.append(numbers)
infile.close()
"""

Temp_[0] = Temp_0/T_0
v0 = np.random.normal(0, math.sqrt(Temp_[0]), size=(n_atom,3))
v[0] = v0

def akselerasjon(x,v):
    a = np.zeros((len(x), len(x),3))
    p = 0
    lj3 = 4 *((1/3)**(12) - (1/3)**(6))
    for i in range(len(x)):
        for j in range(i+1,len(x)):
            dr = x[i] - x[j]
            dr = dr - np.round(dr/L)*L
            r = np.linalg.norm(dr)
            p += 4 *((1/r)**(12) - (1/r)**(6)) - lj3
            if r < 3*sigma:
                a[i,j] = -(24*(2*(r)**(-12) - (r)**(-6)) * (dr) / (r)**2)
                a[j,i] = -a[i,j]
            else:
                a[i,j] = 0
                a[j,i] = 0
    return np.sum(a,axis = 0),p

def kinetisk_energi(v):
    v2 = v**2
    k = np.sum(v2)/2
    return k


for i in range(n-1):
    print(i)
    a[i], p[i] = akselerasjon(x[i], v[i])
    x[i+1] = x[i] + v[i] * dt + (1/2) * a[i] * dt**2
    a[i+1], p[i+1] = akselerasjon(x[i+1], v[i+1])
    v[i+1] = v[i] + 1/2 * (a[i] + a[i+1]) * dt
    k[i+1] = kinetisk_energi(v[i+1])
    Temp_[i+1] = 1.0/(3*n_atom)*np.sum(v[i+1]**2)
    A_ = 0
    for j in range(n_atom):
        A_ += np.dot(v[i+1,j], v[0,j])/np.dot(v[0,j], v[0,j])
    A[i+1] = (1/n_atom)*A_
    #ms[i+1] = (1/n_atom) * np.sum((r[i+1]-r[0])**2)
    #print(ms[i+1])


Temp = Temp_*T_0
a_Temp = np.average(Temp)
print(f"Den gennomsnittlige tempraturen er {a_Temp:.2f}K")
line = np.zeros(len(t))
for i in range(len(t)):
    line[i] = 94.4


D = np.trapz(A, t)/3
print(f"D = {D}")



plt.plot(t,A)
plt.title("A")
plt.show()

plt.title("Temperatur")
plt.xlabel("Tid [t]")
plt.ylabel("Temperatur [K]")
plt.plot(t,Temp, label="Temperatur")
plt.plot(t,line, label="94.4K")
plt.legend()
plt.show()
"""
plt.plot(t, ms)
plt.show()
"""
#print(x[-1])
#print(v[-1])

if __name__ == "__main__":
    r = 3
    print(4 *((1/r)**(12) - (1/r)**(6)))
    e = k + p
    plt.plot()
    plt.plot(t,p, label= "p")
    plt.plot(t,k, label= "k")
    plt.plot(t,e, label= "e")
    plt.legend()
    plt.show()
    infile = open("posisjon_2.txt","w")
    for t in range(n):
        infile.write(f"{n_atom}\n")
        infile.write(f"type x y z\n")
        for l in range(n_atom):
            infile.write(f"Ar {x[t,l,0]} {x[t,l,1]} {x[t,l,2]}\n")
    infile.close()

    infile = open("Ny posisjon.txt", "w")
    for i in range(n_atom):
        for j in range(3):
            infile.write(f"{x[-1,i,j]}")
            infile.write(" ")
        infile.write(f"\n")
    infile.close()

    infile = open("Ny fart.txt", "w")
    for i in range(n_atom):
        for j in range(3):
            infile.write(f"{v[-1,i,j]}")
            infile.write(" ")
        infile.write(f"\n")
    infile.close()
