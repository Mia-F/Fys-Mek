#[t ,atom, x, y]
import numpy  as np
import matplotlib.pyplot as plt
from oppgave_3c import*


sigma = 1

n = 4
L = 6.8

#m_atomes = posisjon(n,L)
#n_atom = len(m_atomes)
n_atom = 4

n = 500
T = 5
dt = T/n
t = np.linspace(0,T, n)

x = np.zeros((len(t),n_atom,3))
v = np.zeros((len(t),n_atom,3))
a = np.zeros((len(t),n_atom,3))
p = np.zeros(len(t))
k = np.zeros(len(t))


#x[0] = m_atomes
x[0] = [[1,0,0],[0,1,0],[-1,0,0], [0,-1,0]]
#v0 = np.random.normal(0, 2, size=(n_atom,3))
#v[0] = v0
def akselerasjon(x,v):
    a = np.zeros((len(x), len(x),3))
    p = 0
    lj3 = 4 *((1/3)**(12) - (1/3)**(6))
    for i in range(len(x)):
        for j in range(i+1,len(x)):
            dr = x[i] - x[j]
            r = np.linalg.norm(dr)
            dr = dr - np.round(dr/L)*L
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




if __name__ == "__main__":
    r = 3
    print(4 *((1/r)**(12) - (1/r)**(6)))
    e = k + p
    plt.plot()
    plt.plot(t,p)
    plt.plot(t,k)
    plt.plot(t,e)
    plt.show()
    infile = open("posisjon4.txt","w")
    for t in range(n):
        infile.write(f"{n_atom}\n")
        infile.write(f"type x y z\n")
        for l in range(n_atom):
            infile.write(f"Ar {x[t,l,0]} {x[t,l,1]} {x[t,l,2]}\n")
    infile.close()
