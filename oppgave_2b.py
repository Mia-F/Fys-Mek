#[t ,atom, x, y]
import numpy  as np
import matplotlib.pyplot as plt
from oppgave_3c import*


sigma = 1
#n = 1
#L = 1.7


#n = 2
#L = 3.4

#n = 3
#L = 5.1

#n = 4
#L = 6.8

def kinetisk_energi(v):
    v2 = v**2
    k = np.sum(v2)/2
    return k

#m_atomes = posisjon(n,L)
#n_atom = len(m_atomes)
n_atom = 2

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
#print(x[0])
x[0] = [[0,0,0],[0.95,0,0]]

#r1 = np.linalg.norm(x[0,0] - x[0,1])
#p[0] = 4 *((1/r1)**(12) - (1/r1)**(6)) - 4 *((1/3)**(12) - (1/3)**(6))
#k[0] = kinetisk_energi(v[0])
#v0 = np.random.normal(0, np.sqrt(44), size=(n_atom,3))
#v[0] = v0
def akselerasjon(x,v):
    a = np.zeros((len(x), len(x),3))
    p = 0
    lj3 = 4 *((1/3)**(12) - (1/3)**(6))
    for i in range(len(x)):
        for j in range(i+1,len(x)):
            dr = x[i] - x[j]
            r = np.linalg.norm(dr)
            p += 4 *((1/r)**(12) - (1/r)**(6)) - lj3
            if r < 3*sigma:
                a[i,j] = -(24*(2*(r)**(-12) - (r)**(-6)) * (dr) / (r)**2)
                a[j,i] = -a[i,j]
            else:
                a[i,j] = 0
                a[j,i] = 0
    return np.sum(a,axis = 0),p,x,v


for i in range(n-1):
    print(i)
    a[i], p[i],x[i],v[i] = akselerasjon(x[i], v[i])
    x[i+1] = x[i] + v[i] * dt + (1/2) * a[i] * dt**2
    a[i+1], p[i+1],x[i+1], v[i+1] = akselerasjon(x[i+1], v[i+1])
    v[i+1] = v[i] + 1/2 * (a[i] + a[i+1]) * dt
    k[i+1] = kinetisk_energi(v[i+1])


"""
plt.plot(x[:,0,0], x[:,0,1])
plt.plot(x[:,1,0], x[:,1,1])
plt.title("Posisjonsgraf")
plt.xlabel("x posisjon [$\sigma$]")
plt.ylabel("y posisjon [$\sigma$]]")
plt.show()

r = x[:,1] - x[:,0]
plt.plot(t, r[:,0],color="orange")
plt.title("Avstand mellom atomene")
plt.xlabel("Tid [t]")
plt.ylabel("Avstand [$\sigma$]")
plt.show()
"""
r = np.zeros(len(x))

d = np.zeros(len(x))
for i in range(len(x)):
    for j in range(i+1,len(x)):
        dr = x[i] - x[j]
        r[i] = np.linalg.norm(dr)
        d[i+1] = d[i] + (r[i]-r[0])

msd = d/n_atom

plt.plot(t, msd)
plt.show()

print(msd)



if __name__ == "__main__":
    r = 3
    print(4 *((1/r)**(12) - (1/r)**(6)))
    e = k + p
    plt.plot()
    plt.plot(t,p, label="Potensiell")
    plt.plot(t,k, label="Kinetisk")
    plt.plot(t,e, label="Total")
    plt.title("Energi graf")
    plt.xlabel("Tid [t]")
    plt.legend()
    plt.show()
    infile = open("posisjon4.txt","w")
    for t in range(n):
        infile.write(f"{n_atom}\n")
        infile.write(f"type x y z\n")
        for l in range(n_atom):
            infile.write(f"Ar {x[t,l,0]} {x[t,l,1]} {x[t,l,2]}\n")
    infile.close()
