import numpy as np
import matplotlib.pyplot as plt

sigma = 1
epsilon = 3

r = np.linspace(0.9, 3, 1000)
u = 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)

p = np.linspace(0.8, 3, 100)
p_y = np.zeros(len(p))

s = np.linspace(0.8, 1,100)
s_x = np.zeros(len(s))

f, diagram = plt.subplots(1)
plt.plot(p,p_y, "--", color="gray")
b = diagram.plot(s,s_x, "-k")
plt.text(0.9, 0.1, "$\sigma$",  fontsize=12)
plt.plot(r, u,"-r")

for i in range(len(u)):
    if u[i] == np.min(u):
        r_0 = diagram.plot(r[i], u[i], "ko")
        print(np.min(u))
        plt.text(r[i] * (1+ 0.05 ), u[i] * (1 + 0.3), "$r_{0}$",  fontsize=12)
        a = diagram.arrow(r[i], u[i], 0,-u[i])
        plt.text(1.15, -0.5, "$\epsilon$",  fontsize=12)
        break
    else:
        continue


plt.grid()
plt.xlabel("r$\sigma$")
y = plt.ylabel("V [J/C]")
plt.title("Lennard-Jones potential \n $\sigma$ = 1")
plt.xlim((0.8,3))
plt.ylim((-3.2, 7))
plt.show()

#sigma = [ 1.5, 0.95]
sigma = [0.95]
colors = ['coral', 'goldenrod', 'blueviolet']

k = 0
for i in sigma:
    u = 4 * epsilon * ((i/r)**12 - (i/r)**6)
    k +=1
    for j in range(len(u)):
        if u[j] == np.min(u):
            print(r[j], u[j])
            print(r[j] - i)
        else:
            continue
    plt.subplot(1,len(sigma),k)
    #if k == 1:
        #y = linspace()
        #plt.plot()
    plt.plot(p,p_y, "--k")
    plt.title(f"$\sigma$ = {i}")
    plt.grid()
    plt.plot(r, u, color=f"{colors[k-1]}", label=f"sigma = {i}")



plt.show()
