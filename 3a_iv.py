import numpy as np
import matplotlib.pyplot as plt

r = np.linspace(0.8, 3,500)

u3 = 4 * ((1/3)**12 - (1/3)**6)
u = 4  * ((1/r)**12 - (1/r)**6) - u3

x = np.zeros(len(r))

f = 24 * (2*(1/r)**2 - (1/r)**6) *r/(r**2)

plt.title("Kraftgraf")
plt.xlabel("r [$\sigma$]")
plt.plot(r,f)
plt.show()

plt.title("Det skiftede potensialet")
plt.ylabel("V [J/c]")
plt.xlabel("r [$\sigma$]")
plt.plot(r,x)
plt.plot(r,u)
plt.show()
