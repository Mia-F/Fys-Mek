import numpy as np
import matplotlib.pyplot as plt

r = np.linspace(0.9,4,100)

lj3 = 4 *((1/3)**(12) - (1/3)**(6))
p = 4 *((1/r)**(12) - (1/r)**(6)) - lj3

f = (24*(2*(r)**(-12) - (r)**(-6)) / (r)**2)

plt.plot(r,p)
plt.show()

plt.plot(r,f)
plt.show()
