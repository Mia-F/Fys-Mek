import numpy as np
import matplotlib.pyplot as plt

v_ri = 1.5
v_rj = 0

n = 10000
T = 5
dt = T/n

t = np.linspace(0,T, n)

v = np.zeros((len(t),2))
x = np.zeros((len(t),2))
a = np.zeros((len(t),2))

v[0] = [0,0]
x[0] = [v_ri,v_rj]

def akselerasjon(x):

    r = abs(x[0] - x[1])

    return 24*(2*abs(r)**(-12) - \
    (r)**(-6)) * (x[0] - x[1]) / (r)**2
