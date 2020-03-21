plt.subplot(3,1,2)
plt.plot(t, v)
plt.ylabel(f"$m/s$")

plt.subplot(3,1,3)
plt.plot(t,x)
plt.ylabel(f"$m$")
plt.xlabel("t")
plt.show()

"""
Velocity-Verlet
"""
v = np.zeros((len(t),2))
x = np.zeros((len(t),2))
a = np.zeros((len(t),2))

v[0] = [0,0]
x[0] = [v_ri,v_rj]

for i in range(n-1):
    a_ = akselerasjon(x[i])
    a[i]= [a_,-a_]
    x[i+1] = x[i] + v[i] * dt + (1/2) * a[i] * dt**2
    a_new = akselerasjon(x[i+1])
    a[i+1] = [a_new, -a_new]
    v[i+1] = v[i] + 1/2 * (a[i] + a[i+1]) * dt

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
