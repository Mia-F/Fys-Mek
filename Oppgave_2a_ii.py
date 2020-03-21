"""
Euler
"""
for i in range(n-1):
    a_ = akselerasjon(x[i])
    a[i+1]= [a_,-a_]
    v[i+1] = v[i] + a[i+1] * dt
    x[i+1] = x[i] + v[i] * dt

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
Euler chromer
- her ser det ut til å være en feil
"""

v = np.zeros((len(t),2))
x = np.zeros((len(t),2))
a = np.zeros((len(t),2))

v[0] = [0,0]
x[0] = [v_ri,v_rj]

for i in range(n-1):
    a_ = akselerasjon(x[i])
    a[i+1]= [a_,-a_]
    v[i+1] = v[i] + a[i+1] * dt
    x[i+1] = x[i] + v[i+1] * dt


plt.subplot(3,1,1)
plt.title("Euler Chromer")
plt.plot(t, a)
plt.ylabel(f"$m/s^2$")
