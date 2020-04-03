class Atom:
    def __init__(t, integrasjon):
        self.t = t
        self.integrasjon = integrasjon

    def akselerasjon(x):
        r = x[0,0] - x[1,0]
        a_ = 24*(2*abs(r)**(-12) - (r)**(-6)) * (x[0] - x[1]) / (r)**2
        return a_

    def Euler(self):
        self.x, self.v, self.a = x,v,a
        for i in range(n-1):
            a_ = akselerasjon(x[i])
            a[i+1]= [a_,-a_]
            v[i+1] = v[i] + a[i+1] * dt
            x[i+1] = x[i] + v[i] * dt

    def Euler_Chromer(self):
        self.x, self.v, self.a = x,v,a
        for i in range(n-1):
            a_ = akselerasjon(x[i])
            a[i+1]= [a_,-a_]
            v[i+1] = v[i] + a[i+1] * dt
            x[i+1] = x[i] + v[i+1] * dt

    def Velocity_verlet(self):
        self.x, self.v, self.a = x,v,a
        for i in range(n-1):
            a_ = akselerasjon(x[i])
            a[i]= [a_,-a_]
            x[i+1] = x[i] + v[i] * dt + (1/2) * a[i] * dt**2
            a_new = akselerasjon(x[i+1])
            a[i+1] = [a_new, -a_new]
            v[i+1] = v[i] + 1/2 * (a[i] + a[i+1]) * dt
        return x,v,a

    def set_initial_condition(x0, v0):
        self.x, self.v = x,v
        x[0] = x0
        v[0] = v0
        return x, v

    def Solver(self,x0,v0):
        self.x = np.zeros((len(t),2,2)
        self.v = np.zeros((len(t),2,2)
        self.x[0], self.v[0] = set_initial_condition(x0,v0)
        self.x, self.v, self.a = self.integrasjon()
