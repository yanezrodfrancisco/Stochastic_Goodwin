import numpy as np
import numba

@numba.jit(nopython=True)
def Euler(f, g1, x, sg1, noise1, dt, params):
    return f(x, params)*dt+(g1(x, params)*sg1*noise1)*dt**0.5

@numba.jit(nopython=True)
def Heun(f, g1, x, sg1, noise1, dt, params):
    F= lambda f, g1, x, sg1, noise1, dt: f(x, params)+(g1(x, params)*sg1*noise1)*dt**-0.5
    Fx=F(f, g1, x, sg1, noise1, dt)
    return 0.5*dt*(Fx+F(f, g1, x+dt*Fx, sg1, noise1, dt))

@numba.jit(nopython=True)
def fp(x, params):
   u, v, p, n= x
   rho, gamma, kappa, lamda, alpha, sigma, delta, mu, bu, bv= params
   return p*alpha*(1-p/sigma) + delta

@numba.jit(nopython=True)
def fu(x, params):
   u, v, p, n= x
   rho, gamma, kappa, lamda, alpha, sigma, delta, mu, bu, bv= params
   return (u-bu/p)*(rho*v-gamma)-u*alpha*(1-p/sigma) - delta*u/p

@numba.jit(nopython=True)
def fv(x, params):
   u, v, p, n= x
   rho, gamma, kappa, lamda, alpha, sigma, delta, mu, bu, bv= params
   return (v-bv/(n*p))*(1-lamda*u)/kappa-v*alpha*(1-p/sigma) - delta*v/p

@numba.jit(nopython=True)
def gu(x, params):
   u, v, p, n= x
   rho, gamma, kappa, lamda, alpha, sigma, delta, mu, bu, bv= params
   return -u/p

@numba.jit(nopython=True)
def gv(x, params):
   u, v, p, n= x
   rho, gamma, kappa, lamda, alpha, sigma, delta, mu, bu, bv= params
   return -v/p



@numba.jit(nopython=True)
def zero(x, params): return 0

@numba.jit(nopython=True)
def one(x, params): return 1


#YOU CAN SUSBTITUTE HERE Heun -> Euler IF YOU PREFER TO USE EULER-MARUYAMA METHOD
@numba.jit(nopython=True)
def unew(x, s1, sg1, dt, params): return Heun(fu, g1=gu, x=x, sg1=sg1, noise1=s1, dt=dt, params=params)

@numba.jit(nopython=True)
def vnew(x, s1, sg1, dt, params): return Heun(fv, g1=gv, x=x, sg1=sg1, noise1=s1, dt=dt, params=params)

@numba.jit(nopython=True)
def pnew(x, s1, sg1, dt, params): return Heun(fp, g1=one, x=x, sg1=sg1, noise1=s1, dt=dt, params=params)



@numba.njit()
def StochasticSim(T, dt, var, params, sg1, pasos, mid_step):

    rho, gamma, kappa, lamda, alpha, sigma, delta, mu, bu, bv= params

    NoiseP=np.random.normal(0, 1, T)

    U=np.empty(T)
    V=np.empty(T)
    P=np.empty(T)
    Noise=np.empty(T)

    U[0]=var[0]
    V[0]=var[1]
    P[0]=var[2]


    for i in range(int(mid_step*(T-1))):
        x_0=np.array([U[i//mid_step], V[i//mid_step], P[i//mid_step], mu])

        U[(i+1)//mid_step]=x_0[0]+unew(x_0, NoiseP[i//pasos], sg1, dt, params)
        V[(i+1)//mid_step]=x_0[1]+vnew(x_0, NoiseP[i//pasos], sg1, dt, params)
        P[(i+1)//mid_step]=x_0[2]+pnew(x_0, NoiseP[i//pasos], sg1, dt, params)
        Noise[(i+1)//mid_step]=NoiseP[i//pasos]*sg1 + delta
    return U, V, P, Noise
