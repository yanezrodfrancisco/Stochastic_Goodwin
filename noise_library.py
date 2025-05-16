import numpy as np
import numba


@numba.jit(nopython=True)
def fp(x, params):
   u, v, p, n= x
   rho, gamma, kappa, lamda, alpha, sigma, delta, mu, bu, bv= params
   return p*alpha*(1-p/sigma) + delta

@numba.jit(nopython=True)
def fn(x, params):
   u, v, p, n= x
   rho, gamma, kappa, lamda, alpha, sigma, delta, mu, bu, bv= params
   return n*delta*(1-n/mu)

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
   #return bu/p*(rho*v-gamma)
   return -u/p

@numba.jit(nopython=True)
def gv(x, params):
   u, v, p, n= x
   rho, gamma, kappa, lamda, alpha, sigma, delta, mu, bu, bv= params
   #return bv*(1-lamda*u)/(n*p*kappa)
   return -v/p



@numba.jit(nopython=True)
def zero(x, params): return 0

@numba.jit(nopython=True)
def one(x, params): return 1


@numba.jit(nopython=True)
def HeunPred(f, g, x, sg, noise, dt, params):
     return f(x, params)+(g(x, params)*sg*noise)*dt**-0.5

@numba.njit()
def Heun_solution(T, dt, var, params, sg1, noise_frec, mid_step:int=100):

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
         predictor=np.array([HeunPred(fu, gu,   x_0, sg1, NoiseP[i//noise_frec], dt, params),
                             HeunPred(fv, gv,   x_0, sg1, NoiseP[i//noise_frec], dt, params),
                             HeunPred(fp, one,  x_0, sg1, NoiseP[i//noise_frec], dt, params),
                             HeunPred(fn, zero, x_0, sg1, NoiseP[i//noise_frec], dt, params)])
         
         U[(i+1)//mid_step]=x_0[0]+0.5*dt*(predictor[0]+HeunPred(fu, gu,  x_0+dt*predictor, sg1, NoiseP[i//noise_frec], dt, params))
         V[(i+1)//mid_step]=x_0[1]+0.5*dt*(predictor[1]+HeunPred(fv, gv,  x_0+dt*predictor, sg1, NoiseP[i//noise_frec], dt, params))
         P[(i+1)//mid_step]=x_0[2]+0.5*dt*(predictor[2]+HeunPred(fp, one, x_0+dt*predictor, sg1, NoiseP[i//noise_frec], dt, params))
         Noise[(i+1)//mid_step]=NoiseP[i//noise_frec]*sg1 + delta


     return U, V, P, Noise
