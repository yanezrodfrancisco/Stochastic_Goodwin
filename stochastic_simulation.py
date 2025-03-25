import numpy as np
import matplotlib.pyplot as plt
import os
plt.rc('font', family='Times New Roman')
import noise_library as my

sigma, mu= 0.95, 1.0
delta=0.01

#IF YOU USE THE SCRIPT bash.sh TO DO MANY SIMULATIONS, YOU MUST DISCOMMENT THIS LINNE
#delta=float(os.getenv("MEAN"))

params=np.array([1, 1, 1, 1, 1e-2, sigma, delta, mu, 2, 1], dtype=float)
rho, gamma, kappa, lamda, alpha, sigma, delta, mu, bu, bv= params



T, dt=int(1e6), 5e-4
sg1, sg2, = 1e-5, 0.0

p_eq=sigma*(0.5+np.sqrt(0.25+delta/(alpha*sigma)))
x_0=np.array([bu/p_eq, bv/p_eq/mu, p_eq, mu], dtype=float)
x_0=np.array([1.001, 0.999, p_eq, mu])

mid_step=50
frec_ruido=int(1e4)


Res= my.StochasticSim(2, dt, var=x_0, params=params, sg1=sg1, pasos=frec_ruido, mid_step=mid_step)
U, V, P, NoiseP= my.StochasticSim(T, dt, var=x_0, params=params, sg1=sg1, pasos=frec_ruido, mid_step=mid_step)





while (min(U)<-20 or min(V)<-10 or max(U)>20 or max(V)>10):
    U, V, P, NoiseP= my.StochasticSim(T, dt, var=x_0, params=params, sg1=sg1, pasos=frec_ruido, mid_step=mid_step)


t=np.linspace(0, T*dt*mid_step, len(P))
fig, (Ax1, ax3)= plt.subplots(1, 2, figsize=(16, 8))
Ax1.remove()
ax1=plt.subplot(321)
ax2=plt.subplot(323)
ax4=plt.subplot(325)

# Configuración de los gráficos principales
ax_1 = ax1.twinx()
ax_1.plot(t, V, 'b', alpha=0.5)
ax1.plot(t, U, 'r', alpha=0.5)
ax1.set(title='Temporal evolution', xlabel='Time (t.u.)')

ax1.set_ylabel('Salary', color='red')
ax1.tick_params(axis='y', labelcolor='r')
ax_1.tick_params(axis='y', labelcolor='b')
ax_1.set_ylabel('Employment', color='blue')

gs=alpha*(1/sigma-1)

ax2.plot(t, NoiseP)
ax2.hlines(gs, t[0], t[-1], color='red', linestyle='--')
# ax2.hlines(-alpha*sigma*0.25, t[0], t[-1], color='red', linestyle='--')
ax2.set(title='Noise in time', xlabel='Time (t.u.)')

# ax3.plot(bu/P, bv/mu/P, 'r.',  markersize=0.1)
ax3.plot(U, V, '.', markersize=0.1)
ax3.set(title='Phase Portrait', xlabel='Salary', ylabel='Employment')


ax4.plot(t, P, 'g')
ax4.hlines(p_eq, 0, t[-1], color='black')
#ax4.hlines(1, 0, t[-1], color='red', alpha=0.5)
ax4.set(title='Productivity')
# ax4.legend()

fig.suptitle(f' $\sigma$ = {sigma}, $\mu$ = {mu}, $T$ = {frec_ruido*dt}, $d$ = {delta}, $g$ = {sg1} ', fontsize=20)


fig.tight_layout()
#DIRECTORIES MUST EXIST
image=f'IMAGE_DIRECTORY/image'
data =f'DATA_DIRECTORY/data'
np.savez(data, np.array([U, V, P, NoiseP]))
fig.savefig(image+'.png')
