import json
import sys
import noise_functions as my
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='Palatino Linotype', size=12)

#Lecture of the configuration file
config_path='config.json'
try:
    with open(config_path, 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print(f"Error: Configuration file '{config_path}' not found.")
    sys.exit(1)

#Extraction of the aprameters from the configuration file
params= list(config["parameters"].values())
x_0 = tuple(config["initial_conditions"].values())
params.append(config["noise_parameters"]["d"])
sg1 = config["noise_parameters"]["g"]
frec_ruido = config["noise_parameters"]["noise_frec"]
T, dt, mid_step= tuple(config["numerical_integration_parameters"].values())

#Execution of the simulations
_ = my.Heun_solution(2, dt, var=x_0, params=params, sg1=sg1, pasos=frec_ruido, mid_step=mid_step)
U, V, P, NoiseP= my.Heun_solution(int(T), dt, var=x_0, params=params, sg1=sg1, pasos=frec_ruido,  mid_step=mid_step)
t=np.linspace(0, T*mid_step*dt, len(P))

#Representation of the results
fig, (ax1, axn)= plt.subplots(1, 2, figsize=(16, 9))
axn.remove()
ax2 = plt.subplot(222)
ax3 = plt.subplot(426)
ax4 = plt.subplot(428)

ax1.plot(U, V)
ax1.set(xlabel='Salary\n(a)', ylabel='Employment')


rho, gamma, kappa, lamda, alpha, sigma, delta, mu, bu, bv, d= params
ax2.hlines(lamda*bu, 0, max(t), 'k', alpha=0.6)
ax2.hlines(rho*bv/(gamma*mu), 0, max(t), 'k', alpha=0.6) #This value is only true if variable n is equal or close to mu

ax2.plot(t, P)
ax2.set(xlabel='Time (t.u.)\n(b)', ylabel='Productivity')

ax3.plot(t, U, 'red')
ax4.plot(t, V, 'b')

ax3.set_ylabel('Salary', color='red')
ax3.tick_params(axis='y', labelcolor='red')
ax4.tick_params(axis='y', labelcolor='b')
ax4.set_ylabel('Employment', color='blue')
ax3.set(xlabel='Time (t.u.)\n(c)')
ax4.set(xlabel='Time (t.u.)\n(d)')

fig.tight_layout()
