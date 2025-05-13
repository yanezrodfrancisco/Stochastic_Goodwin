# Stochastic_Goodwin
In this repository we can found the codes and scripts necessary to reproduce the results obtained in our paper <NAME_NECESARRY>

The use of the scripts and its content is described as follow:

* `setup.sh`: Bash script that creates the python enviroment with the dependencies used in the other programms
* `noise_library.py`: Here we have the functions of our model defined and the stochastic methods to solve them
* `config.json`: File where all neccesary variables and paramaters for the numerical integration are defined
* `stochastic_simulation.py`: Principal programm of the repository. Uses the functions defined in the library, saves the data and create Figures to analize the data obtained

The SDEs (Stochastic Differential Equations) are defined as:

$$
\frac{dx}{dt} = f(x) + \xi g(x) ~ \left(\xi = N(0, 1)\right)
$$

If $g(x) = 0$, there is not an stochastich term in the equation, if $g(x) = cte$ there is denomitaed additive noise, in any other scenario, the noise is considered multiplicative.

Functions in `noise_library.py` written that start with an `f` (`fu`, `fv`, `fp` and `fn`) are the deterministic functions derived form the Goodwin model without noise (ignoring the constant $d$). Functions `gu` and `gv` are the multiplicative noise functions for `u` and `v` respectively. The remaining functions are the `HeunPred` function, an intern function that calculates the predictor for the Heun method; and the `Heun_solution` function, that calculates the solution for the configuration defined in the `config.json`.

The parameters in the `config.json` are those required for the simulation. **To replicate our results, it is the only file the user should modify**. `initial_conditions` and `parameters` are self-explenatory, are those initial values and parameters defined for our system. In the `noise_parameters`, `noise_frec` is the number of time integration steps where the random value of the noise is constant. The `sampling_interval` is the number of intermedia steps used to calculate a new step. It is important to afford memory nad agilize teh representations of the results. It is expressed also in numerical time steps. For numerical restrictions, the `noise_frec` must be bigger than the `sampling_interval`; and obviously these must be very small compared to the total `steps` in the simulations. `delta_t` is also self-explenatoy, it is the time discrezitation used in the numerical simulation.  

The `stochastic_simulation.py` is only the program used to realize the simulation and extract a graphic analogous Figure 3 of our paper. For the parameters set in our example of `config.json`, with no noise, the results should are these:

![Test Image 3](/Figure_3.png)
