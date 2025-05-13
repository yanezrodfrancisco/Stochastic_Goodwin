# Stochastic_Goodwin
In this repository we can found the codes and scripts necessary to reproduce the results obtained in our paper <NAME_NECESARRY>

The use of the scripts and its content is described as follow:

* `setup.sh`: Bash script that creates the python enviroment with the dependencies used in the other programms
* `noise_library.py`: Here we have the functions of our model defined and the stochastic methods to solve them
* `config.json`: File where all neccesary variables and paramaters for the numerical integration are defined
* `stochastic_simulation.py`: Principal programm of the repository. Uses the functions defined in the library, saves the data and create Figures to analize the data obtained

The SDEs (Stochastic Differential Equations) are defined as:

$$
\frac{dx}{dt} = f(x) + \xi\,g(x)
$$


