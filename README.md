# Stochastic_Goodwin
In this repository we can found the codes and scripts necessary to reproduce the results obtained in our paper <NAME_NECESARRY>

The use of the scripts and its content is described as follow:

* `setup.sh`: Bash script that creates the pythonenviroment with the dependencies used in the other programms
* `noise_library.py`: Here we have the functions of our model defined and the stochastic methods to solve them
* `stochastic_simulation.py`: Principal programm of the repository. Uses the functions defined in the library, saves the data and create Figures to analize the data obtained
* `launch.py`: Script that automatizes the simulation launchments.
* `bash.sh`: Script that activate the enviroment and execute the simulation
* `analysis.py`: Read all the files `.npz` of a given directory and realizes the data analysis done in the paper, returning a `.json` file
