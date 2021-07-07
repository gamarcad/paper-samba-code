# SAMBA: a generic framework for Secure federAted Multi-armed BAndits
This project contains the implementation of Samba, a secure framework for multi-armed bandits designed by
Radu Ciucanu, Pascal Lafourcade, Gael Marcadet, and Marta Soare.

## Installation Guide
This section describes the way to install Samba in a local environment.

### Requirements
To run Samba, we need to have some library already installed on your system.
To install them, 
```shell
apt-get install build-essential git python3 python3-pip python3-venv -y
```

### Installation of Samba

Samba needs some libraries with a specific version in order to guarantee
a functional program over the time.

The first thing to do is to download the source from git by typing:
```shell
git clone https://github.com/gamarcad/paper-samba-code.git
cd paper-samba-code
```

To limit the impact of these libraries on the installed system, we provide a
Python virtual environment, allowing to install all libraries we need in a
specific location out of your python system configuration.

To install the virtual environment, type the command
```shell 
./setup_virtual_env.sh
```
This script creates and initializes the virtual environment with all desired libraries installed.
Once executed, a new directory called 'samba' will be created, containing python binaries.
The virtual environment can be managed with these commands to enter in your terminal:

- `source samba/bin/activate` opens an instance in the virtual environment.
    Once executed, you will notice `(samba)` in front of your prompt (e.g., `(samba) user@machine $`), meaning that
    you are in the virtual environment. Note that all your commands installed on
    your system will be usable as usual.
    
- `deactivate` exits the virtual environment.
    Works only when you are in the virtual environment, this command will only affect
    the instance in the virtual environment, meaning that the current path will not change.

### Troubleshooting
When an error occurs during the first steps of the installation, please start by updating your system,
by doing `sudo apt-get update -y && sudo apt-get upgrade -y`.

If an error occurs during *cryptography* library installation, causing by a missing`setuptools_rust` module,
you need to update the pip module inside the environment by doing `samba/bin/python3 -m pip install --upgrade pip` command.
The solution is adapted from the issue claimed at https://github.com/pyca/cryptography/issues/5753

## Benchmarks Execution Guide
This section describes how to reproduce the plots used in the article.

Once the virtual environment is installed, you will be able to run the benchmarks.
Enter the virtual environment and run the following commands in the presented order:

- `sudo python3 bench.py -h`
    This command displays details of each parameter needed by the benchmark program.
    You can complete each required option, or to execute the next command which is basically
    a shell script which executes the `bench.py` python script with appropriate arguments. 
    Some files must be indicated to the program:
    - The `--configs <filename>` option indicates the file containing all pairs of numbers of arms K and budget N to consider,
        in a CSV file format. 
      
    - The `--data <filename>` option indicates the file containing arms probabilities in a textual format, one probability by line.
        
    - Finally, the `--parameters <filename>` option indicates the file containing parameters in a CSV format, used to feed algorithms that
        requires some parameters.

    - To decrease the needed execution time, bench.py supports the multi-processing.
By default, it uses the number of core available on the system.
You can decide the number of core to use with `--cpu <nb_core>` option.

    - In case you want to focus on some algorithm, you can indicate the `--algorithms <IDalgo1> <IDalgo2> ...`
    where supported algorithms, detailed here by name and ID are:
    
| Algorithm | ID |
| --- | --- |
| Pursuit | pursuit | 
| Softmax - Boltzmann | boltzmann | 
| epsilon-greedy | e-greedy |  
| epsilon-greedy decreasing | e-decreasing-greedy |
| Thompson Sampling | thompson-sampling |
| UCB | ucb |


    
- `sudo ./launch_benchmarks.sh`
    This command executes the benchmark with pre-filled parameters.
    Note the `--run` option, which indicates to the program to run the benchmarks.
    Benchmarks may take a long time depending on configurations and data.
    To ensure that the given configuration is well stated, by default the benchmarks program
    displays the configuration based on the inputs.
    When you are sure about the configuration, then indicate the `--run` option.

    Additionally, this scripts calls the plotting program which generates plots, 
    under the `output` directory.
    Hence, no more action is necessary to get the plots.
  
## Only Plotting Guide
The benchmarks take a long time as the number of configuration to be tested is high.
In this repository, we provide our precomputed results located under the `precomputed_results`
folder, in a JSON format.

You can generate plots we used in the paper by executing the `sudo ./only_plots.sh` shell script which 
displays the plots under `precomputed_results/plots` folder.
There are 4 plots:
- `rewards.pdf` plots the cumulative rewards with respect to the budget N.
- `scalability_N.pdf` plots the execution time while varying the budget N with K = 100.
- `scalability_K.pdf` plots the execution time while varying the number of arms K with N = 100000.
- `zoom.pdf` plots the execution time for each entity involved in the process with N = 100000 and respectively K = 10 and K = 100.

## Detailed Data Files Format and Generation
We detailed in this section the format of each file given to the benchmarks python script.

### Data Files 
Data files, located under `data` folder, contains the list of probabilities used in the multi-armed bandits algorithms.
Each probability is placed alone on a single line.
At the beginning of the data file is located the number of probabilities present in the
concerned data file.

In our experiments, we used MovieLens and Jester datasets, that we
preprocessed in order to adapt them for multi-armed bandits.

| Dataset | Location |
| --- | --- |
| MovieLens| [http://files.grouplens.org/datasets/movielens/ml-100k/](http://files.grouplens.org/datasets/movielens/ml-100k/)
| Jester | [http://eigentaste.berkeley.edu/dataset/](http://eigentaste.berkeley.edu/dataset/)

#### MovieLens
The MovieLens dataset is composed of 100 movies rated by 943 number of users, where each rating is between 1 and 5. To
adapt MovieLens for multi-armed bandits, we compute for each movie his the number of ratings greater or equals than a
given threshold, fixed at 4 , divided by the number of ratings of the concerned movie.

#### Jester
The Jester dataset contains 100 jokes rated by 24983 users, where each rating is between -10 for a very bad joke and 10 for
a very good one. Like the MovieLens preprocessing, for each joke, we divide the number of rating greater or equals of a
given threshold, fixed at 4.5 , by the number of ratings for the concerned joke.


To pre-processed these files, we used the `preprocessing.py` scripts.
You do not need to execute this script to launch the experiment as long as data files exists.
In case you want to launch experiments on data generated with a different threshold, then modify
the thresholds variables in the `preprocesing.py` python script and execute it with `python3 preprocessing.py` 
without arguments.


### Parameter Files
Parameter files located under `parameters` folder, contains parameters optimized for datasets we used in our experiments.
Each parameter file is a CSV file, containing the header row *tau, beta, epsilon* columns and a single data row containing
parameter in order defined in the header.

We used these parameters thanks to the `parameters_suggestion.py` script which, for a given dataset, 
suggests the set of parameters which returns the highest cumulative reward.


### Config Files
In the paper, we show experiment both on the execution time and the cumulative reward.
Depending on of the desired experiments, the list of possible combinations of number of arms K and budget N to test
will not be the same.
It is why we create a configuration file, located under `configs` folder, in a CSV format, with a header row *N, K* and data rows, one for each 
combination to test. 

