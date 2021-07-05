# SAMBA: a generic framework for Secure federAted Multi-armed BAndits
This project contains the implementation of Samba, a secure framework for multi-armed bandits designed by
Radu Ciucanu, Pascal Lafourcade, Gael Marcadet, and Marta Soare.

## Installation Guide
This section describes the way to install Samba in a local environment.

### Requirements
To run Samba, we need to have some library already installed on your system.
To install them, 
```shell
apt-get update && apt-get upgrade
apt-get install build-essential curl libssl-dev git
apt-get install python3 python3-pip python3-venv
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
In case where Rust is not installed in your system, please type the command `source $HOME/.cargo/env && reset`
which updates your `.bashrc`.
For more details, visit the Rust webpage https://www.rust-lang.org/tools/install.

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

To install the virtual environment, we propose a bash script called `setup_virtual_env.sh`
in order to create and initialize the virtual environment with all desired libraries installed.
Once executed, a new directory called 'samba' will be created, containing python binaries.
The virtual environment can be managed with these commands to enter in your terminal:

- `source samba/bin/activate`: Open an instance in the virtual environment.
    Once executed, you will notice (samba) in front of your prompt, meaning that
    you are in the virtual environment. Note that all your commands installed on
    your system will be usable as usual.
    
- `deactivate`: Exit the virtual environment.
    Works only when you are in the virtual environment, this command will only affect
    the instance in the virtual environment, meaning that the current path will not changed.

## Benchmarks Execution Guide
This section describes the benchmarks process.

Once the virtual environment is installed, you will be able to run the benchmarks.
Enter to the virtual environment and runs the following commands in the presented order:

- `sudo python3 bench.py -h`
    This command displays a details of each parameter needed by the benchmark program.
    You are free to complete each required option, or to execute the next command which is basically
    a script shell which execute the bench.py python script with appropriated arguments. 
    Some files must be indicated to the program:
        --configs <filename> indicates the file containing all couples of numbers of arms K and budget N to consider,
        in a CSV file format.
        --data <filename> indicates the file containing arms probabilities in a textual format, one probability by line.
        --parameters <filename> indicates the file containing parameters in a CSV format, used to feed algorithms that
        requires some algorithms.

    In case you want to focus on some algorithm, you can indicate the --algorithms <IDalgo1> <IDalgo2> ...
    where supported algorithms, detailed here as his name and his ID:
        Pursuit (ID: pursuit),
        Softmax (ID: boltzmann),
        epsilon-greedy (ID: e-greedy),
        epsilon-greedy decreasing (ID: e-decreasing-greedy),
        Thompson Sampling (ID: thompson-sampling)
        UCB (ID: ucb)

    To decrease the needed execution time, bench.py supports the multi-processing.
    By default, it uses the number of core available on the system.
    You can decide the number of core to use with --cpu <nb_core> option.
    
- `sudo launch_experiments.sh`
    This command executes the benchmark with pre-filled parameters.
    Note the --run option, which indicates to the program to run the benchmarks.
    Benchmarks may take a long time depending on configurations and data.
    To ensure that the given configuration is well stated, by default the benchmarks program
    displays the configuration to execute.
    When you are sure about the configuration, indicate the --run option.

    Additionally, this scripts calls the plotting program which generates plots used in paper.


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
Each parameter file is a CSV file, contains the header row *tau, beta, epsilon* columns and a single data row containing
parameter in order defined in the header.

We used these parameters thanks to the `parameters_suggestion.py` script which, for a given dataset, 
suggests the parameter which returns the highest cumulative reward.


### Config Files
In the paper, we show experiment both on the execution time and the cumulative reward.
Depending on of the desired experiments, the list of possible combinations of number of arms K and budget N to test
will not be the same.
It is why we create a configuration file, located under `configs` folder, in a CSV format, with a header row *N, K* and data rows, one for each 
combination to test. 

