# =====================================================================================================================
#
# SAMBA: Secure Multi-Armed Bandits Framework
#
# This project contains the implementation of Samba, a secure framework for multi-armed bandits designed by
# Radu Ciucanu, Marta Soare, Pascal Lafourcade and Gael Marcadet.
# =====================================================================================================================

# ============================================================
# INSTALLATION GUIDE
# ------------------
# This section describes the way to install Samba in a local
# environment.
# ============================================================
Samba needs some libraries with a specific version in order to guarantee
a functional program over the time.

To limit the impact of these libraries on the installed system, we provide a
Python virtual environment, allowing to install all libraries we need in a
specific location out of your python system configuration.

To install the virtual environment, we propose a bash script called 'setup_virtual_env.sh'
in order to create and initialized the virtual environment with all desired libraries installed.
Once executed, a new directory called 'samba' will be created, containing python binaries.
The virtual environment can be managed with these commands to enter in your terminal:

    - source samba/bin/activate: Open a instance in the virtual environment.
    Once executed, you will notice (samba) in front of your prompt, meaning that
    you are in the virtual environment. Note that all your commands installed on
    your system will be usable as usual.

    - deactivate: Exit the virtual environment.
    Works only when you are in the virtual environment, this command will only affect
    the instance in the virtual environment, meaning that the current path will not changed.

# ============================================================
# EXECUTE BENCHMARKS
# ------------------
# This section describes the benchmarks process.
# ============================================================
Once the virtual environment is installed, you will be able to run the benchmarks.
Enter to the virtual environment and runs the following commands in the presented order:

    - sudo python3 bench.py -h
    This command displays a details of each parameters needed by the benchmark program.
    You are free to complete each required option, or to execute the next command which is basically
    a filled program. Some files must be indicates to the program:
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

    - sudo launch_experiments.sh
    This command executes the benchmark with pre-filled parameters.
    Note the --run option, which indicates to the program to run the benchmarks.
    Benchmarks may take a long time depending configurations and data.
    To ensure that the given configuration is well stated, by default the benchmarks program
    displays the configuration to execute.
    When you are sure about the configuration, indicate the --run option.

    Additionally, this scripts calls the plotting program which generates plots used in paper.
