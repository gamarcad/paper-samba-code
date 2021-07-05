##################################################
# file: launch_experiment.sh
# author: Gael Marcadet <gael.marcadet@etu.univ-orleans.fr>
# description: Launch protocol experiment with configurations.
##################################################

# ================================================
# Set up parameters
# ================================================
CPU=8
REWARDS_NB_ITER=150
BENCHMARKS_NB_ITER=50

OUTPUT_DIRECTORY='output'
PLOTS_OUTPUT_DIRECTORY="$OUTPUT_DIRECTORY/plots/"
JESTER_REWARDS_OUTPUT_DIRECTORY="$OUTPUT_DIRECTORY/rewards/jester"
JESTER_BENCHMARKS_OUTPUT_DIRECTORY="$OUTPUT_DIRECTORY/benchmarks/jester"
MOVIE_LENS_REWARDS_OUTPUT_DIRECTORY="$OUTPUT_DIRECTORY/rewards/movie-lens"
MOVIE_LENS_BENCHMARKS_OUTPUT_DIRECTORY="$OUTPUT_DIRECTORY/benchmarks/movie-lens"

REWARDS_CONFIGS='configs/rewards_configs.csv'
BENCHMARKS_CONFIG='configs/benchmarks_configs.csv'

MOVIE_LENS_DATA="data/MovieLens.txt"
JESTER_DATA="data/JesterLarge.txt"

MOVIE_LENS_PARAMS="parameters/MovieLens.csv"
JESTER_PARAMS="parameters/JesterLarge.csv"

# ================================================
# Runs benchmarks on MovieLens
# ================================================
# Execute experiments with arguments
# note the -u command used to do not awaiting to write in stdin, stdout and stderr
python3 -u bench.py \
      --output $MOVIE_LENS_BENCHMARKS_OUTPUT_DIRECTORY\
      --data $MOVIE_LENS_DATA\
      --configs $BENCHMARKS_CONFIG\
      --nb_iterations $BENCHMARKS_NB_ITER\
      --parameters $MOVIE_LENS_PARAMS\
      --cpu $CPU\
      --run

# ================================================
# Runs benchmarks on Jester
# ================================================
# Execute experiments with arguments
# note the -u command used to do not awaiting to write in stdin, stdout and stderr
python3 -u bench.py \
      --output $JESTER_BENCHMARKS_OUTPUT_DIRECTORY\
      --data $JESTER_DATA\
      --configs $BENCHMARKS_CONFIG\
      --nb_iterations $BENCHMARKS_NB_ITER\
      --parameters $JESTER_PARAMS\
      --cpu $CPU\
      --run

# ================================================
# Runs rewards on MovieLens
# ================================================
# Execute experiments with arguments
# note the -u command used to do not awaiting to write in stdin, stdout and stderr
python3 -u bench.py\
      --output $MOVIE_LENS_REWARDS_OUTPUT_DIRECTORY\
      --data $MOVIE_LENS_DATA\
      --configs $REWARDS_CONFIGS\
      --nb_iterations $REWARDS_NB_ITER\
      --parameters $MOVIE_LENS_PARAMS\
      --standards_only\
      --cpu $CPU\
      --run

# ================================================
# Runs rewards on Jester
# ================================================
# Execute experiments with arguments
# note the -u command used to do not awaiting to write in stdin, stdout and stderr
python3 -u bench.py\
      --output $JESTER_REWARDS_OUTPUT_DIRECTORY\
      --data $JESTER_DATA\
      --configs $REWARDS_CONFIGS\
      --nb_iterations $REWARDS_NB_ITER\
      --parameters $JESTER_PARAMS\
      --standards_only\
      --cpu $CPU\
      --run

# ================================================
# Plots
# ================================================
python3 plot_bench.py \
    --movie-lens-rewards $MOVIE_LENS_REWARDS_OUTPUT_DIRECTORY\
    --movie-lens-benchmarks $MOVIE_LENS_BENCHMARKS_OUTPUT_DIRECTORY\
    --jester-rewards $JESTER_REWARDS_OUTPUT_DIRECTORY\
    --jester-benchmarks $JESTER_BENCHMARKS_OUTPUT_DIRECTORY\
    --output $PLOTS_OUTPUT_DIRECTORY
