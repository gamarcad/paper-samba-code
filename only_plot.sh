##################################################
# file: launch_benchmarks.sh
# author: Gael Marcadet <gael.marcadet@etu.univ-orleans.fr>
# description: Launch protocol experiment with configurations.
##################################################

# ================================================
# Set up parameters
# ================================================
OUTPUT_DIRECTORY='precomputed_results'
PLOTS_OUTPUT_DIRECTORY="$OUTPUT_DIRECTORY/plots/"
JESTER_REWARDS_OUTPUT_DIRECTORY="$OUTPUT_DIRECTORY/rewards/jester"
JESTER_BENCHMARKS_OUTPUT_DIRECTORY="$OUTPUT_DIRECTORY/benchmarks/jester"
MOVIE_LENS_REWARDS_OUTPUT_DIRECTORY="$OUTPUT_DIRECTORY/rewards/movie-lens"
MOVIE_LENS_BENCHMARKS_OUTPUT_DIRECTORY="$OUTPUT_DIRECTORY/benchmarks/movie-lens"

# ================================================
# Plots
# ================================================
python3 plot_bench.py \
    --movie-lens-rewards $MOVIE_LENS_REWARDS_OUTPUT_DIRECTORY\
    --movie-lens-benchmarks $MOVIE_LENS_BENCHMARKS_OUTPUT_DIRECTORY\
    --jester-rewards $JESTER_REWARDS_OUTPUT_DIRECTORY\
    --jester-benchmarks $JESTER_BENCHMARKS_OUTPUT_DIRECTORY\
    --output $PLOTS_OUTPUT_DIRECTORY
