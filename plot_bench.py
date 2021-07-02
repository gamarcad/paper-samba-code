# =======================================================================
# file: plot_bench.py
# description: Generate plots.
# =======================================================================
import json
import os
import re
from argparse import ArgumentParser
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt


# ======================================================================
# Utils functions
# ======================================================================
def read_bench(filename: str):
    with open(filename, 'r') as file:
        return json.loads(file.read())


def estimate_optimal_rewards(arms, N) -> float:
    """Returns the estimated obtained rewards with an optimal strategy"""
    best_arm = max(arms)
    return best_arm * N


def estimate_random_rewards(arms, N):
    """Returns the estimated obtained rewards with a random strategy"""
    return N * (sum(arms) / len(arms))


# ======================================================================
# Enumerations
# ======================================================================
class DataKey:
    """Keys used in data dictionnary."""
    TIME_BY_COMPONENTS = "time_by_components"
    ARMS = 'arms'
    TIME = 'time'
    REWARDS = 'reward'
    ARGS = 'args'


class Algorithm(Enum):
    """Enumeration of algorithms used in plots"""
    E_GREEDY = "e-greedy"
    E_GREEDY_DECREASING = "e-greedy-decreasing"
    SOFTMAX = "boltzmann"
    UCB = "ucb"
    THOMPSON = "thompson-sampling"
    PURSUIT = "pursuit"


class AlgorithmVersion(Enum):
    """Enumeration of versions of algorithms."""
    STD = 'std'
    GEN = 'gen-distributed'
    SEC = 'gen-secure'


class Mean:
    """Utility object that render the mean computation more clear."""

    def __init__(self):
        self.values = []

    def append(self, v): self.values.append(v)

    def mean(self): return sum(self.values) / len(self.values)


# =======================================================================
# Encapsulating Objects
# =======================================================================
class IterationBenchmark:
    """
    class IterationBenchmark

    This class handles a single iteration, defined with a given budget N
    and number of arms K.
    During an iteration, each algorithm is executed.
    """

    @staticmethod
    def create_from_file(filename):
        """Creates an IterationBenchmark from a given filename.
        """
        # extracting data from the filename
        tokens = re.search("N_([0-9]+)_K_([0-9]+)_it_([0-9]+)", filename).group(1, 2, 3)
        N, K, it = tuple(map(int, tokens))

        # reading data containing in file
        data = read_bench(filename)
        arms = data[DataKey.ARMS]

        return IterationBenchmark(N=N, K=K, arms=arms, iteration=it, data=data)

    def __init__(self, N, K, iteration, arms, data):
        self.N = N
        self.K = K
        self.iteration = iteration
        self.arms = arms
        self.iteration_key = f'it-{iteration}'
        self.data = data

    def time(self, algo: Algorithm, version: AlgorithmVersion) -> float:
        """Returns execution time of given algorithm on the specified version."""
        return self.data[algo.value][self.iteration_key][version.value][DataKey.TIME]

    def rewards(self, algo: Algorithm, version: AlgorithmVersion):
        """Returns the cumulative rewards obtained by given algorithm for specified version."""
        return self.data[algo.value][self.iteration_key][version.value][DataKey.REWARDS]

    def time_by_components(self, algo: Algorithm, version: AlgorithmVersion):
        """Returns the time of each component used in Samba"""
        return self.data[algo.value][self.iteration_key][version.value][DataKey.TIME_BY_COMPONENTS]

    def __str__(self):
        return f"<IterationBenchmark: N = {self.N}, K = {self.K}, iteration = {self.iteration}>"


class Benchmarks:
    """Benchmarks object offers some high level features over all iteration benchmark."""

    @staticmethod
    def create_benchmarks_from_directory(directory_path, benchmark_filter=None):
        """Returns a Benchmarks object initialized with iterations benchmarks located in given directory.

        Args:
            directory_path: Directory where IterationBenchmark are located.
            benchmark_filter: Lambda functions that returns True if the IterationBenchmark object must be inserted.
        """
        # reads benchmarks from directory
        benchmarks = {}
        for root, dirs, filenames in os.walk(directory_path):
            for filename in filenames:
                filename_path = os.path.join(root, filename)
                iteration_benchmark = IterationBenchmark.create_from_file(filename_path)
                if benchmark_filter is None or benchmark_filter(iteration_benchmark):
                    key = (iteration_benchmark.N, iteration_benchmark.K)
                    if key in benchmarks:
                        benchmarks[key].append(iteration_benchmark)
                    else:
                        benchmarks[key] = [iteration_benchmark]

        return Benchmarks(benchmarks)

    def __init__(self, benchmarks_by_N_K: {(int, int): [IterationBenchmark]}):
        self.benchmarks_by_N_K: {(int, int): [IterationBenchmark]} = benchmarks_by_N_K

    def mean_time(self, N, K, algo: Algorithm, version: AlgorithmVersion) -> float:
        """
        Returns the mean execution time for an algorithm in a specific version for budget N and the number of arms K.
        """
        mean_time = Mean()
        key = self.__create_key(N, K)
        for iteration_benchmark in self.benchmarks_by_N_K[key]:
            assert type(iteration_benchmark) == IterationBenchmark, \
                'Benchmarks contains object that is not IterationBenchmark'
            mean_time.append(iteration_benchmark.time(algo, version))
        return mean_time.mean()

    def mean_reward(self, N, K, algo: Algorithm, version: AlgorithmVersion) -> float:
        """
        Returns the mean of rewards for an algorithm in a specific version for budget N and the number of arms K.
        """
        mean_time = Mean()
        key = self.__create_key(N, K)
        for iteration_benchmark in self.benchmarks_by_N_K[key]:
            assert type(iteration_benchmark) == IterationBenchmark, \
                'Benchmarks contains object that is not IterationBenchmark'
            mean_time.append(iteration_benchmark.rewards(algo, version))
        return mean_time.mean()

    def max_reward(self, N, K, algo: Algorithm, version: AlgorithmVersion) -> int:
        """
        Returns the highest rewards for an algorithm in a specific version for budget N and the number of arms K.
        """
        max_reward = 0
        key = self.__create_key(N, K)
        for iteration_benchmark in self.benchmarks_by_N_K[key]:
            assert type(iteration_benchmark) == IterationBenchmark, \
                'Benchmarks contains object that is not IterationBenchmark'
            max_reward = max(max_reward, iteration_benchmark.rewards(algo, version))
        return max_reward

    def max_time(self, N, K, algo: Algorithm, version: AlgorithmVersion) -> float:
        """
        Returns the highest execution time for an algorithm in a specific version for budget N and the number of arms K.
        """
        max_time = 0
        key = self.__create_key(N, K)
        for iteration_benchmark in self.benchmarks_by_N_K[key]:
            assert type(iteration_benchmark) == IterationBenchmark, \
                'Benchmarks contains object that is not IterationBenchmark'
            max_time = max(max_time, iteration_benchmark.time(algo, version))
        return max_time

    def mean_rewards_scalability_N(self, K, algo: Algorithm, version: AlgorithmVersion, N_filter=None):
        """
        Returns the list of scalability with N for a specific version of an algorithm.

        Returns:
            The returned list length is equals to the number of budget defined in the Benchmarks.
        """
        return [
            self.mean_reward(N, K, algo, version)
            for N in self.sorted_N()
            if N_filter is None or N_filter(N)
        ]

    def mean_time_by_components(self, N, K, algo: Algorithm, version: AlgorithmVersion) -> {str: float}:
        """
        Returns the mean time of each components.

        Returns:
            The keys of the returned dictionary are the component and values are the mean of execution time for each
            algorithm.
        """
        time_by_components = {}
        key = self.__create_key(N, K)
        for benchmark in self.benchmarks_by_N_K[key]:
            for component, time in benchmark.time_by_components(algo, version).items():
                time_by_components[component] = time_by_components.get(component, 0) + time

        return {
            component: cumulative_time / sum(time_by_components.values())
            for component, cumulative_time in time_by_components.items()
        }

    def mean_times_scalability_N(self, K, algo: Algorithm, version: AlgorithmVersion, N_filter=None) -> [float]:
        """
        Returns the list containing the mean of execution time for a specific version of an algorithm.
        The size of the returned list is equals to the number of budget used in benchmarks.
        """
        return [
            self.mean_time(N, K, algo, version)
            for N in self.sorted_N()
            if N_filter is None or N_filter(N)
        ]

    def mean_times_scalability_K(self, N, algo: Algorithm, version: AlgorithmVersion, K_filter=None) -> [float]:
        """
        Returns the list containing the mean of execution time for a specific version of an algorithm.
        The size of the returned list is equals to the number of budget used in benchmarks.
        """
        return [
            self.mean_time(N, K, algo, version)
            for K in self.sorted_K()
            if K_filter is None or K_filter(K)
        ]

    def sorted_N(self) -> [int]:
        """Returns the sorted list of budgets N used in the benchmarks."""
        return sorted(list(set([
            N for (N, K) in self.benchmarks_by_N_K.keys()
        ])))

    def sorted_K(self) -> [int]:
        """Returns the sorted list of number of arms used in the benchmarks."""
        return sorted(list(set([
            K for (N, K) in self.benchmarks_by_N_K.keys()
        ])))

    def __create_key(self, N, K):
        return (N, K)

    def __str__(self):
        return "<Benchmarks: iterations = {}>".format(list(self.benchmarks_by_N_K.keys()))


# ======================================================================
# Plotting variables and utils
# ======================================================================

NAME = {
    Algorithm.E_GREEDY: '$\\varepsilon$-greedy',
    Algorithm.E_GREEDY_DECREASING: '$\\varepsilon$-decreasing-greedy',
    Algorithm.SOFTMAX: 'Softmax',
    Algorithm.UCB: 'UCB',
    Algorithm.THOMPSON: 'Thompson Sampling',
    Algorithm.PURSUIT: 'Pursuit',
}

COLORS = {
    Algorithm.E_GREEDY: 'tab:green',
    Algorithm.E_GREEDY_DECREASING: 'tab:brown',
    Algorithm.SOFTMAX: 'tab:red',
    Algorithm.UCB: 'tab:orange',
    Algorithm.THOMPSON: 'tab:blue',
    Algorithm.PURSUIT: 'tab:purple',
}

MARKERS = {
    Algorithm.E_GREEDY: 'x',
    Algorithm.E_GREEDY_DECREASING: 'o',
    Algorithm.SOFTMAX: '<',
    Algorithm.UCB: 'p',
    Algorithm.THOMPSON: 'v',
    Algorithm.PURSUIT: 's',
}


class Counter:
    def __init__(self, start=1):
        self.n = start

    def next(self) -> int:
        result = self.n
        self.n += 1
        return result


CONTROLLER = 'Controller'


def format_components_name(component_name: str) -> str:
    if re.match("R[0-9]", component_name):
        return "DO$_{" + str(int(component_name[1]) + 1) + "}$"
    if component_name == 'as': return CONTROLLER
    if component_name == 'dc': return ""
    if component_name == 'comp': return 'Comp'
    return component_name


# ===============================================================
# Plotting functions
# ===============================================================
def format_ticks(ax, x=True, y=True):
    """Format ticks."""

    def format_func(x: float, pos):
        if x / 1000 % 1 == 0:
            return str(int(x // 1000))
        else:
            return str(x / 1000)

    formatter = plt.FuncFormatter(format_func)
    if x:
        ax.xaxis.set_major_formatter(formatter)
    if y:
        ax.yaxis.set_major_formatter(formatter)


def ticks_fontsize(fontsize):
    """Modify the ticks fontsize of the current plot."""
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)


def plot_figure(ml_benchmarks_path, ml_rewards_path, jester_benchmarks_path, jester_rewards_path, output_location):
    """Plots figures needed in the paper."""

    MOVIE_LENS = 'MovieLens'
    JESTER = 'Jester'
    # General plots configuration
    max_ylim_increase_ratio = 1.1
    TICKS_FONTSIZE = 40
    label_fontsize = 40
    title_fontsize = 40

    # -------------------------------------------------
    # plotting rewards with respect to N
    # -------------------------------------------------
    plt.figure(figsize=(20, 11.5))
    N_ROWS, N_COLS = 1, 2
    plt.tight_layout()
    legend_ncol = len(Algorithm) // 2
    legend_size = 35
    legend_anchor = (2.2, 1.37)
    plt.subplots_adjust(top=.77, bottom=0.13)
    subfigure_position_counter = Counter()

    ml_rewards = Benchmarks.create_benchmarks_from_directory(ml_rewards_path)
    jester_rewards = Benchmarks.create_benchmarks_from_directory(jester_rewards_path)
    sorted_algorithms = sorted(
        Algorithm,
        reverse=True,
        key=lambda algo: ml_rewards.mean_reward(N=100000, K=100, algo=algo, version=AlgorithmVersion.STD)
    )

    # search the maximum reward obtained over all algorithms for movie-lens with highest N and K
    N, K = 100000, 100
    max_reward = max([
        ml_rewards.max_reward(N, K, algo, AlgorithmVersion.STD)
        for algo in Algorithm
    ]) * max_ylim_increase_ratio

    K = 100
    for benchmark_index, benchmark in enumerate([ml_rewards, jester_rewards]):
        ax = plt.subplot(N_ROWS, N_COLS, subfigure_position_counter.next())
        ticks_fontsize(fontsize=TICKS_FONTSIZE)
        plt.subplots_adjust(bottom=0.15)
        format_ticks(ax)

        for algo in sorted_algorithms:
            plt.plot(
                benchmark.sorted_N(),
                benchmark.mean_rewards_scalability_N(K, algo, AlgorithmVersion.STD),
                linestyle='--',
                marker=MARKERS[algo],
                label=NAME[algo],
                color=COLORS[algo],
                fillstyle="none"
            )

        if benchmark_index == 0:
            plt.ylabel("Cumulative Reward (x$10^3$)", fontsize=label_fontsize)
            plt.legend(bbox_to_anchor=legend_anchor, fontsize=legend_size, ncol=legend_ncol)
        benchmark_name = MOVIE_LENS if benchmark == ml_rewards else JESTER
        plt.title(benchmark_name, fontsize=title_fontsize)
        plt.xlabel(f"Budget N (x$10^3$)", fontsize=label_fontsize)
        plt.ylim(bottom=15000, top=min(55000, max_reward))
        plt.grid()

    # exporting rewards figure
    plt.savefig(os.path.join(output_location, "rewards.pdf"))
    plt.clf()

    # ---------------------------------------------------
    # benchmarks scalability with respect to the budget N
    # ---------------------------------------------------
    plt.figure(figsize=(20, 11.5))
    N_ROWS, N_COLS = 1, 2
    plt.tight_layout()
    legend_ncol = len(Algorithm) // 2
    legend_size = 35
    legend_anchor = (2.4, 1.37)
    plt.subplots_adjust(top=.77, bottom=0.13)
    subfigure_position_counter = Counter()

    ml_benchmarks = Benchmarks.create_benchmarks_from_directory(ml_benchmarks_path)
    jester_benchmarks = Benchmarks.create_benchmarks_from_directory(jester_benchmarks_path)
    sorted_algorithms = sorted(
        Algorithm,
        reverse=True,
        key=lambda algo: ml_benchmarks.mean_time(N=100000, K=100, algo=algo, version=AlgorithmVersion.SEC)
    )

    # searching the highest execution time in order to adjust plots
    N, K = 100000, 100
    max_time = max([
        ml_benchmarks.max_time(N, K, algo, AlgorithmVersion.SEC)
        for algo in Algorithm
    ]) * max_ylim_increase_ratio

    # plotting rewards curves with a scalability with K
    for benchmark_index, benchmark in enumerate([ml_benchmarks, jester_benchmarks]):
        ax = plt.subplot(N_ROWS, N_COLS, subfigure_position_counter.next())
        ticks_fontsize(fontsize=TICKS_FONTSIZE)
        format_ticks(ax)

        for algo in sorted_algorithms:
            plt.plot(
                benchmark.sorted_N(),
                benchmark.mean_times_scalability_N(K, algo, AlgorithmVersion.STD),
                linestyle='--',
                marker=MARKERS[algo],
                color=COLORS[algo],
                fillstyle="none"
            )

            plt.plot(
                benchmark.sorted_N(),
                benchmark.mean_times_scalability_N(K, algo, AlgorithmVersion.SEC),
                linestyle='-',
                marker=MARKERS[algo],
                label=NAME[algo],
                color=COLORS[algo],
                fillstyle="none"
            )

        if benchmark_index == 0:
            plt.ylabel("Time (x$10^3$ seconds)", fontsize=label_fontsize)
            plt.legend(bbox_to_anchor=legend_anchor, fontsize=legend_size, ncol=legend_ncol)

        benchmark_name = MOVIE_LENS if benchmark == ml_benchmarks else JESTER
        plt.title(benchmark_name, fontsize=title_fontsize)
        plt.xlabel(f"Budget N (x$10^3$)", fontsize=label_fontsize)
        plt.ylim(bottom=0, top=max_time)
        plt.grid()

    # exporting  figure
    plt.savefig(os.path.join(output_location, "scalability_N.pdf"))
    plt.clf()

    # ---------------------------------------------------
    # benchmarks scalability with respect to the budget K
    # ---------------------------------------------------
    plt.figure(figsize=(20, 10))
    N_ROWS, N_COLS = 1, 2
    plt.tight_layout()
    subfigure_position_counter = Counter()

    # plotting rewards curves with a scalability with K
    N = 100000
    for benchmark_index, benchmark in enumerate([ml_benchmarks, jester_benchmarks]):
        ax = plt.subplot(N_ROWS, N_COLS, subfigure_position_counter.next())
        ticks_fontsize(fontsize=TICKS_FONTSIZE)
        format_ticks(ax, x=False)
        plt.subplots_adjust(bottom=0.15)
        sorted_K = benchmark.sorted_K()

        for algo in sorted_algorithms:
            plt.plot(
                sorted_K,
                benchmark.mean_times_scalability_K(N, algo, AlgorithmVersion.STD),
                linestyle='--',
                marker=MARKERS[algo],
                color=COLORS[algo],
                fillstyle="none"
            )

            plt.plot(
                sorted_K,
                benchmark.mean_times_scalability_K(N, algo, AlgorithmVersion.SEC),
                linestyle='-',
                marker=MARKERS[algo],
                label=NAME[algo],
                color=COLORS[algo],
                fillstyle="none"
            )

        if benchmark_index == 0:
            plt.ylabel("Time (x$10^3$ seconds)", fontsize=label_fontsize)

        benchmark_name = MOVIE_LENS if benchmark == ml_benchmarks else JESTER
        plt.title(benchmark_name, fontsize=title_fontsize)
        plt.xlabel(f"Number of Arms K", fontsize=label_fontsize)
        plt.ylim(bottom=0, top=max_time)
        plt.grid()

    # exporting  figure
    plt.savefig(os.path.join(output_location, "scalability_K.pdf"))
    plt.clf()

    # -----------------------------------------
    # plotting execution time by component time
    # -----------------------------------------
    colorset = 'terrain'
    subfigure_position_counter = Counter()
    N = 100000
    for subfigure_index, K in enumerate([10, 100]):
        time_by_components = ml_benchmarks.mean_time_by_components(N, K, Algorithm.PURSUIT, AlgorithmVersion.SEC)
        components, times = time_by_components.keys(), time_by_components.values()
        if subfigure_index == 0:
            components = [format_components_name(component) for component in components]
        else:
            components = [format_components_name(component) if component == 'comp' else '' for component in components]

        # defines colors
        plt.subplot(N_ROWS, N_COLS, subfigure_position_counter.next())
        cm = plt.get_cmap(colorset)
        NUM_COLORS = len(time_by_components)
        colors = [
            cm(1. * i / NUM_COLORS)
            for i in range(NUM_COLORS)
        ]
        pink_rgb = (.95, .7, 1, 1)
        if subfigure_index == 0:
            # colors[components.index(CONTROLLER)] = (.5, .5, .5, 1)
            colors[components.index('Comp')] = pink_rgb
        else:
            colors[components.index('Comp')] = pink_rgb

        # plotting
        plt.pie(times, labels=components, colors=colors, textprops={'fontsize': label_fontsize})

    plt.savefig(os.path.join(output_location, "zoom.pdf"), bbox_inches='tight')


# =======================================================================
# Main
# =======================================================================
def create_parser():
    parser = ArgumentParser()
    parser.add_argument("--movie-lens-rewards", default="results/movie-lens/rewards-exploded")
    parser.add_argument("--movie-lens-benchmarks", default="results/movie-lens/benchmarks-exploded-2")
    parser.add_argument("--jester-rewards", default="results/jester/rewards-exploded")
    parser.add_argument("--jester-benchmarks", default="results/jester/benchmarks-exploded-2")
    parser.add_argument('--output', default="output/pdf/")
    return parser


if __name__ == '__main__':

    parser = create_parser()
    args = parser.parse_args()

    # creates the output folder if not exists
    output_path = Path(args.output)
    output_path.mkdir(parents = True, exist_ok=True)

    # benchmarks execution time
    ml_benchmarks = Benchmarks.create_benchmarks_from_directory(args.movie_lens_benchmarks)

    N = 100000
    for algo in Algorithm:
        plt.plot(
            ml_benchmarks.sorted_K(),
            ml_benchmarks.mean_times_scalability_K(N, algo, AlgorithmVersion.SEC),
            linestyle='--',
            marker=MARKERS[algo],
            label=NAME[algo],
            color=COLORS[algo],
            fillstyle="none"
        )
        plt.legend()
    plt.xlabel("Nb Arms K")
    plt.ylabel("Time (s)")
    plt.clf()

    plot_figure(
        ml_benchmarks_path=args.movie_lens_benchmarks,
        ml_rewards_path=args.movie_lens_rewards,
        jester_benchmarks_path=args.jester_benchmarks,
        jester_rewards_path=args.jester_rewards,
        output_location=args.output
    )
