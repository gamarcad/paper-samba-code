from random import uniform, randint
from abc import abstractmethod, ABC
from time import time, perf_counter
import matplotlib.pyplot as plt

from seed_random import IsolatedBernoulliArm
from permutation import IsolatedPermutation

class Timer:
    """Timer class allows to time a arbitrary blocks of code by using "with" python statement.

    This object allows to time several not nested blocks of code as follows:

    timer = Timer()
    with timer:
        sleep(20)
    print("Execution time (s): {}".format(time.execution_time_in_seconds()))

    Warning: Current implementation of does not support nested blocks timing.
    In a such way, the timer will be reset each time the time is reused in a with statement.
    """
    def __init__(self):
        self.__total_execution_time : float = 0
        self.__running : bool = False
        self.__start : float = 0

    def __enter__(self):
        self.__has_start = True
        self.__start = perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = perf_counter()
        self.__total_execution_time += end - self.__start

        self.__running = False
        self.__start = 0

    def execution_time_in_seconds(self) -> float:
        """Returns elapsed time in seconds.
        Returns:
            The elapsed time between the beginning and the end of the "with" block.
        """
        return self.__total_execution_time



def randint_if_none( value ):
    if value is None:
        return randint( 10, 1000 )
    else:
        return value

def parse_bench_log( filename : str ):
    result = {}
    with open(filename) as file:
        lines = "".join(file.readlines()).replace("\n", "").split("#")
        for line in lines:
            if line == "": continue
            node, cpu_usage_time = line.split(":")
            result[node] = float(cpu_usage_time)
    return result

class BernoulliArm:
    """Bandit arm with a bernoulli bandit arm.
    """
    def __init__(self, p):
        """Bernoulli arm initialization.

        This arm returns 1 value with a given probability p and 0 with a probability 1 - p.

        :param p: Probability to obtains an 1
        """
        self.p = p

    def pull(self):
        """Pulled arm in order to obtain a value in bernoulli distribution.

        Returns 1 with a probability p and 0 with a probability 1 - p.
        """
        return int(uniform(0, 1) < self.p)



class BanditsAlgorithm:
    @abstractmethod
    def play( self, budget ) -> int: pass

class DebugBanditsAlgorithm(BanditsAlgorithm):
    @abstractmethod
    def pulled_arm_at_each_turn(self) -> [BernoulliArm] : pass

    @abstractmethod
    def rewards_at_each_turn(self) -> [int]: pass

    @abstractmethod
    def get_probabilities(self) -> [float]: pass

class StandardBanditsAlgorithm(BanditsAlgorithm, ABC):
    def __init__(self, arms_probabilities: [float], reward_seed = 123):
        self.K = len(arms_probabilities)
        self.arms = [ IsolatedBernoulliArm( p, reward_seed ) for p in arms_probabilities ]

    def get_arm_by_index(self, arm_index) -> IsolatedBernoulliArm: return self.arms[arm_index]



def debug_algorithm( budget : int, algorithms : [DebugBanditsAlgorithm] ):

    if not isinstance(algorithms, list):
        algorithms = [algorithms]

    for algorithm in algorithms:
        print("Debugging ", type(algorithm))
        start = time()
        algorithm.play( budget )
        end = time()


        rewards_at_each_turn = algorithm.rewards_at_each_turn()
        arm_at_each_turn = algorithm.pulled_arm_at_each_turn()
        assert len(arm_at_each_turn) == budget, "Pulled arm at each turn has not the same length that budget: {} instead of {}".format(len(arm_at_each_turn), budget)
        assert len(rewards_at_each_turn) == budget, "Rewards at each turn has not the same length that budget: {} instread of {}".format(len(rewards_at_each_turn), budget)

        # Computing regret at each turn.
        # Starting by searching the best arm's probability.
        probs = algorithm.get_probabilities()
        prob_max = max(probs)
        optimal_rewards = int(budget * prob_max)
        total_regret = optimal_rewards - sum(rewards_at_each_turn)

        regret_at_each_turn = []
        for i in range( budget ):
            regret_at_each_turn.append( i * prob_max - sum(rewards_at_each_turn[:(i+1)]) )
        #plt.plot(regret_at_each_turn, label=type(algorithm).__name__)

        best_arm_pulling_percentage_at_each_turn = []
        best_arm_pulling_number = 0
        for i in range(budget):
            pulled_arm_at_turn_i = arm_at_each_turn[i]
            if pulled_arm_at_turn_i.p == prob_max:
                best_arm_pulling_number += 1
            best_arm_pulling_percentage_at_each_turn.append(best_arm_pulling_number / (i + 1))

        plt.plot( best_arm_pulling_percentage_at_each_turn, label=type(algorithm).__name__ )
    plt.legend()
    plt.show()



def permute_and_max( l, perm_seed : int, turn : int, key = lambda x: x ):
    permutation = IsolatedPermutation.new(len(l), perm_seed, turn)
    permuted_l = permutation.permute(l)
    max_index = 0
    max_value = key(permuted_l[0])
    for i in range(1, len(permuted_l)):
        vi = key(permuted_l[i])
        if vi > max_value:
            max_index = i
            max_value = vi
    return permuted_l[max_index]


def read_arms_from_file( filename ):
    with open(filename) as file:
        lines = file.readlines()[1:]
        arms_probs = []
        for line in lines:
            arms_probs.append(float(line))
        return arms_probs