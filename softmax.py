# ==========================================================================================
# file: softmax.py
# description: Softmax
# ==========================================================================================
from math import exp

from permutation import IsolatedPermutation
from proto import Architecture
from proto import DataOwner, Controller, Comp, Proto, ProtoParameters
from seed_random import IsolatedRandomGenerator
from utils import StandardBanditsAlgorithm, IsolatedBernoulliArm


########################################################################################################################
# Boltzmann utils
########################################################################################################################
def softmax_function(s_i: int, n_i: int, tau: float) -> float:
    """Softmax score computation function.

    Returns:
        e^{\frac{\frac{s_i}{n_i}}{\tau}
    """
    assert n_i != 0, 'Number of pulls cannot be equals to 0'
    assert tau != 0, 'tau cannot be equals to 0'
    return exp((s_i / n_i) / tau)


def divide_each_by_sum(values: [float]) -> [float]:
    """Divide each item of the given list by the sum of the list."""
    s = sum(values)
    if s == 0: s = 1
    return [v / s for v in values]


class SoftmaxParameters:
    """
    Softmax parameters.
    """

    def __init__(self, reward_seed, sigma_seed, tau, random_arm_seed):
        self.reward_seed = reward_seed
        self.sigma_seed = sigma_seed
        self.tau = tau
        self.random_arm_choice = random_arm_seed


########################################################################################################################
# Standard Algorithm
########################################################################################################################
class SoftmaxBanditsAlgorithm(StandardBanditsAlgorithm):
    """Softmax standard bandits algorithm.
    It follows the standard defined in the paper "Algorithms for the multi-armed bandit problem"
    accessible at url https://arxiv.org/pdf/1402.6028.pdf
    """

    def __init__(self, arms_probs: [float], algo_parameters: SoftmaxParameters):
        super().__init__(arms_probs, reward_seed=algo_parameters.reward_seed)
        self.rewards_by_arm = {arm: 0 for arm in self.arms}
        self.nb_pulls_by_arm = {arm: 0 for arm in self.arms}
        self.sigma_seed = algo_parameters.sigma_seed
        self.tau = algo_parameters.tau

        # debug properties
        self.played_arms_memory = []
        self.rewards_memory = []
        self.random_arm_selector = IsolatedRandomGenerator(seed=algo_parameters.random_arm_choice)

    def play(self, N, debug=False):
        # start by playing each arm one time
        t = 1
        for arm in self.arms:
            self.pull_and_update_arm(arm, t)
            t += 1

        # spending remaining budget
        while t <= N:

            # Print in the standard output s_i and n_i for each arm, useful to ensure the correctness.
            if debug:
                for arm in self.arms:
                    i = self.arms.index(arm)
                    print(
                        f"STD Turn {t} R{i} si  {self.rewards_by_arm[arm]} ni {self.nb_pulls_by_arm[arm]}"
                    )

            arms_probabilities = [
                softmax_function(
                    s_i=self.rewards_by_arm[arm],
                    n_i=self.nb_pulls_by_arm[arm],
                    tau=self.tau
                )
                for arm in self.arms
            ]
            arms_probabilities = divide_each_by_sum(arms_probabilities)

            # pulling a random arm depending on associated probs
            permutation = IsolatedPermutation.new(nb_items=self.K, perm_seed=self.sigma_seed, turn=t)
            permuted_probs = permutation.permute(arms_probabilities)
            permuted_pulled_arm_index = self.random_arm_selector.choice(range(self.K), permuted_probs, t)
            pulled_arm_index = permutation.invert_permuted_index(permuted_pulled_arm_index)
            pulled_arm = self.arms[pulled_arm_index]
            self.pull_and_update_arm(pulled_arm, t)
            t += 1

        # once budget is spent, computes and returns total cumulative rewards
        cumulative_rewards = sum(self.rewards_by_arm.values())
        return cumulative_rewards

    def pull_and_update_arm(self, arm: IsolatedBernoulliArm, t):
        reward = arm.pull(t)
        self.rewards_by_arm[arm] += reward
        self.nb_pulls_by_arm[arm] += 1

    def reward_by_arm_index(self, arm_index) -> int:
        return self.rewards_by_arm[self.arms[arm_index]]

    def reward_by_arm(self, arm) -> int:
        return self.rewards_by_arm[arm]


########################################################################################################################
# Specialisation of the generic protocol
########################################################################################################################
class SoftmaxDataOwner(DataOwner):
    """
    Implementation of the DataOwner in the protocol, specialized to the Softmax bandits algorithm.
    In this context, the DataOwner must be able to execute the softmax function.
    """

    def __init__(self, arm_prob, K, i, proto_parameters: ProtoParameters, algo_parameters: SoftmaxParameters):
        super().__init__(arm_prob, K, i, proto_parameters)
        self.tau = algo_parameters.tau

    def compute_value(self, turn: int, iteration: int) -> float:
        return softmax_function(self.s_i, self.n_i, self.tau)

    def handle_select(self, turn: int, iteration: int, b_i: int):
        if b_i == 1:
            self.s_i += self.arm.pull(turn)
            self.n_i += 1


class SoftmaxComp(Comp):
    """
    Implementation of Comp of our protocol, with a specialization for the Softmax protocol.
    This implementation must be able to perform a probability matching, following the Softmax approach.
    """

    def __init__(self, K, proto_parameters: ProtoParameters, algo_parameters: SoftmaxParameters):
        super().__init__(K, proto_parameters)
        self.random_arm_choice = IsolatedRandomGenerator(seed=algo_parameters.random_arm_choice)

    def select_arm(self, turn: int, computation_round: int, arms_probabilities: [float]) -> int:
        arms_probabilities = divide_each_by_sum(arms_probabilities)
        return self.random_arm_choice.choice(
            items=range(self.K),
            weights=arms_probabilities,
            t=turn
        )


class SoftmaxProto(Proto):
    def __init__(self, arms_probs: [float], proto_parameters: ProtoParameters, algo_parameters: SoftmaxParameters):
        super().__init__(arms_probs, proto_parameters)
        self.algo_parameters = algo_parameters

    def provide_do(self, **kwargs) -> DataOwner:
        return SoftmaxDataOwner(**kwargs, algo_parameters=self.algo_parameters)

    def provide_controller(self, **kwargs) -> Controller:
        return Controller(**kwargs)

    def provide_comp(self, **kwargs) -> Comp:
        return SoftmaxComp(**kwargs, algo_parameters=self.algo_parameters)

    def select_architecture(self, turn: int, computation_round: int):
        return Architecture.INFORMED


###################################################
# Algorithms generation facility
###################################################
class SoftmaxFacility:
    def __init__(
            self,
            reward_seed,
            sigma_seed,
            random_arm_seed,
            tau,
            alpha_seed,
            sk,
            pk,
            cloud_key,
            cd_key,
            arms_probs
    ):
        self.tau = tau
        self.random_arm_seed = random_arm_seed
        self.arms_probs = arms_probs
        self.sk = sk
        self.alpha_seed = alpha_seed
        self.sigma_seed = sigma_seed
        self.reward_seed = reward_seed
        self.pk = pk
        self.cd_key = cd_key
        self.cloud_key = cloud_key

    def create_standard(self) -> SoftmaxBanditsAlgorithm:
        return SoftmaxBanditsAlgorithm(
            arms_probs=self.arms_probs,
            algo_parameters=self.__create_algo_parameters()
        )

    def create_generic(self, security: bool) -> SoftmaxProto:
        proto_parameters = ProtoParameters.new_from_keys(
            cloud_key=self.cloud_key,
            cd_key=self.cd_key,
            pk=self.pk,
            sk=self.sk,
            alpha_seed=self.alpha_seed,
            reward_seed=self.reward_seed,
            sigma_seed=self.sigma_seed,
        )
        proto_parameters.security = security

        return SoftmaxProto(
            arms_probs=self.arms_probs,
            proto_parameters=proto_parameters,
            algo_parameters=self.__create_algo_parameters()
        )

    def __create_algo_parameters(self) -> SoftmaxParameters:
        return SoftmaxParameters(
            reward_seed=self.reward_seed,
            sigma_seed=self.sigma_seed,
            random_arm_seed=self.random_arm_seed,
            tau=self.tau,
        )
