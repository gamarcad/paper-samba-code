from math import log
from permutation import IsolatedPermutation
from proto import Architecture
from proto import Proto, Controller, DataOwner, Comp, ProtoParameters
from utils import StandardBanditsAlgorithm, IsolatedBernoulliArm, permute_and_max, randint_if_none
from seed_random import IsolatedRandomGenerator

class EGreedyDecreasingParameters:
    """
    Defines parameters related with the e-greedy decreasing bandit algorithm, and some parameters useful to
    control the randomness.
    """
    def __init__(self, epsilon_seed, reward_seed, sigma_seed, random_arm_seed):
        self.epsilon_seed = epsilon_seed
        self.reward_seed = reward_seed
        self.sigma_seed = sigma_seed
        self.random_arm_seed = random_arm_seed


########################################################################################################################
# Standard algorithm
########################################################################################################################
class EpsilonGreedyDecreasingBanditsAlgorithm(StandardBanditsAlgorithm):
    """
    Implementation of the standard e-greedy bandits algorithm, with modifications in order to
    decrease the epsilon.
    """
    def __init__(
            self,
            arms_probs: [float],
            algo_parameters : EGreedyDecreasingParameters,
    ):
        super().__init__(arms_probs, reward_seed=randint_if_none(algo_parameters.reward_seed))
        self.rewards_by_arm = {arm: 0 for arm in self.arms}
        self.nb_pulls_by_arm = {arm: 0 for arm in self.arms}
        self.epsilon_generator = IsolatedRandomGenerator(seed=randint_if_none(algo_parameters.epsilon_seed))
        self.random_arm_generator = IsolatedRandomGenerator(seed=randint_if_none(algo_parameters.random_arm_seed))
        self.sigma_seed = randint_if_none(algo_parameters.sigma_seed)

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

            # probability epsilon: pulling a random arm
            # probability 1-epsilon: pullin the best arm
            epsilon = 1 / log( t, 2 )
            if self.epsilon_generator.random(t) < epsilon:
                # randint returns a random integer between a and b includes
                # we consider a permutation done by the AS
                random_arm_index = self.random_arm_generator.randint(t, 0, self.K - 1)
                permutation = IsolatedPermutation.new(self.K, self.sigma_seed, t)
                selection_bits = permutation.invert_permutation([1 if i == random_arm_index else 0 for i in range(self.K)])
                selected_arm_index = selection_bits.index(1)
                arm = self.get_arm_by_index(selected_arm_index)
                self.pull_and_update_arm(arm, t)
            else:
                l = [(arm, self.rewards_by_arm[arm] / self.nb_pulls_by_arm[arm]) for arm in self.arms]
                arm, arm_estimation = permute_and_max(l, perm_seed=self.sigma_seed, turn=t, key=lambda x: x[1])
                self.pull_and_update_arm(arm, t)

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
class EGreedyDecreasingDataOwner(DataOwner):
    def __init__(self, arm_prob, K, i, proto_parameters: ProtoParameters, algo_parameters : EGreedyDecreasingParameters):
        super().__init__(arm_prob, K, i, proto_parameters)
        self.random_arm_generator = IsolatedRandomGenerator(seed=algo_parameters.random_arm_seed)

    def compute_value(self, turn: int, iteration: int) -> float:
        return self.s_i / self.n_i

    def handle_select(self, turn, iteration, b_i: int):
        if b_i == 1:
            self.s_i += self.arm.pull(turn)
            self.n_i += 1

class EGreedyDecreasingComp(Comp):
    def select_arm(self, turn, round, values) -> int:
        # searching the best max index
        max_index, max_value = 0, values[0]
        for i in range(1, len(values)):
            if max_value < values[i]:
                max_index, max_value = i, values[i]
        return max_index

class EpsilonGreedyDecreasingProto(Proto):
    def __init__(
            self,
            arms_probs: [float],
            proto_parameters: ProtoParameters,
            algo_parameters: EGreedyDecreasingParameters
    ):
        super().__init__(arms_probs, proto_parameters)
        self.algo_parameters = algo_parameters
        self.epsilon_generator = IsolatedRandomGenerator(seed=algo_parameters.epsilon_seed)
        self.random_arm_seed = algo_parameters.random_arm_seed

    def provide_do(self, **kwargs) -> DataOwner:
        return EGreedyDecreasingDataOwner(**kwargs, algo_parameters=self.algo_parameters)

    def provide_controller(self, **kwargs) -> Controller:
        return Controller(**kwargs)

    def provide_comp(self, **kwargs) -> Comp:
        return EGreedyDecreasingComp(**kwargs)

    def select_architecture(self, turn: int, computation_round: int):
        self.epsilon = 1 / log( turn, 2 )
        if self.epsilon_generator.random(turn) <= self.epsilon:
            return Architecture.RANDOM
        else:
            return Architecture.INFORMED


###################################################
# Algorithms generation facility
###################################################
class EGreedyDecreasingFacility:
    def __init__(
            self,
            epsilon_seed,
            reward_seed,
            sigma_seed,
            random_arm_seed,
            alpha_seed,
            sk,
            pk,
            cloud_key,
            cd_key,
            arms_probs
    ):
        self.arms_probs = arms_probs
        self.sk = sk
        self.alpha_seed = alpha_seed
        self.random_arm_seed = random_arm_seed
        self.sigma_seed = sigma_seed
        self.reward_seed = reward_seed
        self.epsilon_seed = epsilon_seed
        self.pk = pk
        self.cd_key = cd_key
        self.cloud_key = cloud_key

    def create_standard(self) -> EpsilonGreedyDecreasingBanditsAlgorithm:
        return EpsilonGreedyDecreasingBanditsAlgorithm(
            arms_probs=self.arms_probs,
            algo_parameters=self.__create_algo_parameters()
        )

    def create_generic(self, security: bool) -> EpsilonGreedyDecreasingProto:
        proto_parameters = ProtoParameters.new_from_keys(
            cloud_key=self.cloud_key,
            cd_key=self.cd_key,
            pk=self.pk,
            sk=self.sk,
            alpha_seed=self.alpha_seed,
            reward_seed=self.reward_seed,
            sigma_seed=self.sigma_seed,
            random_arm_seed=self.random_arm_seed,
        )
        proto_parameters.security = security

        return EpsilonGreedyDecreasingProto(
            arms_probs=self.arms_probs,
            proto_parameters=proto_parameters,
            algo_parameters=self.__create_algo_parameters()
        )

    def __create_algo_parameters(self) -> EGreedyDecreasingParameters:
        return EGreedyDecreasingParameters(
            epsilon_seed=self.epsilon_seed,
            reward_seed=self.reward_seed,
            sigma_seed=self.sigma_seed,
            random_arm_seed=self.random_arm_seed,
        )
