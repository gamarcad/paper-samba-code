# ==========================================================================================
# file: ucb.py
# description: UCB (Upper Confidence Bounds)
# ==========================================================================================
from math import sqrt, log, e
from proto import Architecture
from proto import Controller, Proto, DataOwner, Comp, ProtoParameters
from seed_random import IsolatedBernoulliArm
from utils import StandardBanditsAlgorithm, permute_and_max


########################################################################################################################
# UCB Utils
########################################################################################################################
def ucb_function( turn, s_i, n_i ):
    exploitation_term = s_i / n_i
    exploration_term = sqrt((2 * log(turn, e)) / n_i)
    return exploitation_term + exploration_term


class UCBParameters:
    def __init__(self, sigma_seed, reward_seed):
        self.sigma_seed = sigma_seed
        self.reward_seed = reward_seed

########################################################################################################################
# Standard Algorithm
########################################################################################################################
class UCBBanditsAlgorithm(StandardBanditsAlgorithm):
    """
    Implementation of the standard UCB-1 bandits algorithm.
    It follows the standard defined in the paper "Algorithms for the multi-armed bandit problem"
    accessible at url https://arxiv.org/pdf/1402.6028.pdf
    """
    def __init__(self, arms_probs: [float], algo_parameters : UCBParameters):
        super().__init__(arms_probs, reward_seed=algo_parameters.reward_seed)
        self.sigma_seed = algo_parameters.sigma_seed
        self.rewards_by_arm = { arm: 0 for arm in self.arms }
        self.nb_pulls_by_arm = { arm: 0 for arm in self.arms }


    def play(self, N, debug = False):
        self.debug = debug

        # start by playing each arm one time
        t = 1
        for arm in self.arms:
            self.pull_and_update_arm(arm, t)
            t += 1

        # spending remaining budget
        while t <= N:

            # Print in the standard output s_i and n_i for each arm, useful to ensure the correctness.
            if self.debug:
                for arm in self.arms:
                    i = self.arms.index(arm)
                    print(f"STD Turn {t} R{i} si  {self.rewards_by_arm[arm]} ni {self.nb_pulls_by_arm[arm]} vi {self.compute_value_by_arm(arm, t)}")

            estimations = [
                (
                    arm,
                    self.compute_value_by_arm(arm, t)
                )
                for arm in self.arms
            ]

            arm, arm_estimation = permute_and_max(estimations, self.sigma_seed, t, key=lambda c: c[1])
            self.pull_and_update_arm(arm, t)

            t += 1

        # once budget is spent, computes and returns total cumulative rewards
        cumulative_rewards = sum(self.rewards_by_arm.values())
        return cumulative_rewards

    def pull_and_update_arm(self, arm : IsolatedBernoulliArm, t : int):
        reward = arm.pull( t )
        self.rewards_by_arm[arm] += reward
        self.nb_pulls_by_arm[arm] += 1

    def compute_value_by_arm(self, arm, turn):
        s_i, n_i = self.rewards_by_arm[arm], self.nb_pulls_by_arm[arm]
        return ucb_function( turn, s_i, n_i )

########################################################################################################################
# Specialisation of the algorithm
########################################################################################################################
class UCBDataOwner(DataOwner):
    """
    Implementation of the DataOwner with a specialisation for the UCB algorithm.
    """
    def compute_value(self, turn: int, iteration: int) -> float:
        exploitation_term = self.s_i / self.n_i
        exploration_term = sqrt( 2 * log(turn) / self.n_i )
        return exploration_term + exploitation_term

    def handle_select(self, turn: int, iteration: int, b_i: int):
        if b_i == 1:
            self.s_i += self.arm.pull(turn)
            self.n_i += 1


class UCBComp(Comp):
    def select_arm(self, turn: int, computation_round: int, values: [float]) -> int:
        # performing an argmax on permuted data
        max_index, max_value = 0, values[0]
        for i in range(1, len(values)):
            if max_value < values[i]:
                max_index, max_value = i, values[i]
        return max_index

class UCBProto(Proto):
    """
   Implementation of the Proto, with a specialization for the standard UCB-1 bandits algorithm.
   It follows the standard defined in the paper "Algorithms for the multi-armed bandit problem"
   accessible at url https://arxiv.org/pdf/1402.6028.pdf
   """

    def __init__(self, arms_probs: [float], proto_parameters: ProtoParameters, algo_parameters : UCBParameters):
        super().__init__(arms_probs, proto_parameters)


    def provide_do(self, **kwargs) -> DataOwner:
        return UCBDataOwner(**kwargs)

    def provide_controller(self, **kwargs) -> Controller:
        return Controller(**kwargs)

    def provide_comp(self, **kwargs) -> Comp:
        return UCBComp(**kwargs)

    def select_architecture(self, turn: int, computation_round: int):
        return Architecture.INFORMED

###################################################
# Algorithms generation facility
###################################################
class UCBFacility:
    def __init__(
            self,
            reward_seed,
            sigma_seed,
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
        self.sigma_seed = sigma_seed
        self.reward_seed = reward_seed
        self.pk = pk
        self.cd_key = cd_key
        self.cloud_key = cloud_key

    def create_standard(self) -> UCBBanditsAlgorithm:
        return UCBBanditsAlgorithm(
            arms_probs=self.arms_probs,
            algo_parameters=self.__create_algo_parameters()
        )

    def create_generic(self, security: bool) -> UCBProto:
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

        return UCBProto(
            arms_probs=self.arms_probs,
            proto_parameters=proto_parameters,
            algo_parameters=self.__create_algo_parameters()
        )

    def __create_algo_parameters(self) -> UCBParameters:
        return UCBParameters(
            reward_seed=self.reward_seed,
            sigma_seed=self.sigma_seed,
        )