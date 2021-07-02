from random import betavariate, seed

from analyzer import check_correctness
from proto import Architecture
from permutation import IsolatedPermutation
from proto import Proto, Controller, Comp, DataOwner, ProtoParameters
from seed_random import IsolatedBernoulliArm, IsolatedRandomGenerator
from utils import StandardBanditsAlgorithm, permute_and_max


########################################################################################################################
# Thompson Sampling Utils
########################################################################################################################
def thompson_sampling_function(s_i, n_i, beta_seed, t) -> float:
    """
    Executes the Thompson Sampling application based on work at
    https://perso.crans.org/besson/phd/notebooks/Introduction_aux_algorithmes_de_bandit__comme_UCB1_et_Thompson_Sampling.html#Approche-bay%C3%A9sienne,-Thompson-Sampling
    """
    seed(beta_seed + t)
    value = betavariate(alpha=s_i + 1, beta=n_i - s_i + 1)
    return value

class ThompsonsSamplingParameters:
    def __init__(self, sigma_seed, reward_seed, beta_seed, random_arm_seed):
        self.sigma_seed = sigma_seed
        self.reward_seed = reward_seed
        self.beta_seed = beta_seed
        self.random_arm_seed = random_arm_seed

########################################################################################################################
# Standard Algorithm
########################################################################################################################
class ThompsonSamplingBanditsAlgorithm(StandardBanditsAlgorithm):

    def __init__(self, arms_probs: [float], algo_parameters : ThompsonsSamplingParameters):
        super().__init__(arms_probs, reward_seed=algo_parameters.reward_seed)
        self.rewards_by_arm = {arm: 0 for arm in self.arms}
        self.nb_pulls_by_arm = {arm: 0 for arm in self.arms}

        # seeds
        self.sigma_seed = algo_parameters.sigma_seed
        self.beta_seed = algo_parameters.beta_seed
        self.random_arm_selector = IsolatedRandomGenerator(seed=algo_parameters.random_arm_seed)

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
                    print(f"STD Turn {t} R{i} si  {self.rewards_by_arm[arm]} ni {self.nb_pulls_by_arm[arm]}")

            # compute probability related with each arm in prevision of the probability matching
            arms_probabilities = [
                (
                    arm,
                    thompson_sampling_function(
                        s_i=self.rewards_by_arm[arm],
                        n_i=self.nb_pulls_by_arm[arm],
                        beta_seed=self.beta_seed,
                        t=t,
                    )
                )
                for arm in self.arms
            ]

            # Argmax
            arm, arm_estimation = permute_and_max(arms_probabilities, self.sigma_seed, t, key=lambda c: c[1])
            self.pull_and_update_arm(arm, t)
            t += 1

        # once budget is spent, computes and returns total cumulative rewards
        cumulative_rewards = sum(self.rewards_by_arm.values())
        return cumulative_rewards

    def compute_value_by_arm(self, arm, t) -> float:
        return thompson_sampling_function(
            s_i=self.rewards_by_arm[arm],
            n_i=self.nb_pulls_by_arm[arm],
            beta_seed=self.beta_seed,
            t=t
        )

    def pull_and_update_arm(self, arm: IsolatedBernoulliArm, t: int):
        reward = arm.pull(t)
        self.rewards_by_arm[arm] += reward
        self.nb_pulls_by_arm[arm] += 1


########################################################################################################################
# Specialisation of the algorithm
########################################################################################################################
class ThompsonSamplingRi(DataOwner):
    def __init__(self, arm_prob, K, i, proto_parameters: ProtoParameters, algo_parameters : ThompsonsSamplingParameters):
        super().__init__(arm_prob, K, i, proto_parameters)
        self.beta_seed = algo_parameters.beta_seed

    def compute_value(self, turn: int, iteration: int) -> float:
        return thompson_sampling_function(
            s_i=self.s_i,
            n_i=self.n_i,
            beta_seed=self.beta_seed,
            t=turn
        )

    def handle_select(self, turn: int, iteration: int, b_i: int):
        if b_i == 1:
            self.s_i += self.arm.pull(turn)
            self.n_i += 1


class ThompsonSamplingComp(Comp):

    def __init__(self, K, proto_parameters: ProtoParameters, algo_parameters : ThompsonsSamplingParameters):
        super().__init__(K, proto_parameters)
        self.random_arm_selector = IsolatedRandomGenerator(seed=algo_parameters.random_arm_seed)

    def select_arm(self, turn: int, computation_round: int, values : [float]) -> int:
        # pulling a random arm i weighted with a given probability.
        max_index, max_value = 0, values[0]
        for i in range(1, len(values)):
            if max_value < values[i]:
                max_index, max_value = i, values[i]
        return max_index

class ThompsonSamplingProto(Proto):

    def __init__(self, arms_probs: [float], proto_parameters: ProtoParameters, algo_parameters : ThompsonsSamplingParameters):
        super().__init__(arms_probs, proto_parameters)
        self.algo_parameters = algo_parameters

    def select_architecture(self, turn: int, computation_round: int):
        return Architecture.INFORMED

    def provide_comp(self, *args, **kwargs):
        return ThompsonSamplingComp(*args,**kwargs, algo_parameters=self.algo_parameters)

    def provide_do(self, *args, **kwargs):
        return ThompsonSamplingRi(*args, **kwargs, algo_parameters=self.algo_parameters)

    def provide_controller(self, *args, **kwargs):
        return Controller(*args, **kwargs)

###################################################
# Algorithms generation facility
###################################################
class ThompsonSamplingFacility:
    def __init__(
            self,
            reward_seed,
            sigma_seed,
            beta_seed,
            random_arm_seed,
            alpha_seed,
            sk,
            pk,
            cloud_key,
            cd_key,
            arms_probs
    ):
        self.beta_seed = beta_seed
        self.random_arm_seed = random_arm_seed
        self.arms_probs = arms_probs
        self.sk = sk
        self.alpha_seed = alpha_seed
        self.sigma_seed = sigma_seed
        self.reward_seed = reward_seed
        self.pk = pk
        self.cd_key = cd_key
        self.cloud_key = cloud_key

    def create_standard(self) -> ThompsonSamplingBanditsAlgorithm:
        return ThompsonSamplingBanditsAlgorithm(
            arms_probs=self.arms_probs,
            algo_parameters=self.__create_algo_parameters()
        )

    def create_generic(self, security: bool) -> ThompsonSamplingProto:
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

        return ThompsonSamplingProto(
            arms_probs=self.arms_probs,
            proto_parameters=proto_parameters,
            algo_parameters=self.__create_algo_parameters()
        )

    def __create_algo_parameters(self) -> ThompsonsSamplingParameters:
        return ThompsonsSamplingParameters(
            reward_seed=self.reward_seed,
            sigma_seed=self.sigma_seed,
            beta_seed=self.beta_seed,
            random_arm_seed=self.random_arm_seed
        )