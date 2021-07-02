from analyzer import check_correctness
from proto import Architecture
from permutation import IsolatedPermutation
from proto import DataOwner, Controller, Comp, Proto, ProtoParameters
from seed_random import IsolatedRandomGenerator
from utils import StandardBanditsAlgorithm, IsolatedBernoulliArm, permute_and_max, read_arms_from_file


########################################################################################################################
# Pursuit Utils
########################################################################################################################
class PursuitParameters:
    def __init__(self, beta, reward_seed, sigma_seed, random_arm_seed):
        self.beta = beta
        self.reward_seed = reward_seed
        self.sigma_seed = sigma_seed
        self.random_arm_seeed = random_arm_seed

########################################################################################################################
# Standard Pursuit
########################################################################################################################
class PursuitBanditsAlgorithm(StandardBanditsAlgorithm):
    """
    Implementation of the standard Pursuit algorithms.
    It follows the standard defined in the paper "Algorithms for the multi-armed bandit problem"
    accessible at url https://arxiv.org/pdf/1402.6028.pdf
    """
    def __init__(self, arms_probs: [float], algo_parameters : PursuitParameters):
        super().__init__(arms_probs, reward_seed=algo_parameters.reward_seed)
        self.rewards_by_arm = { arm: 0 for arm in self.arms }
        self.nb_pulls_by_arm = { arm: 0 for arm in self.arms }
        self.probabilities = { arm: 1 / self.K for arm in self.arms }
        self.beta = algo_parameters.beta
        self.sigma_seed = algo_parameters.sigma_seed
        self.random_arm_selector = IsolatedRandomGenerator( seed = algo_parameters.random_arm_seeed )


    def play(self, N, debug = False):
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

            # Pursuit is divided into two parts:

            # 1. The first part update the probability for each arm to pulled.
            # The probability update works as follows, we denote by pi the probability to be pulled for a given arm i:
            #   pi = pi + beta * (1 - pi), if the arm has the best estimation si/ni
            #   pi = pi + beta * (0 - pi), otherwise.
            estimations = [
                (
                    arm,
                    self.rewards_by_arm[arm] / self.nb_pulls_by_arm[arm]
                )
                for arm in self.arms
            ]

            best_arm, _ = permute_and_max(estimations, perm_seed=self.sigma_seed, turn=t, key = lambda x: x[1])

            for arm in self.arms:
                if arm == best_arm:
                    self.probabilities[arm] = self.probabilities[arm] + self.beta * ( 1 - self.probabilities[arm] )
                else:
                    self.probabilities[arm] = self.probabilities[arm] + self.beta * ( 0 - self.probabilities[arm] )

            # 2. The second part pulls an arm, chosen randomly regarding on a probability matching.
            permutation = IsolatedPermutation.new(nb_items=self.K, perm_seed=self.sigma_seed, turn=t)
            probs = [ self.probabilities[arm] for arm in self.arms ]
            permuted_probs = permutation.permute(probs)
            permuted_pulled_arm_index = self.random_arm_selector.choice( range(self.K), permuted_probs, t)
            pulled_arm_index = permutation.invert_permuted_index(permuted_pulled_arm_index)
            pulled_arm = self.arms[pulled_arm_index]
            self.pull_and_update_arm(pulled_arm, t)

            t += 1

        # once budget is spent, computes and returns total cumulative rewards
        cumulative_rewards = sum(self.rewards_by_arm.values())
        return cumulative_rewards

    def pull_and_update_arm(self, arm : IsolatedBernoulliArm, t : int):
        reward = arm.pull( t )
        self.rewards_by_arm[arm] += reward
        self.nb_pulls_by_arm[arm] += 1


########################################################################################################################
# Specialisation of the protocol
########################################################################################################################
class PursuitRi(DataOwner):
    """
    Implementation of the DataOwner, for the pursuit bandits algorithm.
    """
    def __init__(self, arm_prob, K, i, proto_parameters: ProtoParameters, algo_parameters : PursuitParameters):
        super().__init__(arm_prob, K, i, proto_parameters)
        self.beta = algo_parameters.beta
        self.p_i = -1

    def compute_value(self, turn : int, iteration : int) -> float:
        if self.p_i == -1:
            self.p_i = 1 / self.K

        if iteration == 1:
            return self.s_i / self.n_i
        else:
            return self.p_i

    def handle_select(self, turn, iteration : int, b_i: int):
        # the initial stage where each arm is pulled once must be a standard pulling
        if turn <= self.K and b_i == 1:
            reward = self.arm.pull(turn)
            #print(f"Pulling arm {self.i} at turn {turn} with seed {self.arm.random_generator.seed} and reward {reward}")
            self.s_i += reward
            self.n_i += 1
            return

        # The pursuit algorithm implemented in our algorithm
        # works in two steps:
        if iteration == 1:
            # The first one aims to update probability as described belows where \beta is
            # learning rate between 0 and 1 includes:
            #       p_i = p_i + \beta * (1 - p_i) if the best estimated arm
            #       p_i = p_i + \beta * (0 - p_i) otherwise
            if b_i == 1:
                self.p_i = self.p_i + self.beta * (1 - self.p_i)
            else:
                self.p_i = self.p_i + self.beta * (0 - self.p_i)
        else:
            # The second one is a pulling step where the pulled arm is chosen randomly
            # in function of a probability matching.
            # Note that the DataOwner node does not performs arm selection with probability.
            if b_i == 1:
                reward = self.arm.pull( turn )
                self.s_i += reward
                self.n_i += 1

class PursuitComp(Comp):
    def __init__(self, K, proto_parameters: ProtoParameters, algo_parameters : PursuitParameters):
        super().__init__(K, proto_parameters)
        self.random_arm_generator = IsolatedRandomGenerator(seed=algo_parameters.random_arm_seeed)

    def select_arm(self, turn, round, values) -> int:
        if round == 1:
            # searching the best max index
            max_index, max_value = 0, values[0]
            for i in range(1, len(values)):
                if max_value < values[i]:
                    max_index, max_value = i, values[i]
            return max_index
        else:
            permuted_pulled_arm_index = self.random_arm_generator.choice(items=range(self.K), weights=values, t=turn)
            return permuted_pulled_arm_index


class PursuitProto(Proto):
    def __init__(self, arms_probs: [float], proto_parameters: ProtoParameters, algo_parameters: PursuitParameters):
        super().__init__(arms_probs, proto_parameters)
        self.algo_parameters = algo_parameters

    def provide_comp(self, *args, **kwargs):
        return PursuitComp(*args,**kwargs,algo_parameters=self.algo_parameters,)

    def provide_do(self, *args, **kwargs):
        return PursuitRi(*args,**kwargs, algo_parameters=self.algo_parameters)

    def provide_controller(self, *args, **kwargs):
        return Controller(*args, **kwargs)

    def select_architecture(self, turn: int, computation_round: int):
        return Architecture.INFORMED

    def nb_computation_rounds_by_turn(self, turn : int) -> int:
        return 2

###################################################
# Algorithms generation facility
###################################################
class PursuitFacility:
    def __init__(
            self,
            reward_seed,
            sigma_seed,
            random_arm_seed,
            beta,
            alpha_seed,
            sk,
            pk,
            cloud_key,
            cd_key,
            arms_probs
    ):
        self.arms_probs = arms_probs
        self.beta = beta
        self.random_arm_seed = random_arm_seed
        self.alpha_seed = alpha_seed
        self.sigma_seed = sigma_seed
        self.reward_seed = reward_seed
        self.sk = sk
        self.pk = pk
        self.cd_key = cd_key
        self.cloud_key = cloud_key

    def create_standard(self) -> PursuitBanditsAlgorithm:
        return PursuitBanditsAlgorithm(
            arms_probs=self.arms_probs,
            algo_parameters=self.__create_algo_parameters()
        )

    def create_generic(self, security: bool) -> PursuitProto:
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

        return PursuitProto(
            arms_probs=self.arms_probs,
            proto_parameters=proto_parameters,
            algo_parameters=self.__create_algo_parameters()
        )

    def __create_algo_parameters(self) -> PursuitParameters:
        return PursuitParameters(
            reward_seed=self.reward_seed,
            sigma_seed=self.sigma_seed,
            random_arm_seed=self.random_arm_seed,
            beta=self.beta,
        )