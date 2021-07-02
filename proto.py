# ====================================================================================
# file: proto.py
# description: Contains generic implementation of Samba.
# ====================================================================================
from abc import abstractmethod
from copy import copy
from time import perf_counter

from phe import generate_paillier_keypair

from encryption import generate_symetric_key, AES256GCMCipher
from permutation import IsolatedPermutation
from seed_random import IsolatedBernoulliArm, IsolatedRandomGenerator
from utils import Timer


class Architecture:
    """Architecture is an enumeration which contains the optimized random and informed flags."""
    RANDOM = 0
    INFORMED = 1


class ProtoParameters:
    """Container that contains all parameters used in the protocol.
    """

    @staticmethod
    def new(alpha_seed: int, reward_seed: int, sigma_seed: int, security=True, asymmetric_key_length=2048,
            random_arm_seed: int = 1):
        """Creates and Returns a new ProtoParameters object, initialized with given parameters.
        """
        cloud_key, cd_key = generate_symetric_key(), generate_symetric_key()
        if security:
            pk, sk = generate_paillier_keypair(n_length=asymmetric_key_length)
        else:
            pk, sk = None, None

        return ProtoParameters(
            cloud_key=cloud_key,
            cd_key=cd_key,
            pk_dc=pk,
            sk_dc=sk,
            alpha_seed=alpha_seed,
            reward_seed=reward_seed,
            sigma_seed=sigma_seed,
            random_arm_seed=random_arm_seed,
            security=security,
        )

    @staticmethod
    def new_from_keys(alpha_seed: int, reward_seed: int, sigma_seed: int, pk, sk, cloud_key, cd_key,
                      random_arm_seed: int = 1):
        """Returns a new instance of ProtoParameters initialized with proto argument and security keys."""
        assert pk is not None, "Public Key cannot be set to None"
        assert sk is not None, "Secret Key cannot be set to None"
        assert cloud_key is not None, "Cloud Key cannot be set to None"
        assert cd_key is not None, "CD Key cannot be set to None"

        return ProtoParameters(
            cloud_key=cloud_key,
            cd_key=cd_key,
            pk_dc=pk,
            sk_dc=sk,
            alpha_seed=alpha_seed,
            reward_seed=reward_seed,
            sigma_seed=sigma_seed,
            random_arm_seed=random_arm_seed,
            security=True
        )

    def disable_security(self):
        """Disable security during the execution of the protocol."""
        proto_copy = copy(self)
        proto_copy.security = False
        return proto_copy

    def __init__(self, cloud_key, cd_key, pk_dc, sk_dc, alpha_seed, reward_seed, sigma_seed, random_arm_seed, security):
        # Cloud and Comp/DataOwners keys
        self.cloud_cipher = AES256GCMCipher(cloud_key)
        self.cd_cipher = AES256GCMCipher(cd_key)

        # DataCustomer's asymmetric public and private key
        self.pk_dc = pk_dc
        self.sk_dc = sk_dc

        # the alpha seed is used to generate the same over all DataOwners
        self.alpha_seed = alpha_seed

        # other seeds are includes in order to control the randomness
        self.random_arm_seed = random_arm_seed
        self.reward_seed = reward_seed
        self.sigma_seed = sigma_seed

        # a boolean state used to highlight if security has been enabled or not
        self.security = security


class DataOwner:
    """DataOwner class handles the local variables used in the protocol, as defined with our assumptions with a
    vertical cutting in the federated learning context.

    """

    def __init__(self, arm_prob, K, i, proto_parameters: ProtoParameters):
        # Bandits related properties
        self.arm = IsolatedBernoulliArm(arm_prob, proto_parameters.reward_seed)
        self.i = i
        self.s_i = 0
        self.n_i = 0
        self.K = K

        # Security related properties
        # Note that the mask generator is not initialized because of the controller
        # must compute the alpha seed and send it at step 1.
        self.cd_cipher = proto_parameters.cd_cipher
        self.cloud_cipher = proto_parameters.cloud_cipher
        self.pk_key = proto_parameters.pk_dc
        self.mask_generator = None

        # bench related properties: does not belong to the model
        self.bench = False
        self.timer = Timer()

    def enable_bench(self):
        """Enables benchmark in order to measure the execution time performed by the algorithm."""
        self.bench = True

    def receive_encrypted_pulling_bit(self, turn: int, computation_round: int, encrypted_b_i):
        """Update local variables with a given encrypted pulling bit."""
        b_i = self.cd_cipher.decrypt(encrypted_b_i)
        self.receive_pulling_bit(turn, computation_round, b_i)

    def receive_pulling_bit(self, turn: int, computation_round: int, b_i):
        """Update local variables with a given pulling bit."""
        self.handle_select(turn, computation_round, b_i)

    def mask_value(self, mask: float, v_i: float) -> float:
        """Returns the v_i value masked by using the given mask.

        The v_i value is masked by performing a product between v_i and the given mask.

        Args:
          mask: Mask used to mask the real v_i value
          v_i: The value to be masked.

        Returns:
            The v_i value masked with the given mask.
        """
        return mask * v_i

    @abstractmethod
    def compute_value(self, turn: int, iteration: int) -> float:
        """Returns a value computed by each node at specified turn and computation round.

        Args:
           turn: Turn where value must be computed;
           iteration: iteration where the value is computed.

        Returns:
            The generic value computed by each node R_i at specified turn and computation rond.
        """
        ...

    @abstractmethod
    def handle_select(self, turn: int, iteration: int, b_i: int):
        """Handles pulling from a given pulling bit at a specified turn and computation round.

        The received pulling bit b_i must be equals to either 0 to specify that arm is not pulled or either 1
        to specify that arm is pulled at given turn and computation round.

        The behavior of this function is not specific.

        Args:
          turn: Turn where value must be computed;
          iteration: iteration where the value is computed.

        Returns:
           The generic value computed by each node R_i at specified turn and computation rond.
        """
        ...

    def get_encrypted_partial_cumulative_reward(self):
        """Returns the encrypted partial cumulative reward s_i."""
        return self.pk_key.encrypt(self.s_i)

    def get_partial_cumulative_reward(self):
        """Returns the partial cumulative reward s_i."""
        return self.s_i

    def compute_encrypted_masked_scores(self, turn, computation_round):
        """Returns the encrypted masked values."""
        return self.cd_cipher.encrypt(self.compute_masked_scores(turn, computation_round))

    def compute_masked_scores(self, turn, computation_round):
        """Returns the masked scores"""
        mask = self.mask_generator.random(turn)
        vi = self.compute_value(turn, computation_round)
        return mask * vi

    def receive_alpha_seed_from_controller(self, alpha_seed):
        """Updates mask generator with the received seed."""
        self.mask_generator = IsolatedRandomGenerator(seed=alpha_seed)


class Controller:
    """
    Controller handles the permutation done at step 3 and the invert of permutation at step 4.
    """

    def __init__(self, K, proto_parameters: ProtoParameters):
        # Bandits related property
        self.K = K

        # Security related property
        self.cloud_cipher = proto_parameters.cloud_cipher

        # Randomness sigma seed
        self.sigma_seed = proto_parameters.sigma_seed
        self.alpha_seed = proto_parameters.alpha_seed

        # bench related properties: does not belong to the model
        self.bench = False
        self.timer = Timer()

    def enable_bench(self): self.bench = True


class Comp:
    """
    Comp handles performs the arm selection function on scores sends by all DataOwners and forwards by the Controller.
    """

    def __init__(self, K, proto_parameters: ProtoParameters):
        self.K = K
        self.cd_cipher = proto_parameters.cd_cipher
        self.random_arm_generator = IsolatedRandomGenerator(proto_parameters.random_arm_seed)

        # bench related property: does not belong to the model
        self.bench = False
        self.timer = Timer()

    def enable_bench(self):
        """Enables benchmarks in order ot measure the execution time of Comp server."""
        self.bench = True

    def compute_encrypted_pulling_bits(self, turn: int, computation_round: int, encrypted_values: [float],
                                       architecture):
        """
        Returns the encrypted pulling bits.
        """
        if architecture == Architecture.INFORMED:
            values = [
                self.cd_cipher.decrypt(encrypted_value)
                for encrypted_value in encrypted_values
            ]
            i_m = self.select_arm(turn, computation_round, values)
        else:
            i_m = self.random_arm_generator.randint(turn, 0, self.K - 1)
        encrypted_pulling_bits = [
            self.cd_cipher.encrypt(
                1 if i == i_m else 0
            )
            for i in range(self.K)
        ]
        return encrypted_pulling_bits

    def compute_pulling_bits(self, turn: int, computation_round: int, values: [float], architecture):
        """
        Returns the pulling bits.
        """
        if architecture == Architecture.INFORMED:
            i_m = self.select_arm(turn, computation_round, values)
        else:
            i_m = self.random_arm_generator.randint(turn, 0, self.K - 1)
        return [
            1 if i == i_m else 0
            for i in range(self.K)
        ]

    @abstractmethod
    def select_arm(self, turn: int, computation_round: int, values: [float]) -> int:
        """Selects and returns the permuted index of the arm to be pulled.

        This function is called only when the selected architecture at given turn and computation round
        is Architecture.INFORMED. Provided values list has been permuted by the AS before to be sent.
        The permutation is not known by Comp hence the returned index of the pulled arm
        must be also permuted.

        Args:
           turn: Turn where architecture must be chosen.
           computation_round: Iteration over the architecture, in case there are several computations roynd in the
                same turn.
           values: permuted List of values where each item of this list is the result of the local computation function
                performed on a Ri.

        Returns:
           The pulled arm's permuted index.
        """
        ...


class DataCustomer:
    """
    DataCustomer is a cloud extern entity which sends the budget and get back the final cumulative from the server.
    """

    def __init__(self, proto_parameters: ProtoParameters):
        self.sk_dc = proto_parameters.sk_dc

        # bench related properties: does not belong to the model
        self.bench = False
        self.timer = Timer()

    def enable_bench(self): self.bench = True

    def decrypt_and_return_encrypted_reward(self, encrypted_R):
        return self.sk_dc.decrypt(encrypted_R)


class Proto:
    """
    Proto is the object that runs the protocol
    """

    def __init__(self, arms_probs: [float], proto_parameters: ProtoParameters):
        # Bandits related properties
        self.K = len(arms_probs)
        self.arms_probs = arms_probs

        # Prototype parameters
        self.proto_parameters = proto_parameters

        # debug and bench properties: does not belong to the model
        self.debug = False
        self.bench = False
        self.security_enabled = proto_parameters.security

    def enable_bench(self):
        """Enables benchmark."""
        self.bench = True

    def disable_security(self):
        """Disables encryption in protocol."""
        self.security_enabled = False

    def play(self, N: int, debug=False):
        """Runs the protocol."""
        # In the protocol, K + 3 entities are requires: K arms, the Data Client, the Controller and the Comp node
        self.controller = self.provide_controller(K=self.K, proto_parameters=self.proto_parameters)
        self.comp_node = self.provide_comp(K=self.K, proto_parameters=self.proto_parameters)
        self.dc_node = DataCustomer(proto_parameters=self.proto_parameters)
        self.do_nodes = [
            self.provide_do(
                arm_prob=arm_prob,
                proto_parameters=self.proto_parameters,
                K=self.K,
                i=i,
            )
            for i, arm_prob in enumerate(self.arms_probs)
        ]

        # Starting timer in order to measure the protocol execution time
        start = perf_counter()

        # =============================================================
        # Beginning of the protocol
        # =============================================================

        # Step 0: The DataCustomer sends to the Controller the budget N
        # As only this function requires the budget N, does not need to execute this step.

        # Step 1: The Controller sends the budget N and the alpha seed to all DataOwners
        for do in self.do_nodes:
            with do.timer:
                do.receive_alpha_seed_from_controller(self.controller.alpha_seed)

        # Initial exploration stage where each arm is pulled a fixed number of times
        turn = 1
        nb_pulling = 1
        for _ in range(nb_pulling):
            for do in self.do_nodes:
                with do.timer:
                    do.handle_select(turn=turn, iteration=-1, b_i=1)
                turn += 1

        # Core of the protocol
        while turn <= N:

            # Print in the standard output s_i and n_i for each arm, useful to ensure the correctness.
            if debug:
                for i, do in enumerate(self.do_nodes):
                    print(f"#GEN Turn {turn} R{i} si {do.s_i} ni {do.n_i}")

            nb_rounds = self.nb_computation_rounds_by_turn(turn)
            for computation_round in range(1, nb_rounds + 1):
                # We does the permutation directly in the protocol because of we use the permutation twice.
                with self.controller.timer:
                    permutation = IsolatedPermutation.new(
                        nb_items=self.K,
                        perm_seed=self.controller.sigma_seed,
                        turn=turn
                    )

                # The chosen architecture depends on turn and computation turn
                architecture = self.select_architecture(turn, computation_round=computation_round)

                # More common case where the next pulled arm is chosen by using global computations e.g. argmax or
                # probability matching.
                if architecture == Architecture.INFORMED:
                    # Step 2: Each DataOwner sends his computed value to the Controller.
                    masked_values = []
                    for do in self.do_nodes:
                        if self.security_enabled:
                            with do.timer:
                                vi = do.compute_encrypted_masked_scores(turn, computation_round)
                        else:
                            with do.timer:
                                vi = do.compute_masked_scores(turn, computation_round)
                        masked_values.append(vi)

                    # Step 3: The Controller performs a permutation on received values and sends the result to Comp node
                    with self.controller.timer:
                        permuted_masked_values = permutation.permute(masked_values)
                else:
                    permuted_masked_values = None

                # Step 4: The Comp node sends permuted pulling bits to the Controller
                if self.security_enabled:
                    with self.comp_node.timer:
                        permuted_pulling_bits = self.comp_node.compute_encrypted_pulling_bits(
                            turn=turn,
                            computation_round=computation_round,
                            encrypted_values=permuted_masked_values,
                            architecture=architecture,
                        )
                else:
                    with self.comp_node.timer:
                        permuted_pulling_bits = self.comp_node.compute_pulling_bits(
                            turn=turn,
                            computation_round=computation_round,
                            values=permuted_masked_values,
                            architecture=architecture
                        )

                # Step 5: The Controller inverts the permutation and sends pulling bit to each DataOwner node
                with self.controller.timer:
                    pulling_bits = permutation.invert_permutation(permuted_pulling_bits)
                for i, do in enumerate(self.do_nodes):
                    if self.security_enabled:
                        with do.timer:
                            do.receive_encrypted_pulling_bit(turn, computation_round, pulling_bits[i])
                    else:
                        with do.timer:
                            do.receive_pulling_bit(turn, computation_round, pulling_bits[i])

            turn += 1

        # Steps 6, 7: Sends total cumulative reward s_1 + ... + s_K to the DataCustomer
        if self.security_enabled:
            # Step 6: Each R_i sends his encrypted partial cumulative reward to the Controller
            r0 = self.do_nodes[0]
            with r0.timer:
                encrypted_R = r0.get_encrypted_partial_cumulative_reward()
            for i in range(1, self.K):
                do = self.do_nodes[i]
                with do.timer:
                    encrypted_R += do.get_encrypted_partial_cumulative_reward()

            # Step 7: Sends the encrypted total cumulative reward to the DataCustomer
            with self.dc_node.timer:
                R = self.dc_node.decrypt_and_return_encrypted_reward(encrypted_R)
        else:
            # Step 6: Each Ri sends his partial cumulative reward to the Controller
            r0 = self.do_nodes[0]
            with r0.timer:
                R = r0.get_partial_cumulative_reward()

            # Step 7: Sends the total cumulative reward the the DataCustomer
            for i in range(1, self.K):
                do = self.do_nodes[i]
                with do.timer:
                    R += do.get_partial_cumulative_reward()

        # Stops the execution counter
        end = perf_counter()

        # When benchmark is required, reward, execution time and execution time by components are returned
        # Otherwise, only the reward is returned
        if self.bench:
            return R, end - start, self.time_by_component()
        else:
            return R

    def time_by_component(self):
        """Returns a dictionary where entries are components and values are execution time.
        """
        times = {
            f"R{i}": ri.timer.execution_time_in_seconds() for i, ri in enumerate(self.do_nodes)
        }

        times["as"] = self.controller.timer.execution_time_in_seconds()
        times["dc"] = self.dc_node.timer.execution_time_in_seconds()
        times["comp"] = self.comp_node.timer.execution_time_in_seconds()

        return times

    @abstractmethod
    def provide_do(self, arm_prob: float, K: int, i: int, proto_parameters: ProtoParameters) -> DataOwner:
        """Function that returns an initialized DataOwner object."""
        pass

    @abstractmethod
    def provide_controller(self, K: int, proto_parameters: ProtoParameters) -> Controller:
        """Function that returns an initialized Controller object."""
        pass

    @abstractmethod
    def provide_comp(self, K, proto_parameters: ProtoParameters) -> Comp:
        """Function that returns an initialized Com object."""
        pass

    def nb_computation_rounds_by_turn(self, turn: int) -> int:
        """Returns the number of computation rounds at specified turn.

        Args:
          turn: Turn in which caller want to know the number of communication round.

        Returns:
           The number of computation rounds at specified turn.
           This number is greater or equals than 1.
        """
        return 1

    @abstractmethod
    def select_architecture(self, turn: int, computation_round: int):
        """Returns the chosen architecture at a specified turn and computation round.

        Specified architecture must either Architecture.INFORMED to perform a pulling from data computed by each
        nodes either Architecture.BLIND to perform a pulling without any data.

        Args:
           turn: Turn where architecture must be chosen.
           computation_round: Iteration over the architecture, in case there are several computations roynd in the
                same turn.

        Returns:
           either Architecture.INFORMED either Architecture.BLIND.
        """
        ...
