# =============================================================================
# file: parameters_suggestion.py
# description: Script that suggests bandits algorithm parameters optimized for a given dataset.
# =============================================================================

from e_greedy import EpsilonGreedyBanditsAlgorithm, EGreedyParameters
from softmax import SoftmaxBanditsAlgorithm, SoftmaxParameters
from pursuit import PursuitBanditsAlgorithm, PursuitParameters
from utils import read_arms_from_file


def suggest_parameters( probs, nb_iteration = 10, N = 50000 ) -> (float, float, float):
    """Suggests parameters optimized for arm probabilities.

    :Returns (epsilon, tau, beta)
    """
    epsilon_key = "epsilon"
    tau_key = "tau"
    beta_key = "beta"

    parameters_config = {
        epsilon_key:    {"min": 0,    "max": 1,   "step": 0.1,  "nb_decimals": 1 },
        tau_key:        {"min": 0.01, "max": 0.1, "step": 0.01, "nb_decimals": 2 },
        beta_key:       {"min": 0.1,  "max": 1,   "step": 0.1,  "nb_decimals": 1 },
    }
    results = {
        epsilon_key: {},
        tau_key: {},
        beta_key: {}
    }



    sigma_seed = 156
    random_arm_seed = 1
    
    # e_greedy
    epsilon_seed = 123
    config = parameters_config[epsilon_key]
    min_val, max_val, step, nb_decimals = config["min"], config["max"], config["step"], config["nb_decimals"]
    nb_steps = int((max_val - min_val) / step)
    for step_index in range(nb_steps):
        epsilon = round( min_val + step * step_index, nb_decimals )
        
        # computing mean reward over several iterations
        rewards = []
        for iteration in range(nb_iteration):
            # creating algorithm
            algo = EpsilonGreedyBanditsAlgorithm(
                arms_probs=probs,
                algo_parameters=EGreedyParameters(
                    epsilon=epsilon,
                    epsilon_seed=epsilon_seed,
                    reward_seed=iteration,
                    sigma_seed=sigma_seed,
                    random_arm_seed=random_arm_seed,
                )
            )
            
            # adding reward
            rewards.append(algo.play(N = N))
        
        # saving mean rewards
        results[epsilon_key][epsilon] = sum(rewards) / len(rewards)
    
    # pursuit (beta)
    config = parameters_config[beta_key]
    min_val, max_val, step, nb_decimals = config["min"], config["max"], config["step"], config["nb_decimals"]
    nb_steps = int((max_val - min_val) / step)
    for step_index in range(nb_steps):
        beta = round( min_val + step * step_index, nb_decimals )
        
        # computing mean reward over several iterations
        rewards = []
        for iteration in range(nb_iteration):
            # creating algorithm
            algo = PursuitBanditsAlgorithm(
                arms_probs=probs, 
                algo_parameters=PursuitParameters(
                    beta=beta,
                    reward_seed=iteration,
                    sigma_seed=sigma_seed,
                    random_arm_seed=random_arm_seed,
            ))
            
            # adding reward
            rewards.append(algo.play(N = N))
        
        # saving mean rewards
        results[beta_key][beta] = sum(rewards) / len(rewards)
        
    # Softmax (tau=
    config = parameters_config[tau_key]
    min_val, max_val, step, nb_decimals = config["min"], config["max"], config["step"], config["nb_decimals"]
    nb_steps = int((max_val - min_val) / step)
    for step_index in range(nb_steps):
        tau = round( min_val + step * step_index, nb_decimals )
        
        # computing mean reward over several iterations
        rewards = []
        for iteration in range(nb_iteration):
            # creating algorithm
            algo = SoftmaxBanditsAlgorithm(
                arms_probs=probs,
                algo_parameters=SoftmaxParameters(
                    tau=tau,
                    reward_seed=iteration,
                    sigma_seed=sigma_seed,
                    random_arm_seed=random_arm_seed,
                )
            )
            
            # adding reward
            rewards.append(algo.play(N = N))
        
        # saving mean rewards
        results[tau_key][tau] = sum(rewards) / len(rewards)

    # searching the best parameters for each algorithm
    best_argument = lambda dict_results: max(dict_results.items(), key=lambda couple: couple[1])
    return (
        best_argument(results[epsilon_key]),
        best_argument(results[tau_key]),
        best_argument(results[beta_key]),
    )


if __name__ == '__main__':
    N = 40000
    probs = read_arms_from_file( "data/JesterLarge.txt" )
    print(suggest_parameters( probs ))
