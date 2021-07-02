# =======================================================================================
# File: bench.py
# Description: Runs benchmarks.
# =======================================================================================
import csv
import json
import os
import datetime
from sys import stderr
from json.decoder import JSONDecodeError
from random import randint
from time import time
from phe import generate_paillier_keypair
from softmax import SoftmaxFacility
from e_greedy import EGreedyFacility
from e_greedy_decreasing import EGreedyDecreasingFacility
from encryption import generate_symetric_key
from parameters_suggestion import suggest_parameters
from pursuit import PursuitFacility
from thompson_sampling import ThompsonSamplingFacility
from ucb import UCBFacility
from utils import read_arms_from_file
from multiprocessing import Pool, cpu_count
from argparse import ArgumentParser
from pathlib import Path


E_GREEDY = "e-greedy"
E_GREEDY_DECREASING = "e-greedy-decreasing"
SOFTMAX = "boltzmann"
UCB = "ucb"
THOMPSON = "thompson-sampling"
PURSUIT = "pursuit"

ALGORITHMS = [
    E_GREEDY,
    E_GREEDY_DECREASING,
    SOFTMAX,
    UCB,
    THOMPSON,
    PURSUIT,
]


def log_message( message ):
    """Print log message in the standard output."""
    print("[PID:{}@{}] {}".format(
        os.getpid(),
        datetime.datetime.now().strftime("%H:%M:%S-%d-%m-%y"),
        message
    ))


STD_KEY = "std"
GEN_DIS_KEY = "gen-distributed"
GEN_SEC_KEY = "gen-secure"
GEN_CLASS_KEY = "gen"


def instantiate_args( args ):
    """Instantiation of arguments."""
    res = {}
    for args_name, args_value in args.items():
        if args_value == int:
            res[ args_name ] = randint( 10, 1000 )
        else:
            res[args_name] = args_value
    return res


def do_bench( config ):
    """
    This functions aims to perform the benchmarks.

    do_bench is designed to be executed in multi-processing environment.

    Args:
        config: Benchmark configuration
    """

    # reading config
    arms_probs = config['arms']
    N = config['N']
    K = len(arms_probs)
    executions = config['executions']
    bench_filename = config['bench_filename']
    iteration = config["iteration"]
    pk, sk, cloud_key, cd_key = config['keys']
    targeted_algorithms = config["algorithms"]
    force_run = config["force"]
    standards_only = config["standards_only"]

    # starting worker in log
    process_id = os.getpid()
    print(f"[PID:{process_id}] Starting with N = {N}, K = {len(arms_probs)}, nb iterations = {nb_iterations} and bench filename = {bench_filename}")

    # creating results dictionary and try to initialize it with a previously computed results
    results = {
        "arms":  arms_probs,
        "N": N,
        "nb_iterations": nb_iterations,
    }
    if os.path.exists(bench_filename) and os.path.isfile(bench_filename):
        try:
            with open(bench_filename, 'r') as file:
                results = json.loads(file.read())
                log_message("Benchmark file already exists: Successfully parsed")
        except JSONDecodeError as e:
            print(f"File reading failure: {e}")

    # Once results are initialized, runs each execution composed by an list of algorithm,
    for exec in executions:
        # checks if current algorithm must be executed, based on some targeted algorithms given in parameters.
        algo = exec["algo"]
        if targeted_algorithms != [] and algo not in targeted_algorithms: continue

        if algo not in results:
            results[algo] = {}
        log_message(f"Bench {algo} algorithm")

        exec_result = results[algo]

        # checks that current executed iteration has not already be done, otherwise initialize dictionary results
        iteration_id = f"it-{iteration}"
        if iteration_id not in exec_result:
            exec_result[iteration_id] = {}

        # defining arguments for the benchmark
        # arguments must be the same over all algorithms
        if not force_run and "args" in exec_result[iteration_id]:
            args = exec_result[iteration_id]["args"]
        else:
            args = instantiate_args(exec["args"])
            exec_result[iteration_id]["args"] = args
        # modifying reward seed in order to continuous reward
        exec_result[iteration_id]['args']['reward_seed'] = iteration
        log_message(f"Iteration {iteration} uses args {args}")

        # setting up facility in order to create models
        try:
            facility = exec["facility"](**args, cloud_key=cloud_key, sk=sk, pk=pk, cd_key=cd_key, arms_probs=arms_probs)
        except Exception as e:
            print("args: {}".format(args))
            raise Exception("Facility creation failed: ", e)

        # Execute standard algorithm
        if force_run or STD_KEY not in exec_result[iteration_id]:
            log_message(f"Executing {algo}.{iteration_id}.{STD_KEY} ({iteration}/{nb_iterations}, N={N}, K={K})")
            std = facility.create_standard()
            std_start = time()
            std_R = std.play( N )
            std_end = time()
            std_exec_time = std_end - std_start

            exec_result[iteration_id][STD_KEY] = {
                "reward": std_R,
                "time": std_exec_time,
            }
        else:
            log_message(f"Skipping execution {algo}.{iteration_id}.{STD_KEY} ({iteration}/{nb_iterations}, N={N}, K={K})")
        std_R = exec_result[iteration_id][STD_KEY]['reward']

        # exporting results
        with open(bench_filename, "w") as file:
            file.write(json.dumps(results, sort_keys=True, indent=4))

        # when only standards are required, stop at this point
        if standards_only: continue

        # execute generic model with security disabled
        if force_run or GEN_DIS_KEY not in exec_result[iteration_id]:
            log_message(f"Executing {algo}.{iteration_id}.{GEN_DIS_KEY} ({iteration}/{nb_iterations}, N={N}, K={K})")
            gen = facility.create_generic(security=False)
            gen.enable_bench()
            gen_dis_R, gen_exec_time, gen_exec_time_by_nodes = gen.play(N)

            exec_result[iteration_id][GEN_DIS_KEY] = {
                "reward": gen_dis_R,
                "time": gen_exec_time,
                "time_by_components": gen_exec_time_by_nodes,
            }
        else:
            log_message(f"Skipping execution {algo}.{iteration_id}.{GEN_DIS_KEY} ({iteration}/{nb_iterations}, N={N}, K={K})")
            gen_dis_R = exec_result[iteration_id][GEN_DIS_KEY]['reward']

        # exporting results
        with open(bench_filename, "w") as file:
            file.write(json.dumps(results, sort_keys=True, indent=4))

        # execute generic model with security enabled
        if force_run or GEN_SEC_KEY not in exec_result[iteration_id]:
            log_message(f"Executing {algo}.{iteration_id}.{GEN_SEC_KEY} ({iteration}/{nb_iterations}, N={N}, K={K})")
            gen = facility.create_generic(security=True)
            gen.enable_bench()
            gen_sec_R, gen_exec_time, gen_exec_time_by_nodes = gen.play(N)

            exec_result[iteration_id][GEN_SEC_KEY] = {
                "reward": gen_sec_R,
                "time": gen_exec_time,
                "time_by_components": gen_exec_time_by_nodes,
            }
        else:
            log_message(f"Skipping execution {algo}.{iteration_id}.{GEN_SEC_KEY} ({iteration}/{nb_iterations}, N={N}, K={K})")
            gen_sec_R = exec_result[iteration_id][GEN_SEC_KEY]['reward']

        # exporting results
        with open(bench_filename, "w") as file:
            file.write(json.dumps(results, sort_keys=True, indent=4))

        assert std_R == gen_dis_R, "[PID:{}] Total cumulative reward between std and gen distributed are not the same std={} gen={} with args {}, K = {}, N = {} and arms {}".format(
            process_id,
            std_R,
            gen_dis_R,
            args,
            len(arms_probs),
            N,
            arms_probs,
        )

        assert std_R == gen_sec_R, "[PID:{}] Total cumulative reward between std and gen secured are not the same std={} gen={} with args {},K = {}, N = {} and arms {}".format(
            process_id,
            std_R,
            gen_sec_R,
            args,
            len(arms_probs),
            N,
            arms_probs,
        )

    # The returns avoid to raise an exception once the thread has done his work
    return True


def create_executions( epsilon, beta, tau ):
    """
    Returns the execution done at each possible combinaison of iteration, budget N and, number of arms K.
    """
    return [
        {
            'algo': E_GREEDY,
            'facility': EGreedyFacility,
            'args':  {
                "sigma_seed": int,
                "reward_seed": int,
                "random_arm_seed": int,
                "epsilon":  epsilon,
                "epsilon_seed": int,
                "alpha_seed": int,
            }
        },
        {
            'algo': E_GREEDY_DECREASING,
            'facility': EGreedyDecreasingFacility,
            'args': {
                "sigma_seed": int,
                "reward_seed": int,
                "random_arm_seed": int,
                "epsilon_seed": int,
                "alpha_seed": int,
             }
        },
        {
            'algo': PURSUIT,
            'facility': PursuitFacility,
            'args': {
                'beta': beta,
                'sigma_seed':  int,
                'reward_seed': int,
                'random_arm_seed':  int,
                "alpha_seed": int,
            },
        },
        {
            'algo': SOFTMAX, # The so called SOFTMAX is the Softmax algorithm following boltzmann score computation
            'facility': SoftmaxFacility,
            'args': {
                'reward_seed': int,
                'sigma_seed': int,
                'random_arm_seed': int,
                'tau' : tau,
                "alpha_seed": int,
            }
        },
        {
            'algo': UCB,
            'facility': UCBFacility,
            'args': {
                'reward_seed': int,
                'sigma_seed': int,
                "alpha_seed": int,
            }
        },
        {
            'algo': THOMPSON,
            'facility': ThompsonSamplingFacility,
            'args': {
                'reward_seed': int,
                'sigma_seed': int,
                'beta_seed': int,
                'random_arm_seed': int,
                'alpha_seed': int,
            }
        }
    ]


if __name__ == '__main__':
    # parse arguments
    parser = ArgumentParser()
    parser.add_argument( "--output", help="Output directory where bench results are written.", required=True )
    parser.add_argument( "--nb_iterations", help="Number of iterations done for each algorithm, each N and K", type=int, required=True )
    parser.add_argument( "--sort", help="Sort arms to avoid reward up when K increase", action='store_true', default=True )
    parser.add_argument( "--data", help="Data location which contains arms used for the bench", required=True)
    parser.add_argument("--configs", help="Configs location in a CSV format with two columns K and N", required=True)
    parser.add_argument("--cpu", help="Number of workers used to perform the bench. By default, set at available core", type=int, default=cpu_count())
    parser.add_argument("--run", help="If omitted, only displays constructed configuration from parameters", action="store_true", default=False)
    parser.add_argument("--algorithms", help="Specify algorithms to run. Runs all algorithms if omitted", nargs="*", default=[])
    parser.add_argument("--force", help="Specify if realized benchmarks must be considered or not", default=False, action="store_true")
    parser.add_argument("--standards_only", help="Run only standards version, useful to perform an reward analysis", default=False, action="store_true")
    parser.add_argument("--parameters", help="Parameters CSV file that provides parameters for some algorithm. Computed if missing")
    args = parser.parse_args()

    # pre-process arguments
    output_directory = args.output
    if not os.path.exists(output_directory):
        path = Path(output_directory)
        path.mkdir(parents=True, exist_ok=True)

    nb_iterations = args.nb_iterations
    if nb_iterations <= 0:
        stderr.write(f"Error: Nb iterations must be strictly postive, got {nb_iterations}")
        exit(0)

    data_filename = args.data
    if not os.path.exists(data_filename) or not os.path.isfile(data_filename):
        stderr.write(f"Data file '{data_filename}' does not exist or not a file")
        exit(0)

    # load data
    data = read_arms_from_file(data_filename)
    if args.sort:
        data.sort(reverse=True)

    # loads parameters if file is specified, or computes them if missing
    csv.register_dialect(
        'dialect',
        quotechar='"',
        skipinitialspace=True,
        quoting=csv.QUOTE_NONE,
        lineterminator='\n',
        strict=True
    )
    if args.parameters:
        with open(args.parameters) as file:
            print("[MASTER] Loading parameters...")
            configs_reader = csv.DictReader(file, dialect='dialect')
            for row in configs_reader:
                epsilon, tau, beta = float(row["epsilon"]), float(row["tau"]), float(row['beta'])
                pass
    else:
        print("[MASTER] Computing parameters...")
        epsilon, tau, beta = suggest_parameters( data )
    print(f"[MASTER] Considered parameters: epsilon = {epsilon}, tau = {tau} and beta = {beta}")

    # ensuring that specified algorithms exists
    for algo in args.algorithms:
        if algo not in ALGORITHMS:
            print(f"Error: {algo} algorithm not recognized")
            exit(1)

    # creating security keys to avoid unnecessary keys creation
    print("[MASTER] Generating keys...")
    pk, sk = generate_paillier_keypair()
    cloud_key = generate_symetric_key()
    cd_key = generate_symetric_key()
    keys = pk, sk, cloud_key, cd_key
    print("[MASTER] Keys generated")

    print("[MASTER] Building configs")
    executions = create_executions( epsilon, beta, tau )
    config = []
    configs_history = []
    with open(args.configs) as file:
        configs_reader = csv.DictReader(file)
        for row in configs_reader:
            N, K = int(row["N"]), int(row["K"])
            if (N, K) in configs_history: continue
            configs_history.append((N, K))
            for iteration in range(1, nb_iterations + 1):
                config.append(
                    {
                        'arms': data[:K],
                        'N': N,
                        'executions': executions,
                        'bench_filename': os.path.join(output_directory, f'benchmark_N_{N}_K_{K}_it_{iteration}.json'),
                        'iteration': iteration,
                        'keys': keys,
                        'algorithms': args.algorithms,
                        'force': args.force,
                        'standards_only': args.standards_only,
                    }
                )
    print("[MASTER] Configs built")

    nb_cpu = args.cpu
    if not args.run:
        print(f"[MASTER] Details {len(config)} bench launched with pool of {nb_cpu} workers over {nb_iterations} iterations")
        print("=============================")
        print("Algorithms:", args.algorithms)
        print("Standards only ?", "YES" if args.standards_only else "NO")
        print("-----------------------------")
        for c in config:
            print("- N = {}, K = {}, it = {}, ".format(c['N'], len(c["arms"]), c['iteration']))
        exit(0)
    else:
        # launch processes by using workers
        if nb_cpu == 1:
            print(f"[MASTER] Running {len(config)} bench on main thread over {nb_iterations} iterations")
            for c in config:
                do_bench(c)
        else:
            print(f"[MASTER] Running {len(config)} bench with pool of {nb_cpu} workers over {nb_iterations} iterations")
            with Pool(processes=nb_cpu) as pool:
                pool.map( do_bench, config )
