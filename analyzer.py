# =========================================================================
# file: analyzer.py
# description: Correctness analyzing tool.
# =========================================================================
def parse_line( line ):
    """Returns data located in the line."""
    tokens = line.split()
    return  int(tokens[1]), tokens[2], (
        int(tokens[4]), # n_i
        int(tokens[6]), # s_i
    )

def validate_correctness_from_logfile(filename) -> bool:
    """Returns True if the correctness is stated, False otherwise."""
    gen = {}
    std = {}
    with open(filename, 'r') as file:
        N, K = 0, 0
        for line_index, line in enumerate(file.readlines()):
            if line_index == 0:
                tokens = line.split()
                N, K = int(tokens[0]), int(tokens[1])
                continue
            if "GEN" not in line and "STD" not in line:
                continue

            if line.count( "GEN" ) != 0:
                for l in line.split("#"):
                    if l.count("GEN") == 0: continue

                    l = l.replace("GEN", "")
                    if l.strip() == "": continue
                    t, i, data = parse_line(l)
                    gen[(t, i)] = data
            elif line.count("STD") != 0:
                line = line.replace( "STD", "" )
                t, i, data = parse_line(line)
                std[(t, i)] = data


        # compute diff
        has_diff = False
        for key, value in std.items():
            if key not in gen: raise Exception(f"{key} in std but not in gen")
            t, i = key
            std_si, std_ni = value
            gen_si, gen_ni = gen[key]
            if std_si != gen_si or std_ni != gen_ni:
                has_diff = True
                print( f"Diff at turn {t} for arm {i}: (GEN: si={gen_si}, ni={gen_ni}), (STD: si={std_si}, ni={std_ni})" )

        return not has_diff

import sys
def check_correctness( arms_probs : [float], standard_algorithm, generic_algorithm, N, filename : str ):
    """Runs standard and generic algorithms and ensures that correctness is ensured"""

    # opening output redirection
    previous_stdout = sys.stdout
    sys.stdout = open(filename, "w")
    print(f"{N} {len(arms_probs)}")

    debug = True
    standard_algorithm.play(N, debug=debug)
    generic_algorithm.play(N, debug=debug)

    # closing output redirection
    sys.stdout.close()
    sys.stdout = previous_stdout

    is_valid = validate_correctness_from_logfile(filename)
    if not is_valid:
        raise Exception("Aborting program: correctness not stated")

