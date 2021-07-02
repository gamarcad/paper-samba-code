# ==============================================================================
# file: permutation.py
# description: Permutation implementation used in the protocol.
# ==============================================================================
from random import shuffle, getstate, setstate, seed


class IsolatedPermutation:
    """Permutation class allows to permute a list and to invert the permutation of a permuted list.

    This implementation has been designed to be controlled by a seed, useful to applied the same permutation
    over several algorithms.
    """

    @staticmethod
    def new(nb_items: int, perm_seed: int, turn: int):
        """Creates and returns an initialized permutation adapted to permute a list of a given size.

       Args:
           nb_items: Number of items in the list to permute or to invert.
           perm_seed: Permutation seed used to initialize the random state.
           turn: Integer used to create the same permutation at a given time step.
       """
        permutation = IsolatedPermutation(nb_items, perm_seed)
        permutation.reset(turn)
        return permutation

    def __init__(self, nb_items, perm_seed: int):
        self.nb_items = nb_items
        self.perm_seed = perm_seed

    def reset(self, turn: int):
        """Resets the permutation for a given time step."""
        permuted_index = [i for i in range(self.nb_items)]

        save_state = getstate()
        seed(self.perm_seed + turn)
        shuffle(permuted_index)
        setstate(save_state)

        self.__permutation = {index: perm for index, perm in enumerate(permuted_index)}
        self.__inverse = {perm: index for index, perm in enumerate(permuted_index)}

    def permute(self, values):
        """Returns the permuted list given in parameter.

        Args:
            values: List to be permuted.
        """
        if len(values) != self.nb_items:
            raise Exception("Cannot permutate list with a size different from {}".format(self.nb_items))

        permuted_values = [None for _ in range(self.nb_items)]
        for real_index, permuted_index in self.__permutation.items():
            permuted_values[permuted_index] = values[real_index]
        return permuted_values

    def invert_permutation(self, permuted_values):
        """Returns the permuted list where each item has been replaced to his original position.

        Args:
            permuted_values: Permuted list to be inverted.
        """
        if len(permuted_values) != self.nb_items:
            raise Exception("Cannot permutate list with a size different from {}".format(self.nb_items))

        values = [None for _ in range(self.nb_items)]
        for real_index, permuted_index in self.__permutation.items():
            values[real_index] = permuted_values[permuted_index]
        return values

    def invert_permuted_index(self, permuted_index):
        """Returns the original position of the given permuted index.

        Args:
            permuted_index: Permuted index.
        """
        return self.__inverse[permuted_index]

    def __str__(self):
        return str(
            "Permutation({} -> {}), Inverse({} -> {})".format(self.__permutation.keys(), self.__permutation.values(),
                                                              self.__inverse.keys(), self.__inverse.values()))
