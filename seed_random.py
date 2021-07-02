# ====================================================================================
# file: seed_random.py
# description: Random generator functions.
# ====================================================================================
from random import random, seed, randint, choices


class IsolatedRandomGenerator:
    """
    Random number generator able to produces random float and int, with respect to a given seed.
    """
    def __init__(self, seed):
        self.seed = seed

    def random(self, t) -> float:
        """Returns a random float between 0 and 1 includes given a time step."""
        seed(self.seed + t)
        value = random()

        return value

    def randint(self, t, min, max) -> int:
        """
        Returns a random int between min and max includes given a time step.
        """
        seed(self.seed + t)
        value = randint(min, max)
        return value

    def choice(self, items, weights, t):
        """Returns randomly a item contained in a given list of items, with a given probability weights."""
        seed(self.seed + t)
        value = choices( items, weights=weights, k=1 )[0]
        return value


class IsolatedBernoulliArm:
    """Simulates a bandit arm which returns a reward with a given probability, with respect to a reward seed."""
    def __init__(self, p, seed):
        self.p = p
        self.random_generator = IsolatedRandomGenerator(seed=seed)

    def pull(self, t) -> int:
        """Returns a reward 0 or 1 randomly with respect to a time step t."""
        random_value = self.random_generator.random(t)
        reward = int(random_value < self.p)
        return reward

    def __str__(self):
        return str(self.p)