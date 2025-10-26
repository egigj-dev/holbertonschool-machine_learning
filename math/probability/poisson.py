#!/usr/bin/env python3
"""
Poisson tasks class
"""


class Poisson:
    """Represents a Poisson distribution"""
    def __init__(self, data=None, lambtha=1.):
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """A function that calculates the PMF/a given number of successes"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        mean = self.lambtha
        return (exp(-mean) * mean ** k) / factorial(k)

    def cdf(self, k):
        """Calculate the value of the CDF for a given number of “successes”"""
        if not isinstance(k, int): 
            k = int(k)
        if k < 0: 
            return 0
        # CDF = sum of PMF from 0 to k
        return sum(self.pmf(i) for i in range(k + 1))
