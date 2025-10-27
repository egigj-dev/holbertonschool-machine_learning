#!/usr/bin/env python3
"""
Normal distribution class
"""


class Normal:
    """Represents a normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Class constructor

        Args:
            data (list): list of data to estimate the distribution
            mean (float): mean of the distribution
            stddev (float): standard deviation of the distribution
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.mean = sum(data) / len(data)
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = self._sqrt(variance)

    # ---------- helper math methods ----------

    def _sqrt(self, x):
        """Compute square root using Newton’s method"""
        if x == 0:
            return 0
        guess = x / 2
        for _ in range(10):
            guess = 0.5 * (guess + x / guess)
        return guess

    def _exp(self, x):
        """Compute exponential using Taylor series"""
        term = 1.0
        total = 1.0
        for n in range(1, 50):
            term *= x / n
            total += term
        return total

    def _erf(self, x):
        """Approximation of error function"""
        t = 1.0 / (1.0 + 0.3275911 * abs(x))
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        poly = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t
        ans = 1 - poly * self._exp(-x * x)
        if x < 0:
            ans = -ans
        return ans

    # ---------- distribution methods ----------

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        return z * self.stddev + self.mean

    def pdf(self, x):
        """Calculates the value of the PDF for a given x-value"""
        pi = 3.141592653589793
        e = 2.718281828459045
        exp = self._exp(-((x - self.mean) ** 2) / (2 * self.stddev ** 2))
        return (1 / (self.stddev * self._sqrt(2 * pi))) * exp

    def cdf(self, x):
        """Calculates the value of the CDF for a given x-value"""
        z = (x - self.mean) / (self.stddev * self._sqrt(2))
        return 0.5 * (1 + self._erf(z))
