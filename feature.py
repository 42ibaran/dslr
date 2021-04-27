import pandas as pd
import numpy as np
import math

import statistics as st

STATS_ROWS = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']

class Feature():
    series = None

    def __init__(self, series):
        self.series = series
        self.filter_series()
        # self.standardize()

    def standardize(self):
        self.mean_denorm = self.mean()
        self.std_denorm = self.std()
        self.series = (self.series - self.mean_denorm) / self.std_denorm

    def filter_series(self):
        self.series = pd.Series([xi for xi in self.series if not np.isnan(xi)])

    def get_statistics(self):
        return [
            self.count(),
            self.mean(),
            self.std(),
            self.min(),
            self.percentile(25),
            self.percentile(50),
            self.percentile(75),
            self.max()
        ]

    def sum(self):
        sum = 0
        for xi in self.series:
            sum += xi
        return sum

    def count(self):
        return len(self.series)

    def mean(self):
        return self.sum() / self.count()

    def std(self):
        mean = self.mean()
        deviations = Feature(pd.Series([(xi - mean) ** 2 for xi in self.series]))
        variance = deviations.sum() / (deviations.count() - 1)
        return math.sqrt(variance)

    def min(self):
        minX = self.series[0]
        for xi in self.series:
            if xi < minX:
                minX = xi
        return minX

    def percentile(self, q):
        x = self.series.sort_values().reset_index(drop=True)
        findex = q * (self.count() - 1) / 100
        lower = math.floor(findex)
        fraction = findex - lower
        return x[lower] + (x[lower + 1] - x[lower]) * fraction

    def max(self):
        maxX = self.series[0]
        for xi in self.series:
            if xi > maxX:
                maxX = xi
        return maxX
