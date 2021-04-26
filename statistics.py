import numpy as np
import math

def filter_array(x):
    return [xi for xi in x if not np.isnan(xi)]

def sum(x):
    x = filter_array(x)
    sum = 0
    for xi in x:
        sum += xi
    return sum

def count(x):
    return len(filter_array(x))

def mean(x):
    return sum(x) / count(x)

def std(x):
    x = filter_array(x)
    meanX = mean(x)
    deviations = [(xi - meanX) ** 2 for xi in x]
    variance = sum(deviations) / (count(deviations) - 1)
    return math.sqrt(variance)

def min(x):
    x = filter_array(x)
    minX = x[0]
    for xi in x:
        if xi < minX:
            minX = xi
    return minX

def percentile(x, q):
    x = x.sort_values().reset_index(drop=True)
    findex = q * (count(x) - 1) / 100
    lower = math.floor(findex)
    fraction = findex - lower
    return x[lower] + (x[lower + 1] - x[lower]) * fraction

def max(x):
    x = filter_array(x)
    maxX = x[0]
    for xi in x:
        if xi > maxX:
            maxX = xi
    return maxX
