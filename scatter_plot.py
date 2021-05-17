import argparse
import pandas as pd

from logger import *
from data import Data

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str)

args = parser.parse_args()

try:
    data = Data(args.filename)
    feat1, feat2 = data.find_most_similar()
    data.scatter_plot(feat1, feat2)
except Exception as e:
    log.error(e)
    exit(1)
