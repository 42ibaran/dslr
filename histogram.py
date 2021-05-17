import argparse
import pandas as pd

from logger import *
from data import Data

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str)
parser.add_argument("--all", "-a", help="plot histograms for all features (descending homogeneity order)", action='store_true')

args = parser.parse_args()

try:
    data = Data(args.filename)
    result = data.find_most_homogeneous(all=args.all)
    if args.all:
        data.histogram_all(result)
    else:
        data.histogram_one(result)
except Exception as e:
    log.error(e)
    exit(1)
