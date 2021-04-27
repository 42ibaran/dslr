import argparse
import pandas as pd

from logger import *
from data import Data

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)

args = parser.parse_args()

try:
    datasetDF = pd.read_csv(args.filename, index_col=0)
except:
    log.error("Data file (%s) not found or invalid csv file." % args.filename)
    exit(1)

data = Data(datasetDF)
data.pair_plot()
