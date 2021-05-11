import argparse
import pandas as pd

from logger import *
from data import Data

RESULT_FILENAME = "describe.csv"

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str)
parser.add_argument('-f', help="Save statistics to file", action='store_true')
parser.add_argument('-n', help="Name of the file with statistics (see -f)", type=str, default=RESULT_FILENAME)

args = parser.parse_args()

try:
    datasetDF = pd.read_csv(args.filename, index_col=0)
except:
    log.error("Data file (%s) not found or invalid csv file." % args.filename)
    exit(1)

datasetDF = datasetDF.select_dtypes(include=[int, float])

data = Data(datasetDF)
data.describe(args.f, args.n)
