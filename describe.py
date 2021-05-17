import argparse
import pandas as pd

from logger import *
from data import Data

RESULT_FILENAME = "describe.csv"

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str)
parser.add_argument('--to-file', '-tf', help="Save statistics to file", action='store_true')
parser.add_argument('--output-filename', '-of', help="Name of the file with statistics (see -tf)", type=str, default=RESULT_FILENAME)

args = parser.parse_args()

try:
    data = Data(args.filename)
    data.describe(args.to_file, args.output_filename)
except Exception as e:
    log.error(e)
    exit(1)
