import argparse
import pandas as pd

from logger import *
from data import Data

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str)
parser.add_argument('-c', '--corner', action='store_true', help="display corner plot instead")

args = parser.parse_args()

try:
    data = Data(args.filename)
    data.pair_plot(corner=args.corner)
except Exception as e:
    log.error(e)
    exit(1)
