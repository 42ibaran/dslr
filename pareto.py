import argparse
import pandas as pd

from logger import *
from model import Model

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str)
parser.add_argument('--percentage', '-p', help="Percentage of dataset to put in training set.", type=int, default=80)

args = parser.parse_args()

if args.percentage > 100 or args.percentage < 0:
    log.error("Percentage should be between 0 and 100.")
    exit(1)

try:
    datasetDF = pd.read_csv(args.filename, index_col=0)
except:
    log.error("Data file (%s) not found or invalid csv file." % args.filename)
    exit(1)

size = datasetDF.shape[0]
index_big = int(size * args.percentage / 100)

df_big = datasetDF[:index_big]
df_small = datasetDF[index_big:]

filename = args.filename
ext_pos = filename.find(".csv")

filename_big = filename[:ext_pos] + '_' + str(args.percentage) + filename[ext_pos:]
filename_small = filename[:ext_pos] + '_' + str(100 - args.percentage) + filename[ext_pos:]

df_big.to_csv(filename_big)
df_small.to_csv(filename_small)
