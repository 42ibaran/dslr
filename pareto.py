import argparse
import pandas as pd

from logger import *
from model import Model

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)

args = parser.parse_args()

try:
    datasetDF = pd.read_csv(args.filename, index_col=0)
except:
    log.error("Data file (%s) not found or invalid csv file." % args.filename)
    exit(1)

i100 = datasetDF.shape[0]
i80 = int(i100 * 0.8)

df80 = datasetDF[:i80]
df20 = datasetDF[i80:]

filename = args.filename
ext_pos = filename.find(".csv")

filename80 = filename[:ext_pos] + '_80' + filename[ext_pos:]
filename20 = filename[:ext_pos] + '_20' + filename[ext_pos:]

df80.to_csv(filename80)
df20.to_csv(filename20)
