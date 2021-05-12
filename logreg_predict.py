import argparse
import pandas as pd

from logger import *
from model import Model, MODE_PREDICT

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str)
parser.add_argument("--test", "-t", action='store_true')

args = parser.parse_args()

try:
    datasetDF = pd.read_csv(args.filename, index_col=0)
except:
    log.error("Data file (%s) not found or invalid csv file." % args.filename)
    exit(1)

model = Model(args.filename, {
    'mode': MODE_PREDICT
})
model.predict()
