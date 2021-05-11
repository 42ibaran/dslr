import argparse
import pandas as pd

from logger import *
from model import Model

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str)
parser.add_argument('--on-features', '-f', nargs='+')

args = parser.parse_args()

try:
    datasetDF = pd.read_csv(args.filename, index_col=0)
except:
    log.error("Data file (%s) not found or invalid csv file." % args.filename)
    exit(1)

model = Model(datasetDF)

if args.on_features:
    model.set_training_features(args.on_features)

model.fit()
model.save()
