import argparse
import pandas as pd

from logger import *
from model import Model

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str)
parser.add_argument('--cost', '-c', action='store_true')
parser.add_argument('--epochs', '-e', type=int)
parser.add_argument('--learning-rate', '-lr', type=float)
parser.add_argument('--features', '-f', nargs='+')

args = parser.parse_args()

try:
    datasetDF = pd.read_csv(args.filename, index_col=0)
except:
    log.error("Data file (%s) not found or invalid csv file." % args.filename)
    exit(1)

model = Model(datasetDF, args.features)
model.configure({
    'epochs': args.epochs,
    'learning_rate': args.learning_rate,
    'cost_evolution': args.cost
})

model.fit()
model.save()
