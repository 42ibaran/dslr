import argparse
import pandas as pd

from logger import *
from model import Model

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str)
parser.add_argument('--cost-evolution', '-c', action='store_true')
parser.add_argument('--epochs', '-e', type=int)
parser.add_argument('--learning-rate', '-lr', type=float)
parser.add_argument('--features', '-f', nargs='+')
parser.add_argument('--features-select', '-fs', action='store_true') # NOT IMPL
parser.add_argument('--test', '-t', action='store_true') # NOT IMPL
parser.add_argument('--split-percent', '-sp', type=float, default=0.8) # NOT IMPL

args = parser.parse_args()

model = Model(args.filename, {
    'features': args.features,
    'features_select': args.features_select,
    'epochs': args.epochs,
    'learning_rate': args.learning_rate,
    'cost_evolution': args.cost_evolution,
    'test': args.test,
    'split_percent': args.split_percent,
})

model.fit()
model.save()
