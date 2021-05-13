import argparse
import pandas as pd

from logger import *
from model import Model

parser = argparse.ArgumentParser()
parser.add_argument('filename',
                    type=str,
                    help="csv file with dataset for training")
parser.add_argument('-c',  '--cost-evolution',
                    action='store_true', 
                    help="display cost evolution after training")
parser.add_argument('-e',  '--epochs',
                    type=int,
                    help="number of iterations for training")
parser.add_argument('-lr', '--learning-rate',
                    type=float,
                    help="learning rate for training")
parser.add_argument('-f',  '--features',
                    nargs='+',
                    help="list of features for training")
parser.add_argument('-fs', '--features-select',
                    action='store_true',
                    help="display feature selector. -f option will be ignored")
parser.add_argument('-t',  '--test',
                    action='store_true',
                    help="display accuracy statistics after training. will split provided dataset by Pareto rule")
parser.add_argument('-sp', '--split-percent',
                    type=float,
                    help="fraction of dataset to be used for training (see --test)")
parser.add_argument('-s',  '--seed',
                    type=int,
                    help="seed to use for splitting dataset (see --test)")
parser.add_argument('-v',  '--verbose',
                    type=int,
                    help="verbose level")
args = parser.parse_args()

try:
    model = Model(args.filename, {
        'features': args.features,
        'features_select': args.features_select,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'cost_evolution': args.cost_evolution,
        'test': args.test,
        'split_percent': args.split_percent,
        'seed': args.seed,
        'verbose': args.verbose,
    })
    model.fit()
except Exception as e:
    log.error(e)
    exit(1)
