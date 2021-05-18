import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from logger import *

TRUTH_FILENAME = "datasets/dataset_truth.csv"
PREDICTION_FILENAME = "houses.csv"

HOUSES = {
    'Gryffindor' : 0,
    'Hufflepuff' : 1,
    'Ravenclaw'  : 2,
    'Slytherin'  : 3
}

try:
    truth_df = pd.read_csv(TRUTH_FILENAME, sep=',', index_col=0)
    prediction_df = pd.read_csv(PREDICTION_FILENAME, sep=',', index_col=0)
except:
    log.error("One or both files (%s or %s) not found or invalid." % (TRUTH_FILENAME, PREDICTION_FILENAME))
    exit(1)

y_true = truth_df.replace(HOUSES).to_numpy().reshape((400, ))
y_pred = prediction_df.replace(HOUSES).to_numpy().reshape((400, ))

print("Your score on test set: {}".format(accuracy_score(y_true, y_pred)))
