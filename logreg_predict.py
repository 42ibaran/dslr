import argparse
import pandas as pd

from logger import *
from model import Model, MODE_PREDICT

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str)

args = parser.parse_args()

# try:
model = Model(args.filename, {
    'mode': MODE_PREDICT
})
model.predict()
# except Exception as e:
    # log.error(e)
    # exit(1)
