import argparse
import pandas as pd
import numpy as np

from logger import *
import statistics as st

RESULT_FILENAME = "describe.csv"

def getStats(x):
    return [
        st.count(x),
        st.mean(x),
        st.std(x),
        st.min(x),
        st.percentile(x, 25),
        st.percentile(x, 50),
        st.percentile(x, 75),
        st.max(x)
    ]

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
parser.add_argument("-f", help="Save statistics to file", action='store_true')
parser.add_argument("-n", help="Name of the file with statistics (see -f)", type=str, default=RESULT_FILENAME)

args = parser.parse_args()

try:
    datasetDF = pd.read_csv(args.filename, index_col=0)
except:
    log.error("Data file (%s) not found or invalid csv file." % args.filename)
    exit(1)

datasetDF = datasetDF.select_dtypes(include=[int, float])
statsDF = pd.DataFrame(index=['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'])

for column in datasetDF:
    statsDF[column] = getStats(datasetDF[column])

if args.f:
    statsDF.to_csv(args.n)
else:
    print(statsDF)
