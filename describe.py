import argparse
import pandas as pd
import numpy as np

from logger import *
import statistics as st

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

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
args = parser.parse_args()

try:
    df = pd.read_csv(args.filename, index_col=0)
except:
    log.error("Data file (%s) not found or invalid csv file." % args.filename)
    exit(1)

df = df.select_dtypes(include=[int, float])
df1 = pd.DataFrame(index=['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'])

for column in df:
    df1[column] = getStats(df[column])

print(df1)
df1.to_csv("toto.csv")
