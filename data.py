import pandas as pd

from feature import Feature, STATS_ROWS

class Data():
    dataframe = None

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def describe(self, toFile=False, filename=None):
        statsDF = pd.DataFrame(index=STATS_ROWS)

        for column in self.dataframe:
            feature = Feature(self.dataframe[column])
            statsDF[column] = feature.getStats()

        if toFile:
            statsDF.to_csv(filename)
        else:
            print(statsDF)
