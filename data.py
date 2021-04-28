import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

from logger import *
from feature import Feature, STATS_ROWS

GROUP_BY = 'Hogwarts House'
HOUSE_COLORS_DICT = {
    'Gryffindor' : '#AE0001',
    'Hufflepuff' : '#FFDB00',
    'Ravenclaw'  : '#222F5B',
    'Slytherin'  : '#2A623D'
}

class Data():
    dataframe = None

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __get_numeric_columns(self):
        return self.dataframe.select_dtypes(include=[int, float])

    def describe(self, toFile=False, filename=None):
        statsDF = pd.DataFrame(index=STATS_ROWS)

        for column in self.dataframe:
            feature = Feature(self.dataframe[column])
            statsDF[column] = feature.get_statistics()

        if toFile:
            statsDF.to_csv(filename)
        else:
            print(statsDF)

    def find_most_homogeneous(self, all=False):
        tmpDF = self.__get_numeric_columns()
        homogeneity_dict = {}

        for column in tmpDF:
            homogeneity_dict[column] = tmpDF[column].std()
        homogeneity_dict = {key: value for key, value in sorted(homogeneity_dict.items(), key=lambda item: item[1])}
        if all:
            return homogeneity_dict
        return next(iter(homogeneity_dict))

    # TODO: replace stats with custom methods?
    def find_most_similar(self):
        tmpDF = self.__get_numeric_columns()
        p_dict = {}
        i = 0

        for column1 in tmpDF:
            j = 0
            x = tmpDF[column1]
            for column2 in tmpDF:
                y = tmpDF[column2]
                if j >= i:
                    continue
                sqrt1 = math.sqrt(((x - x.mean()) ** 2).sum())
                sqrt2 = math.sqrt(((y - y.mean()) ** 2).sum())
                key = (column1, column2)
                p = ((x - x.mean()) * (y - y.mean())).sum() / (sqrt1 * sqrt2)
                p_dict[key] = math.fabs(p)
                j += 1
            i += 1
        p_dict = {key: value for key, value in sorted(p_dict.items(), reverse=True, key=lambda item: item[1])}
        return next(iter(p_dict))

    def histogram_all(self, homogeneity_dict):
        for column in homogeneity_dict.keys():
            self.histogram_one(column)

    def histogram_one(self, feature):
        sns.histplot(self.dataframe, x=feature,
                     bins=50, multiple='stack', hue=GROUP_BY, palette=HOUSE_COLORS_DICT)
        plt.show()

    def scatter_plot(self, feat1, feat2):
        sns.scatterplot(x=self.dataframe[feat1], y=self.dataframe[feat2],
                        hue=self.dataframe[GROUP_BY], palette=HOUSE_COLORS_DICT)
        plt.show()

    def pair_plot(self):
        log.info("Creating a pair plot. It might take a while, please wait.")
        sns.pairplot(self.dataframe, corner=True,
                    diag_kind = 'hist', diag_kws = {'bins':25, 'multiple':'stack'},
                    hue=GROUP_BY, palette=HOUSE_COLORS_DICT)
        plt.subplots_adjust(bottom=0.04)
        plt.show()
