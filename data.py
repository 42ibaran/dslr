import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns

from feature import Feature, STATS_ROWS

### Order: Gryffindor, Hufflepuff, Ravenclaw, Slytherin
# HOUSE_COLORS = ['#AE0001', '#FFDB00', '#222F5B', '#2A623D']
HOUSE_COLORS_DICT = {
    'Gryffindor' : '#AE0001',
    'Hufflepuff' : '#FFDB00',
    'Ravenclaw'  : '#222F5B',
    'Slytherin'  : '#2A623D'
}
GROUP_BY = 'Hogwarts House'

class Data():
    dataframe = None

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def get_numeric_columns(self):
        return self.dataframe.select_dtypes(include=[int, float])

    def get_grouped_by_house(self):
        return [x for _, x in self.dataframe.groupby([GROUP_BY])]

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
        tmpDF = self.get_numeric_columns()
        homogeneity_dict = {}

        for column in tmpDF:
            homogeneity_dict[column] = tmpDF[column].std()
        homogeneity_dict = {key: value for key, value in sorted(homogeneity_dict.items(), key=lambda item: item[1])}
        if all:
            return homogeneity_dict
        return next(iter(homogeneity_dict))

    # TODO: replace stats with custom methods
    def find_most_similar(self, all=False):
        tmpDF = self.get_numeric_columns()
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
        if all:
            return p_dict
        return next(iter(p_dict))

    def histogram_all(self):
        for column in self.get_numeric_columns():
            self.histogram_one(column)
            # self.dataframe[column]
        # fig, axes = plt.subplots(5, 3)
        # for i, column in enumerate(self.get_numeric_columns().columns):
        #     sns.histplot(self.dataframe[column], hue=self.dataframe[GROUP_BY], ax=axes[i//3,i%3])
        # sns.histplot()
        # df0, df1, df2, df3 = self.get_grouped_by_house()
        # fig, axs = plt.subplots(5, 3)
        # i = 0
        # j = 0

        # for column in self.get_numeric_columns():
        #     axs[i, j].hist(df0[column], 25, alpha=0.7, histtype='barstacked', color=HOUSE_COLORS[0])
        #     axs[i, j].hist(df1[column], 25, alpha=0.7, histtype='barstacked', color=HOUSE_COLORS[1])
        #     axs[i, j].hist(df2[column], 25, alpha=0.7, histtype='barstacked', color=HOUSE_COLORS[2])
        #     axs[i, j].hist(df3[column], 25, alpha=0.7, histtype='barstacked', color=HOUSE_COLORS[3])
        #     axs[i, j].set_title(column)
        #     j += 1
        #     if j == 3:
        #         i += 1
        #         j = 0
        # fig.delaxes(axs[4][1])
        # fig.delaxes(axs[4][2])
        # for ax in fig.get_axes():
        #     ax.label_outer()
        # plt.show()

    def histogram_one(self, feature):
        sns.histplot(self.dataframe, x=feature, bins=50, multiple="stack", hue=GROUP_BY, palette=HOUSE_COLORS_DICT)
        # df0, df1, df2, df3 = self.get_grouped_by_house()
        # _, axs = plt.subplots()

        # axs.hist(df0[feature], 25, alpha=0.7, histtype='barstacked', color=HOUSE_COLORS[0])
        # axs.hist(df1[feature], 25, alpha=0.7, histtype='barstacked', color=HOUSE_COLORS[1])
        # axs.hist(df2[feature], 25, alpha=0.7, histtype='barstacked', color=HOUSE_COLORS[2])
        # axs.hist(df3[feature], 25, alpha=0.7, histtype='barstacked', color=HOUSE_COLORS[3])
        plt.show()

    def scatter_plot(self, feat1, feat2):
        sns.scatterplot(x=self.dataframe[feat1], y=self.dataframe[feat2], hue=self.dataframe[GROUP_BY], palette=HOUSE_COLORS_DICT)
        # df0, df1, df2, df3 = self.get_grouped_by_house()
        # _, ax = plt.subplots()

        # ax.scatter(df0[feat1], df0[feat2], color=HOUSE_COLORS[0])
        # ax.scatter(df1[feat1], df1[feat2], color=HOUSE_COLORS[1])
        # ax.scatter(df2[feat1], df2[feat2], color=HOUSE_COLORS[2])
        # ax.scatter(df3[feat1], df3[feat2], color=HOUSE_COLORS[3])
        plt.show()

    def pair_plot(self):
        # df0, df1, df2, df3 = self.get_grouped_by_house()
        # _, axs = plt.subplots(13, 13)
        # i = 0
        
        sns.pairplot(self.dataframe, hue=GROUP_BY, palette=HOUSE_COLORS_DICT)
        # for column1 in self.get_numeric_columns():
        #     j = 0
        #     for column2 in self.get_numeric_columns():
        #         axs[i, j].scatter(df0[column1], df0[column2], alpha=0.7, color=HOUSE_COLORS[0])
        #         axs[i, j].scatter(df1[column1], df1[column2], alpha=0.7, color=HOUSE_COLORS[1])
        #         axs[i, j].scatter(df2[column1], df2[column2], alpha=0.7, color=HOUSE_COLORS[2])
        #         axs[i, j].scatter(df3[column1], df3[column2], alpha=0.7, color=HOUSE_COLORS[3])
        #         # axs[i, j].set_xlabel(column1)
        #         # axs[i, j].set_ylabel(column2)
        #         j += 1
        #     i += 1
        plt.show()
        return