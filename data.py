import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from feature import Feature, STATS_ROWS

### Gryffindor, Hufflepuff, Ravenclaw, Slytherin (alphabetical)
HOUSE_COLORS = ['#AE0001', '#FFDB00', '#222F5B', '#2A623D']

class Data():
    dataframe = None

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def describe(self, toFile=False, filename=None):
        statsDF = pd.DataFrame(index=STATS_ROWS)

        for column in self.dataframe:
            feature = Feature(self.dataframe[column])
            statsDF[column] = feature.get_statistics()

        if toFile:
            statsDF.to_csv(filename)
        else:
            print(statsDF)

    def histogram(self):
        fig, axs = plt.subplots(5, 3)
        
        i = 0
        j = 0

        df0, df1, df2, df3 = [x for _, x in self.dataframe.groupby(['Hogwarts House'])]
        for column in self.dataframe.select_dtypes(include=[int, float]):
            axs[i, j].hist(df0[column], 25, alpha=0.7, histtype='barstacked', color=HOUSE_COLORS[0])
            axs[i, j].hist(df1[column], 25, alpha=0.7, histtype='barstacked', color=HOUSE_COLORS[1])
            axs[i, j].hist(df2[column], 25, alpha=0.7, histtype='barstacked', color=HOUSE_COLORS[2])
            axs[i, j].hist(df3[column], 25, alpha=0.7, histtype='barstacked', color=HOUSE_COLORS[3])
            axs[i, j].set_title(column)

            j += 1
            if j == 3:
                i += 1
                j = 0
        
        fig.delaxes(axs[4][1])
        fig.delaxes(axs[4][2])
        
        for ax in fig.get_axes():
            ax.label_outer()
        plt.show()
            
    # TODO: replace stats with custom methods
    def find_most_similar(self):
        tmpDF = self.dataframe.select_dtypes(include=[int, float])
        feature1 = ""
        feature2 = ""

        p_max = 0
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
                p = ((x - x.mean()) * (y - y.mean())).sum() / (sqrt1 * sqrt2)

                if math.fabs(p) > p_max:
                    p_max = math.fabs(p)
                    feature1 = column1
                    feature2 = column2
                j += 1
            i += 1
        
        return (feature1, feature2)

    def scatter_plot(self, x, y):
        df0, df1, df2, df3 = [x for _, x in self.dataframe.groupby(['Hogwarts House'])]

        _, ax = plt.subplots()
        ax.scatter(df0[x], df0[y], color=HOUSE_COLORS[0])
        ax.scatter(df1[x], df1[y], color=HOUSE_COLORS[1])
        ax.scatter(df2[x], df2[y], color=HOUSE_COLORS[2])
        ax.scatter(df3[x], df3[y], color=HOUSE_COLORS[3])
        plt.grid()
        plt.show()

    def pair_plot(self):
        return