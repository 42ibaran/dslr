import pandas as pd
import matplotlib.pyplot as plt

from feature import Feature, STATS_ROWS

HOUSE_COLORS = ['#740001', '#1A472A', '#0E1A40', '#FFDB00']

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
            axs[i, j].hist(df0[column], alpha=0.7, color=HOUSE_COLORS[0])
            axs[i, j].hist(df1[column], alpha=0.7, color=HOUSE_COLORS[1])
            axs[i, j].hist(df2[column], alpha=0.7, color=HOUSE_COLORS[2])
            axs[i, j].hist(df3[column], alpha=0.7, color=HOUSE_COLORS[3])
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
            

    def scatter_plot(self):
        return
        
    def pair_plot(self):
        return