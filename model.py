import pandas as pd
import numpy as np

COLUMN_Y = 'Hogwarts House'
COLUMNS_X = ['Herbology', 'Defense Against the Dark Arts', 'Divination', 'History of Magic']

TRAIN_ITER = 1000
LEARNING_RATE = 0.01

class Model():
    x = None
    y = None
    thetas = None

    def __init__(self, df):
        self.x = df[COLUMNS_X]
        # self.houses = df[COLUMN_Y].astype("category")
        # self.y = pd.DataFrame()
        self.y = df[COLUMN_Y].astype('category').cat.codes
        self.m = self.x.shape[0]
        self.n_classes = len(self.y.unique())
        self.guess = np.zeros([self.m, self.n_classes])
        # self.guess = pd.DataFrame(self.guess)

        for i in range(0, self.n_classes):
            for j in range(0, self.m):
                if self.y[j] == self.y.unique()[i]:
                    # self.guess.iloc[j, i] = 1
                    self.guess[j, i] = 1
                else:
                    # self.guess.iloc[j, i] = 0
                    self.guess[j, i] = 0

        print(self.guess)


        # print(self.n_classes)
        # print(self.houses)
        # print(self.x.shape[0])
        # print(self.x.T)
        # print(pd.concat([self.y, self.x], axis=1))
        # self.thetas = np.zeros(len(COLUMNS_X))
        # self._normalize()

    # def _normalize_feature(self, column):
        # self.normalization = pd.DataFrame(index=['std', 'mean'])
        # std = self.x[column].std()
        # mean = self.x[column].mean()
        # self.x[column] = (self.x[column] - mean) / std
        # self.normalization[column] = [std, mean]

    def _normalize(self):
        self.normalization = pd.DataFrame(index=['std', 'mean'])
        x_norm = pd.DataFrame()
        for column in self.x:
            # self._normalize_feature(column)
            std = self.x[column].std()
            mean = self.x[column].mean()
            x_norm[column] = (self.x[column] - mean) / std
            self.normalization[column] = [std, mean]
        self.x = x_norm
        # print(pd.concat([self.y, self.x], axis=1))

    def cost(self):
        guess = self.hypothesis(self.x)
        return (-1 / self.m) * (self.y * np.log(guess) + (1 - y) * np.log(1 - guess)).sum()

    def hypothesis(self, x):
        return 1 / (1 + np.exp(-np.dot(self.thetas, x)))

    # def fit(self):
    #     for _ in range(TRAIN_ITER):
    #         for _ in range(self.n_classes):

    #     return